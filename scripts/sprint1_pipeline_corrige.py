# =============================================================================
# SPRINT 1 : pipeline corrigé (PatchCore / Anomalib / MVTec AD)
# Chaque "# %%" = une cellule Kaggle. Colle-les dans l'ordre.
# =============================================================================

# %% CELLULE 0 : installation (à lancer une fois par session)
# !pip install -q anomalib kornia FrEIA open-clip-torch lightning torchmetrics kagglehub

# %% CELLULE 1 : configuration + tirage ALÉATOIRE de la catégorie
import os, json, random, time, shutil, warnings
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import kagglehub

warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"

TOUTES_CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                     "leather", "metal_nut", "pill", "screw", "tile",
                     "toothbrush", "transistor", "wood", "zipper"]
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}   # pas de "pièce" isolée

CATEGORIE_FORCEE = None   # ex: "metal_nut" pour figer la catégorie, None = aléatoire
SEED = None               # None = nouvelle graine à chaque exécution
SEUIL_PIXEL_RATIO_TROU = 0.005   # un trou interne > 0,5 % de l'image = vrai trou (ex: centre de l'écrou)

seed_run = SEED if SEED is not None else int.from_bytes(os.urandom(4), "little")
rng = random.Random(seed_run)
np.random.seed(seed_run); torch.manual_seed(seed_run)

path_mvtec = kagglehub.dataset_download("ipythonx/mvtec-ad")
DATASET_ROOT = None
for root, dirs, _ in os.walk(path_mvtec):
    if any(c in dirs for c in TOUTES_CATEGORIES):
        DATASET_ROOT = Path(root); break
assert DATASET_ROOT is not None, "Dossier racine MVTec introuvable"

disponibles = [c for c in TOUTES_CATEGORIES if (DATASET_ROOT / c).is_dir()]

# Historique : on ne rejoue pas une catégorie déjà vue tant que toutes ne sont pas passées
FICHIER_HISTO = Path("/kaggle/working/historique_categories.txt")
deja_vues = FICHIER_HISTO.read_text().split() if FICHIER_HISTO.exists() else []
restantes = [c for c in disponibles if c not in deja_vues] or disponibles
if not [c for c in disponibles if c not in deja_vues]:
    deja_vues = []                                           # cycle complet -> on repart à zéro

CATEGORIE = CATEGORIE_FORCEE or rng.choice(restantes)
FICHIER_HISTO.write_text("\n".join(deja_vues + [CATEGORIE]))

OUT_DIR = Path(f"/kaggle/working/resultats_{CATEGORIE}_{seed_run}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Graine : {seed_run} | Catégorie tirée : {CATEGORIE.upper()}")
print(f"Déjà traitées dans cette session : {deja_vues}")
print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun")

# %% CELLULE 2 : données + entraînement PatchCore
from anomalib.models import Patchcore
from anomalib.engine import Engine
try:
    from anomalib.data import MVTecAD as MVTecDataset
except ImportError:
    from anomalib.data import MVTec as MVTecDataset

datamodule = MVTecDataset(root=DATASET_ROOT, category=CATEGORIE,
                          train_batch_size=32, eval_batch_size=32, num_workers=2)
datamodule.setup()

model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                  pre_trained=True, coreset_sampling_ratio=0.05)
engine = Engine(default_root_dir=str(OUT_DIR / "anomalib"),
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1, enable_progress_bar=False)

print("Entraînement (construction de la mémoire PatchCore)...")
engine.fit(datamodule=datamodule, model=model)
print("Inférence sur le jeu de test...")
predictions = engine.predict(datamodule=datamodule, model=model)   # UN SEUL appel
print("Lots prédits :", len(predictions))

# %% CELLULE 3 : collecte unique des prédictions dans une liste de dictionnaires
def extraire_champ(batch, noms):
    for nom in noms:
        if hasattr(batch, nom) and getattr(batch, nom) is not None:
            return getattr(batch, nom)
        if isinstance(batch, dict) and batch.get(nom) is not None:
            return batch[nom]
    return None

def vers_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

records, INPUT_HW = [], None
for batch in predictions:
    chemins = extraire_champ(batch, ["image_path", "image_paths"])
    hms = vers_np(extraire_champ(batch, ["anomaly_map", "anomaly_maps"]))
    pms = vers_np(extraire_champ(batch, ["pred_mask", "pred_masks"]))
    gts = extraire_champ(batch, ["gt_mask", "mask", "masks"])
    gts = vers_np(gts) if gts is not None else np.zeros_like(pms)
    labels = vers_np(extraire_champ(batch, ["gt_label", "label", "labels"]))
    scores = vers_np(extraire_champ(batch, ["pred_score", "pred_scores"]))
    plabels = vers_np(extraire_champ(batch, ["pred_label", "pred_labels"]))
    img_t = extraire_champ(batch, ["image"])
    if INPUT_HW is None and img_t is not None:
        INPUT_HW = tuple(img_t.shape[-2:])
    for i in range(len(chemins)):
        records.append(dict(
            path=str(chemins[i]), type=Path(str(chemins[i])).parent.name,
            label=int(labels[i]), score=float(scores[i]), pred_label=int(plabels[i]),
            hm=np.squeeze(hms[i]).astype(np.float32),
            pm=(np.squeeze(pms[i]) > 0).astype(np.uint8),
            gt=(np.squeeze(gts[i]) > 0).astype(np.uint8)))

H, W = records[0]["hm"].shape[:2]
INPUT_HW = INPUT_HW or (H, W)
print(f"{len(records)} images de test | cartes {H}x{W} | entrée modèle {INPUT_HW}")

# %% CELLULE 4 : segmentation de l'objet (CORRECTIF 1 : Otsu + polarité + textures)
def charger_image_alignee(chemin, H, W, taille_resize=256):
    """Reproduit la géométrie d'Anomalib : Resize(256) puis CenterCrop(H,W).
    Sans cela, le masque objet est décalé par rapport à la heatmap et au GT."""
    bgr = cv2.imread(chemin)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if H <= taille_resize and W <= taille_resize:
        rgb = cv2.resize(rgb, (taille_resize, taille_resize), interpolation=cv2.INTER_AREA)
        y0, x0 = (taille_resize - H) // 2, (taille_resize - W) // 2
        return rgb[y0:y0 + H, x0:x0 + W]
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

def segmenter_objet(image_rgb, categorie):
    h, w = image_rgb.shape[:2]
    if categorie in TEXTURES:                       # toute l'image est la surface inspectée
        return np.ones((h, w), np.uint8)
    gris = cv2.GaussianBlur(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bord = np.concatenate([binaire[0], binaire[-1], binaire[:, 0], binaire[:, -1]])
    if np.mean(bord > 0) > 0.5:                     # fond clair -> on inverse (objet = 255)
        binaire = 255 - binaire
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binaire = cv2.morphologyEx(binaire, cv2.MORPH_CLOSE, k)
    contours, hier = cv2.findContours(binaire, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    masque = np.zeros((h, w), np.uint8)
    if not contours:
        return masque + 1
    idx = int(np.argmax([cv2.contourArea(c) for c in contours]))
    cv2.drawContours(masque, contours, idx, 1, -1)
    for i, c in enumerate(contours):                # on retire uniquement les vrais trous de l'objet
        if hier[0][i][3] == idx and cv2.contourArea(c) > SEUIL_PIXEL_RATIO_TROU * h * w:
            cv2.drawContours(masque, contours, i, 0, -1)
    return masque

for r in records:
    r["obj"] = segmenter_objet(charger_image_alignee(r["path"], H, W), CATEGORIE)

# %% CELLULE 5 : métriques (CORRECTIF 2 : IoU/Dice BRUT et FILTRÉ rapportés ensemble)
K_BORD = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def iou_dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter, union, s = (a & b).sum(), (a | b).sum(), a.sum() + b.sum()
    return (inter / union if union > 0 else 1.0), (2 * inter / s if s > 0 else 1.0)

def metriques(pred, gt, obj):
    pred = (pred > 0).astype(np.uint8)
    pred_f = pred & cv2.dilate(obj, K_BORD)         # filtre "hors pièce"
    iou_b, dice_b = iou_dice(pred, gt)
    iou_f, dice_f = iou_dice(pred_f, gt)
    n_obj = obj.sum()
    return dict(iou_brut=iou_b, dice_brut=dice_b, iou_filtre=iou_f, dice_filtre=dice_f,
                pct_pred=100.0 * pred_f.sum() / n_obj if n_obj else 0.0,
                pct_gt=100.0 * (gt.astype(bool) & obj.astype(bool)).sum() / n_obj if n_obj else 0.0,
                pred_filtre=pred_f)

# %% CELLULE 6 : séparation calibration / évaluation (CORRECTIF 3 : fuite de seuil)
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             accuracy_score, precision_recall_curve, confusion_matrix)

def split_stratifie(recs, ratio=0.5):
    calib, evalu = [], []
    for lab in (0, 1):
        groupe = [r for r in recs if r["label"] == lab]
        rng.shuffle(groupe)
        n = int(len(groupe) * ratio)
        calib += groupe[:n]; evalu += groupe[n:]
    return calib, evalu

calib, evalu = split_stratifie(records)
print(f"Calibration : {len(calib)} images | Évaluation (jamais vue pour les seuils) : {len(evalu)}")

# Seuil image : maximise le F1 sur la calibration UNIQUEMENT
y_c = np.array([r["label"] for r in calib]); s_c = np.array([r["score"] for r in calib])
if len(set(y_c)) == 2:
    p, rc, t = precision_recall_curve(y_c, s_c)
    f1 = 2 * p * rc / (p + rc + 1e-12)
    SEUIL_IMG = float(t[int(np.argmax(f1[:-1]))])
else:
    SEUIL_IMG = None
    print("Calibration mono-classe : seuil image Anomalib par défaut conservé.")

# Seuil pixel : maximise le Dice (filtré) moyen sur les images défectueuses de calibration
def dice_moyen(recs, t):
    return np.mean([metriques(r["hm"] > t, r["gt"], r["obj"])["dice_filtre"]
                    for r in recs if r["label"] == 1])

defect_c = [r for r in calib if r["label"] == 1]
valeurs = np.concatenate([r["hm"].ravel() for r in defect_c])
candidats = np.quantile(valeurs, np.linspace(0.80, 0.999, 40))
dices = [dice_moyen(defect_c, t) for t in candidats]
SEUIL_PIX = float(candidats[int(np.argmax(dices))])
print(f"Seuil image calibré : {SEUIL_IMG} | Seuil pixel calibré : {SEUIL_PIX:.4f} "
      f"(Dice calibration = {max(dices):.3f})")

# %% CELLULE 7 : évaluation honnête sur le jeu d'ÉVALUATION
y_e = np.array([r["label"] for r in evalu]); s_e = np.array([r["score"] for r in evalu])
y_def = np.array([r["pred_label"] for r in evalu])
y_cal = (s_e >= SEUIL_IMG).astype(int) if SEUIL_IMG is not None else y_def

def rapport_image(y, yhat):
    return dict(f1=f1_score(y, yhat), acc=accuracy_score(y, yhat),
                prec=precision_score(y, yhat, zero_division=0),
                rec=recall_score(y, yhat, zero_division=0))

auroc_img = roc_auc_score(y_e, s_e) if len(set(y_e)) == 2 else float("nan")
y_pix = np.concatenate([r["gt"].ravel() for r in evalu])
s_pix = np.concatenate([r["hm"].ravel() for r in evalu])
auroc_pix = roc_auc_score(y_pix, s_pix) if len(set(y_pix)) == 2 else float("nan")

res_def = [metriques(r["pm"], r["gt"], r["obj"]) for r in evalu if r["label"] == 1]
res_cal = [metriques(r["hm"] > SEUIL_PIX, r["gt"], r["obj"]) for r in evalu if r["label"] == 1]
types = [r["type"] for r in evalu if r["label"] == 1]

def moy(res, k): return float(np.mean([x[k] for x in res]))
mae = lambda res: float(np.mean([abs(x["pct_pred"] - x["pct_gt"]) for x in res]))

print("=" * 72)
print(f"ÉVALUATION SUR {len(evalu)} IMAGES NON UTILISÉES POUR LES SEUILS ({CATEGORIE})")
print("=" * 72)
print(f"AUROC image : {auroc_img:.4f} | AUROC pixel : {auroc_pix:.4f}")
for nom, yh in (("seuil Anomalib", y_def), ("seuil calibré ", y_cal)):
    m = rapport_image(y_e, yh)
    print(f"[{nom}] F1 {m['f1']:.3f} | Acc {m['acc']:.3f} | Préc {m['prec']:.3f} | Rappel {m['rec']:.3f}")
print("-" * 72)
print(f"{'':28s}{'IoU brut':>10s}{'IoU filtré':>12s}{'Dice brut':>11s}{'Dice filtré':>13s}{'MAE %':>8s}")
for nom, res in (("Masque Anomalib (défaut)", res_def), ("Seuil pixel calibré", res_cal)):
    print(f"{nom:28s}{moy(res,'iou_brut'):10.3f}{moy(res,'iou_filtre'):12.3f}"
          f"{moy(res,'dice_brut'):11.3f}{moy(res,'dice_filtre'):13.3f}{mae(res):8.2f}")
print("-" * 72)
print("Par type de défaut (seuil calibré) :")
for t in sorted(set(types)):
    sel = [x for x, ty in zip(res_cal, types) if ty == t]
    print(f"  {t:14s} n={len(sel):2d} | IoU filtré {moy(sel,'iou_filtre'):.3f} | "
          f"Dice {moy(sel,'dice_filtre'):.3f} | MAE surface {mae(sel):.2f} %")
print("=" * 72)
print(confusion_matrix(y_e, y_cal), "<- matrice de confusion (seuil calibré)")

# %% CELLULE 8 : images d'étude TIRÉES AU HASARD (+ contours pour valider la segmentation)
def afficher_exemples(n=5):
    defect_eval = [r for r in evalu if r["label"] == 1]
    choix = rng.sample(defect_eval, min(n, len(defect_eval)))
    for j, r in enumerate(choix, 1):
        img = charger_image_alignee(r["path"], H, W)
        m = metriques(r["hm"] > SEUIL_PIX, r["gt"], r["obj"])
        cont = img.copy()
        c_obj, _ = cv2.findContours(r["obj"] * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        c_def, _ = cv2.findContours(m["pred_filtre"] * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_gt, _ = cv2.findContours(r["gt"] * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cont, c_obj, -1, (0, 255, 0), 1)     # vert  : objet segmenté
        cv2.drawContours(cont, c_def, -1, (255, 0, 0), 2)     # rouge : défaut prédit
        cv2.drawContours(cont, c_gt, -1, (0, 128, 255), 1)    # bleu  : vérité terrain
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"{CATEGORIE}/{r['type']} | surface {m['pct_pred']:.2f}% (réel {m['pct_gt']:.2f}%) | "
                     f"IoU {m['iou_filtre']:.3f} (brut {m['iou_brut']:.3f}) | Dice {m['dice_filtre']:.3f}",
                     fontweight="bold")
        ax[0].imshow(img); ax[0].set_title("Image")
        ax[1].imshow(img); ax[1].imshow(r["hm"], cmap="jet", alpha=0.5); ax[1].set_title("Heatmap")
        ax[2].imshow(m["pred_filtre"], cmap="gray"); ax[2].set_title("Masque prédit")
        ax[3].imshow(cont); ax[3].set_title("Vert=objet | Rouge=prédit | Bleu=GT")
        for a in ax: a.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"exemple_{j}_{r['type']}.png", dpi=150)
        plt.show()

afficher_exemples(5)
# À VÉRIFIER À L'ŒIL : le contour vert colle-t-il à la pièce ? Sinon le recadrage
# (Resize/CenterCrop) de charger_image_alignee() ne correspond pas à ta version d'Anomalib.

# %% CELLULE 9 : latence propre (CORRECTIF 4 : forward seul, warm-up, synchronize)
def mesurer_latence(model, hw, n_warmup=20, n_iter=100):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.model.to(dev).eval()
    x = torch.randn(1, 3, *hw, device=dev)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    with torch.inference_mode():
        for _ in range(n_warmup): net(x)
        sync(); t0 = time.perf_counter()
        for _ in range(n_iter): net(x)
        sync(); dt = time.perf_counter() - t0
    return 1000 * dt / n_iter

try:
    ms = mesurer_latence(model, INPUT_HW)
    print(f"Latence forward (batch=1, {INPUT_HW}) : {ms:.1f} ms/image ({1000/ms:.1f} FPS)")
    print("Hors prétraitement, chargement disque et post-traitement.")
except Exception as e:
    ms = None
    print("Mesure impossible sur cette version d'Anomalib :", e)

# %% CELLULE 10 : export des résultats de l'exécution (traçabilité)
resume = dict(
    graine=seed_run, categorie=CATEGORIE, n_test=len(records),
    n_calibration=len(calib), n_evaluation=len(evalu),
    seuil_image=SEUIL_IMG, seuil_pixel=SEUIL_PIX,
    auroc_image=auroc_img, auroc_pixel=auroc_pix,
    iou_filtre=moy(res_cal, "iou_filtre"), iou_brut=moy(res_cal, "iou_brut"),
    dice_filtre=moy(res_cal, "dice_filtre"), dice_brut=moy(res_cal, "dice_brut"),
    mae_surface_pct=mae(res_cal), latence_ms=ms)
(OUT_DIR / "resume.json").write_text(json.dumps(resume, indent=2, default=float))
shutil.make_archive(str(OUT_DIR), "zip", OUT_DIR)
print("Archive :", f"{OUT_DIR}.zip")
