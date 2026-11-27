"""Interface de visualisation mono-écran pour PatchCore + OpenCV (Sprint 2)."""
import json
from pathlib import Path
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Inspection Visuelle Industrielle",
    layout="wide",
)


def decouvrir_packs(base_dirs):
    """Recherche tous les dossiers contenant manifest.json sans valeurs codées en dur."""
    packs = []
    vus = set()
    for base in base_dirs:
        p = Path(base)
        if not p.exists():
            continue
        for mf in sorted(p.rglob("manifest.json")):
            resolved = str(mf.resolve())
            if resolved not in vus:
                vus.add(resolved)
                packs.append(mf.parent)
    return packs


@st.cache_data
def charger_manifest(manifest_path_str):
    return json.loads(Path(manifest_path_str).read_text(encoding="utf-8"))


def calculer_stats_dynamiques(hm, gt, obj, seuil_pix):
    pred_bin = (hm > seuil_pix).astype(np.uint8)
    k_bord = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    pred_f = pred_bin & cv2.dilate(obj, k_bord)
    a, b = pred_f.astype(bool), gt.astype(bool)
    inter, union, s = (a & b).sum(), (a | b).sum(), a.sum() + b.sum()
    iou_f = float(inter / union) if union > 0 else 1.0
    dice_f = float(2 * inter / s) if s > 0 else 1.0
    n_obj = float(obj.sum())
    pct_pred = float(100.0 * pred_f.sum() / n_obj) if n_obj > 0 else 0.0
    pct_gt = float(100.0 * (a & b).sum() / n_obj) if n_obj > 0 else 0.0
    return pred_f, iou_f, dice_f, pct_pred, pct_gt


def construire_contours(img_rgb, obj, pred_f, gt):
    out = img_rgb.copy()
    c_obj, _ = cv2.findContours((obj * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    c_def, _ = cv2.findContours((pred_f * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_gt, _ = cv2.findContours((gt * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, c_obj, -1, (0, 255, 0), 1)
    cv2.drawContours(out, c_def, -1, (255, 0, 0), 2)
    cv2.drawContours(out, c_gt, -1, (0, 128, 255), 1)
    return out


def main():
    packs = decouvrir_packs([".", "/kaggle/working", "results"])
    if not packs:
        st.warning("Aucun `manifest.json` détecté. Exécutez d'abord l'export du pipeline.")
        return

    # Barre de contrôle compacte en haut (mono-écran)
    c_run, c_type, c_stat, c_img = st.columns([2.2, 1.4, 1.3, 2.5])

    with c_run:
        pack_choisi = st.selectbox(
            "Exécution exportée",
            options=packs,
            format_func=lambda p: f"{p.parent.name} ({p.name})",
        )

    data = charger_manifest(str(pack_choisi / "manifest.json"))
    echantillons = data.get("echantillons", [])
    resume = data.get("resume", {})

    if not echantillons:
        st.info("Aucun échantillon trouvé dans ce manifeste.")
        return

    types_dispo = ["Tous"] + sorted({e["type"] for e in echantillons})
    statuts_dispo = ["Tous"] + sorted({e["statut"] for e in echantillons})

    with c_type:
        type_filtre = st.selectbox("Type / Dossier", options=types_dispo)
    with c_stat:
        statut_filtre = st.selectbox("Statut", options=statuts_dispo)

    filtres = [
        e for e in echantillons
        if (type_filtre == "Tous" or e["type"] == type_filtre)
        and (statut_filtre == "Tous" or e["statut"] == statut_filtre)
    ] or echantillons

    with c_img:
        sample = st.selectbox(
            "Échantillon d'évaluation",
            options=filtres,
            format_func=lambda e: f"#{e['id']:03d} | {e['type']} | {e['statut']} | score={e['score']:.3f}",
        )

    seuil_pix_defaut = float(data.get("seuil_pixel") if data.get("seuil_pixel") is not None else 0.5)
    seuil_img = data.get("seuil_image")

    hm = np.load(str(pack_choisi / f"{sample['prefix']}_hm.npy"))
    hm_min = float(data.get("hm_min", hm.min()))
    hm_max = float(data.get("hm_max", hm.max()))
    if hm_max <= hm_min:
        hm_max = hm_min + 1e-6

    seuil_pix = st.slider(
        "Seuil pixel (heatmap)",
        min_value=hm_min,
        max_value=hm_max,
        value=float(np.clip(seuil_pix_defaut, hm_min, hm_max)),
        step=float((hm_max - hm_min) / 200.0),
    )

    # Indicateurs globaux dynamiques depuis resume.json
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Catégorie", str(data.get("categorie", "-")).upper())
    k2.metric("AUROC Image", f"{resume['auroc_image']:.4f}" if resume.get("auroc_image") is not None else "N/A")
    k3.metric("Dice Filtré (moy.)", f"{resume['dice_filtre']:.3f}" if resume.get("dice_filtre") is not None else "N/A")
    k4.metric("MAE Surface", f"{resume['mae_surface_pct']:.2f} %" if resume.get("mae_surface_pct") is not None else "N/A")
    k5.metric("Latence Forward", f"{resume['latence_ms']:.1f} ms" if resume.get("latence_ms") is not None else "N/A")

    # Chargement des calques de l'échantillon sélectionné
    img_rgb = cv2.cvtColor(cv2.imread(str(pack_choisi / f"{sample['prefix']}_orig.png")), cv2.COLOR_BGR2RGB)
    gt_u8 = (cv2.imread(str(pack_choisi / f"{sample['prefix']}_gt.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    obj_u8 = (cv2.imread(str(pack_choisi / f"{sample['prefix']}_obj.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

    pred_f, iou_f, dice_f, pct_pred, pct_gt = calculer_stats_dynamiques(hm, gt_u8, obj_u8, seuil_pix)

    hm_norm = np.clip((hm - hm_min) / (hm_max - hm_min), 0.0, 1.0)
    hm_jet_rgb = cv2.cvtColor(cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.addWeighted(img_rgb, 0.55, hm_jet_rgb, 0.45, 0)
    contours_rgb = construire_contours(img_rgb, obj_u8, pred_f, gt_u8)

    # Métriques de l'image courante
    m1, m2, m3, m4, m5 = st.columns(5)
    seuil_txt = f"{seuil_img:.3f}" if seuil_img is not None else "défaut"
    m1.metric("Score Anomalie", f"{sample['score']:.3f}", delta=f"seuil={seuil_txt}", delta_color="off")
    m2.metric("Surface Prédite", f"{pct_pred:.2f} %")
    m3.metric("Surface Réelle (GT)", f"{pct_gt:.2f} %")
    m4.metric("Dice Filtré", f"{dice_f:.3f}")
    m5.metric("IoU Filtré", f"{iou_f:.3f}")

    # Grille 4 vues alignées sur l'écran unique
    v1, v2, v3, v4 = st.columns(4)
    v1.image(img_rgb, caption="Image alignée", use_container_width=True)
    v2.image(overlay_rgb, caption="Heatmap superposée", use_container_width=True)
    v3.image(pred_f * 255, caption="Masque défaut filtré", clamp=True, use_container_width=True)
    v4.image(contours_rgb, caption="Vert=Objet | Rouge=Prédit | Bleu=GT", use_container_width=True)


if __name__ == "__main__":
    main()
