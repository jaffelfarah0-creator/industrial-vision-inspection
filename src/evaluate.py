"""Collecte des prédictions Anomalib et métriques (IoU/Dice brut et filtré, surface, AUROC)."""
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

K_BORD = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def extraire_champ(batch, noms):
    for nom in noms:
        if hasattr(batch, nom) and getattr(batch, nom) is not None:
            return getattr(batch, nom)
        if isinstance(batch, dict) and batch.get(nom) is not None:
            return batch[nom]
    return None


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def collecter(predictions):
    """Transforme la sortie de engine.predict en liste de dictionnaires + taille d'entrée du modèle."""
    records, input_hw = [], None
    for b in predictions:
        chemins = extraire_champ(b, ["image_path", "image_paths"])
        hms = _np(extraire_champ(b, ["anomaly_map", "anomaly_maps"]))
        pms = _np(extraire_champ(b, ["pred_mask", "pred_masks"]))
        gts = extraire_champ(b, ["gt_mask", "mask", "masks"])
        gts = _np(gts) if gts is not None else np.zeros_like(pms)
        labels = _np(extraire_champ(b, ["gt_label", "label", "labels"]))
        scores = _np(extraire_champ(b, ["pred_score", "pred_scores"]))
        plabels = _np(extraire_champ(b, ["pred_label", "pred_labels"]))
        img = extraire_champ(b, ["image"])
        if input_hw is None and img is not None:
            input_hw = tuple(img.shape[-2:])
        for i in range(len(chemins)):
            records.append(dict(
                path=str(chemins[i]), type=Path(str(chemins[i])).parent.name,
                label=int(labels[i]), score=float(scores[i]), pred_label=int(plabels[i]),
                hm=np.squeeze(hms[i]).astype(np.float32),
                pm=(np.squeeze(pms[i]) > 0).astype(np.uint8),
                gt=(np.squeeze(gts[i]) > 0).astype(np.uint8)))
    return records, input_hw or records[0]["hm"].shape[:2]


def iou_dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter, union, s = (a & b).sum(), (a | b).sum(), a.sum() + b.sum()
    return (inter / union if union > 0 else 1.0), (2 * inter / s if s > 0 else 1.0)


def metriques(pred, gt, obj):
    """IoU/Dice bruts et filtrés (masque objet dilaté), % de surface prédit et réel."""
    pred = (pred > 0).astype(np.uint8)
    pred_f = pred & cv2.dilate(obj, K_BORD)
    iou_b, dice_b = iou_dice(pred, gt)
    iou_f, dice_f = iou_dice(pred_f, gt)
    n = obj.sum()
    return dict(iou_brut=iou_b, dice_brut=dice_b, iou_filtre=iou_f, dice_filtre=dice_f,
                pct_pred=100.0 * pred_f.sum() / n if n else 0.0,
                pct_gt=100.0 * (gt.astype(bool) & obj.astype(bool)).sum() / n if n else 0.0,
                pred_filtre=pred_f)


def evaluer(evalu, seuil_pix, seuil_img=None):
    """Évalue sur le jeu d'évaluation (jamais utilisé pour calibrer les seuils)."""
    y = np.array([r["label"] for r in evalu])
    s = np.array([r["score"] for r in evalu])
    yhat = (s >= seuil_img).astype(int) if seuil_img is not None else \
        np.array([r["pred_label"] for r in evalu])
    deux = len(set(y)) == 2
    defect = [r for r in evalu if r["label"] == 1]
    res = [metriques(r["hm"] > seuil_pix, r["gt"], r["obj"]) for r in defect]
    ypix = np.concatenate([r["gt"].ravel() for r in evalu])
    spix = np.concatenate([r["hm"].ravel() for r in evalu])
    moy = lambda k: float(np.mean([x[k] for x in res]))
    return dict(
        auroc_image=float(roc_auc_score(y, s)) if deux else None,
        auroc_pixel=float(roc_auc_score(ypix, spix)) if len(set(ypix)) == 2 else None,
        f1=float(f1_score(y, yhat)), accuracy=float(accuracy_score(y, yhat)),
        precision=float(precision_score(y, yhat, zero_division=0)),
        rappel=float(recall_score(y, yhat, zero_division=0)),
        iou_filtre=moy("iou_filtre"), iou_brut=moy("iou_brut"),
        dice_filtre=moy("dice_filtre"), dice_brut=moy("dice_brut"),
        mae_surface_pct=float(np.mean([abs(x["pct_pred"] - x["pct_gt"]) for x in res])),
        n_evaluation=len(evalu))
