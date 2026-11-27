"""Calcul des métriques IoU et Dice (bruts et filtrés) et pourcentages de surface."""
import cv2
import numpy as np

K_BORD = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def iou_dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter, union, s = (a & b).sum(), (a | b).sum(), a.sum() + b.sum()
    return (inter / union if union > 0 else 1.0), (2 * inter / s if s > 0 else 1.0)


def metriques(pred, gt, obj, k_bord=K_BORD):
    pred = (pred > 0).astype(np.uint8)
    pred_f = pred & cv2.dilate(obj, k_bord)
    iou_b, dice_b = iou_dice(pred, gt)
    iou_f, dice_f = iou_dice(pred_f, gt)
    n_obj = obj.sum()
    return dict(
        iou_brut=iou_b,
        dice_brut=dice_b,
        iou_filtre=iou_f,
        dice_filtre=dice_f,
        pct_pred=100.0 * pred_f.sum() / n_obj if n_obj else 0.0,
        pct_gt=100.0 * (gt.astype(bool) & obj.astype(bool)).sum() / n_obj if n_obj else 0.0,
        pred_filtre=pred_f,
    )
