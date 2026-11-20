"""Calibration des seuils image (F1 max) et pixel (Dice max) sur le jeu de calibration uniquement."""
import numpy as np
from sklearn.metrics import precision_recall_curve

from .evaluate import metriques


def calibrer_seuil_image(calib):
    y = np.array([r["label"] for r in calib])
    s = np.array([r["score"] for r in calib])
    if len(set(y)) < 2:
        return None                                  # repli : seuil Anomalib par défaut
    p, rc, t = precision_recall_curve(y, s)
    f1 = 2 * p * rc / (p + rc + 1e-12)
    return float(t[int(np.argmax(f1[:-1]))])


def calibrer_seuil_pixel(calib, n_candidats=40):
    defect = [r for r in calib if r["label"] == 1]
    valeurs = np.concatenate([r["hm"].ravel() for r in defect])
    candidats = np.quantile(valeurs, np.linspace(0.80, 0.999, n_candidats))
    dices = [np.mean([metriques(r["hm"] > t, r["gt"], r["obj"])["dice_filtre"] for r in defect])
             for t in candidats]
    k = int(np.argmax(dices))
    return float(candidats[k]), float(dices[k])
