"""Segmentation de la pièce (Otsu + polarité + morphologie) pour calculer N_objet."""
import cv2
import numpy as np

from .data_preprocessing import TEXTURES


def segmenter_objet(image_rgb, categorie, ratio_trou=0.005):
    h, w = image_rgb.shape[:2]
    if categorie in TEXTURES:                      # toute l'image est la surface inspectée
        return np.ones((h, w), np.uint8)
    gris = cv2.GaussianBlur(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bord = np.concatenate([binaire[0], binaire[-1], binaire[:, 0], binaire[:, -1]])
    if np.mean(bord > 0) > 0.5:                    # fond clair -> inversion (objet = 255)
        binaire = 255 - binaire
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binaire = cv2.morphologyEx(binaire, cv2.MORPH_CLOSE, k)
    contours, hier = cv2.findContours(binaire, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    masque = np.zeros((h, w), np.uint8)
    if not contours:
        return masque + 1
    idx = int(np.argmax([cv2.contourArea(c) for c in contours]))
    cv2.drawContours(masque, contours, idx, 1, -1)
    for i, c in enumerate(contours):               # retire les vrais trous internes
        if hier[0][i][3] == idx and cv2.contourArea(c) > ratio_trou * h * w:
            cv2.drawContours(masque, contours, i, 0, -1)
    return masque
