"""Alignement géométrique et segmentation d'objet par seuillage d'Otsu."""
import cv2
import numpy as np

TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
SEUIL_PIXEL_RATIO_TROU = 0.005


def charger_image_alignee(chemin, H, W, taille_resize=256):
    """Reproduit Resize(256) puis CenterCrop(H, W) d'Anomalib."""
    bgr = cv2.imread(chemin)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if H <= taille_resize and W <= taille_resize:
        rgb = cv2.resize(rgb, (taille_resize, taille_resize), interpolation=cv2.INTER_AREA)
        y0, x0 = (taille_resize - H) // 2, (taille_resize - W) // 2
        return rgb[y0:y0 + H, x0:x0 + W]
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)


def segmenter_objet(image_rgb, categorie, textures=TEXTURES, ratio_trou=SEUIL_PIXEL_RATIO_TROU):
    """Retourne un masque binaire (1 = objet inspecté, 0 = fond)."""
    h, w = image_rgb.shape[:2]
    if categorie in textures:
        return np.ones((h, w), np.uint8)
    gris = cv2.GaussianBlur(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bord = np.concatenate([binaire[0], binaire[-1], binaire[:, 0], binaire[:, -1]])
    if np.mean(bord > 0) > 0.5:
        binaire = 255 - binaire
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binaire = cv2.morphologyEx(binaire, cv2.MORPH_CLOSE, k)
    contours, hier = cv2.findContours(binaire, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    masque = np.zeros((h, w), np.uint8)
    if not contours:
        return masque + 1
    idx = int(np.argmax([cv2.contourArea(c) for c in contours]))
    cv2.drawContours(masque, contours, idx, 1, -1)
    for i, c in enumerate(contours):
        if hier[0][i][3] == idx and cv2.contourArea(c) > ratio_trou * h * w:
            cv2.drawContours(masque, contours, i, 0, -1)
    return masque
