"""Chargement des données MVTec AD, choix de la catégorie, graines, découpage calibration/évaluation."""
import os
import random
from pathlib import Path

import cv2
import numpy as np

TOUTES_CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                     "leather", "metal_nut", "pill", "screw", "tile",
                     "toothbrush", "transistor", "wood", "zipper"]
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def fixer_graine(seed=None):
    """Retourne (graine, rng). seed=None -> nouvelle graine aléatoire."""
    import torch
    graine = seed if seed is not None else int.from_bytes(os.urandom(4), "little")
    np.random.seed(graine)
    torch.manual_seed(graine)
    return graine, random.Random(graine)


def trouver_racine(chemin):
    for root, dirs, _ in os.walk(chemin):
        if any(c in dirs for c in TOUTES_CATEGORIES):
            return Path(root)
    raise FileNotFoundError(f"Aucune catégorie MVTec trouvée sous {chemin}")


def telecharger_dataset():
    import kagglehub
    return trouver_racine(kagglehub.dataset_download("ipythonx/mvtec-ad"))


def choisir_categorie(racine, demande, rng, fichier_histo="historique_categories.txt"):
    """demande = nom de catégorie, ou 'random' (jamais deux fois la même avant d'avoir tout parcouru)."""
    disponibles = [c for c in TOUTES_CATEGORIES if (Path(racine) / c).is_dir()]
    if demande != "random":
        assert demande in disponibles, f"Catégorie inconnue : {demande}"
        return demande
    f = Path(fichier_histo)
    vues = f.read_text().split() if f.exists() else []
    restantes = [c for c in disponibles if c not in vues]
    if not restantes:
        vues, restantes = [], disponibles
    choix = rng.choice(restantes)
    f.write_text("\n".join(vues + [choix]))
    return choix


def charger_image_alignee(chemin, H, W, taille_resize=256):
    """Reproduit la géométrie d'Anomalib (Resize puis CenterCrop) pour aligner image, heatmap et masque."""
    rgb = cv2.cvtColor(cv2.imread(str(chemin)), cv2.COLOR_BGR2RGB)
    if H <= taille_resize and W <= taille_resize:
        rgb = cv2.resize(rgb, (taille_resize, taille_resize), interpolation=cv2.INTER_AREA)
        y0, x0 = (taille_resize - H) // 2, (taille_resize - W) // 2
        return rgb[y0:y0 + H, x0:x0 + W]
    return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)


def split_stratifie(records, rng, ratio=0.5):
    """Sépare en (calibration, évaluation), stratifié par label."""
    calib, evalu = [], []
    for lab in (0, 1):
        groupe = [r for r in records if r["label"] == lab]
        rng.shuffle(groupe)
        n = int(len(groupe) * ratio)
        calib += groupe[:n]
        evalu += groupe[n:]
    return calib, evalu
