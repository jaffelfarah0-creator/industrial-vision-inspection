"""Quantification de la surface défectueuse et décision Conforme / Mineur / Non conforme."""
import numpy as np
import yaml


def pourcentage_surface(masque_defaut, masque_objet):
    n = int(np.sum(masque_objet))
    return 100.0 * float(np.sum(np.logical_and(masque_defaut > 0, masque_objet > 0))) / n if n else 0.0


def decision(pct, image_anormale, config_path="configs/decision.yaml"):
    seuils = yaml.safe_load(open(config_path, encoding="utf-8"))["seuils"]
    if not image_anormale:
        return "Conforme"
    if pct >= seuils["non_conforme_pct"]:
        return "Non conforme"
    if pct >= seuils["mineur_pct"]:
        return "Mineur"
    return "Conforme"
