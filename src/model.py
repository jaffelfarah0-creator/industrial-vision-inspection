"""Fabrique de modèles Anomalib (référence PatchCore + modèles de comparaison)."""


def construire_modele(nom="patchcore"):
    from anomalib import models
    nom = nom.lower()
    if nom == "patchcore":
        return models.Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                                pre_trained=True, coreset_sampling_ratio=0.05)
    if nom == "padim":
        return models.Padim()
    if nom == "stfpm":
        return models.Stfpm()
    raise ValueError(f"Modèle inconnu : {nom}")
