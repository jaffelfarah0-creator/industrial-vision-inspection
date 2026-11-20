"""Pipeline complet : entraînement, prédiction, calibration, évaluation, export de resume.json.

Exemples :
    python -m src.train --category metal_nut --seed 42
    python -m src.train --category random
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import torch

from . import calibration, evaluate, segmentation
from .benchmark import mesurer_latence
from .data_preprocessing import (charger_image_alignee, choisir_categorie,
                                 fixer_graine, split_stratifie, telecharger_dataset,
                                 trouver_racine)
from .model import construire_modele


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="metal_nut", help="nom de catégorie MVTec ou 'random'")
    ap.add_argument("--seed", type=int, default=None, help="graine (défaut : aléatoire)")
    ap.add_argument("--model", default="patchcore", choices=["patchcore", "padim", "stfpm"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    os.environ["TQDM_DISABLE"] = "1"
    from anomalib.engine import Engine
    try:
        from anomalib.data import MVTecAD as MVTec
    except ImportError:
        from anomalib.data import MVTec

    graine, rng = fixer_graine(args.seed)
    racine = trouver_racine(args.data_root) if args.data_root else telecharger_dataset()
    categorie = choisir_categorie(racine, args.category, rng)
    out = Path(args.out) / f"{categorie}_{args.model}_{graine}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Graine {graine} | catégorie {categorie} | modèle {args.model}")

    dm = MVTec(root=racine, category=categorie, train_batch_size=32,
               eval_batch_size=32, num_workers=2)
    dm.setup()
    model = construire_modele(args.model)
    engine = Engine(default_root_dir=str(out / "anomalib"),
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1, enable_progress_bar=False)
    engine.fit(datamodule=dm, model=model)
    predictions = engine.predict(datamodule=dm, model=model)

    records, hw = evaluate.collecter(predictions)
    H, W = records[0]["hm"].shape[:2]
    for r in records:
        r["obj"] = segmentation.segmenter_objet(charger_image_alignee(r["path"], H, W), categorie)

    calib, evalu = split_stratifie(records, rng)
    seuil_img = calibration.calibrer_seuil_image(calib)
    seuil_pix, dice_calib = calibration.calibrer_seuil_pixel(calib)
    resume = evaluate.evaluer(evalu, seuil_pix, seuil_img)
    try:
        latence = mesurer_latence(model, hw)
    except Exception as e:                      # dépend de la version d'Anomalib
        latence = None
        print("Latence non mesurée :", e)

    resume.update(graine=graine, categorie=categorie, modele=args.model,
                  n_test=len(records), n_calibration=len(calib),
                  seuil_image=seuil_img, seuil_pixel=seuil_pix,
                  dice_calibration=dice_calib, latence_ms=latence)
    (out / "resume.json").write_text(json.dumps(resume, indent=2))
    print(json.dumps(resume, indent=2))


if __name__ == "__main__":
    main()
