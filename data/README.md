# Données

Le dataset **n'est pas versionné** (taille et licence). Il doit être téléchargé localement.

## Source
- Officielle : https://www.mvtec.com/company/research/datasets/mvtec-ad
- Miroir Kaggle : https://www.kaggle.com/datasets/ipythonx/mvtec-ad

## Téléchargement automatique (Kaggle)
```python
import kagglehub
path = kagglehub.dataset_download("ipythonx/mvtec-ad")
print(path)
```
`python -m src.train` effectue ce téléchargement si `--data-root` n'est pas fourni.

## Organisation attendue
```
<racine>/<categorie>/
├── train/good/                      images saines (entraînement)
├── test/good/                       images saines (test)
├── test/<type_defaut>/              images défectueuses (ex: bent, color, flip, scratch)
└── ground_truth/<type_defaut>/      masques de vérité terrain
```

## Prétraitement
Redimensionnement et recadrage gérés par Anomalib (Resize 256 puis CenterCrop, selon la version), normalisation ImageNet. La segmentation de la pièce est faite dans `src/segmentation.py`.

## Licence
MVTec AD : CC BY-NC-SA 4.0 (usage non commercial). Citer Bergmann et al. (voir `docs/references.md`).
