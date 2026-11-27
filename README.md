# Detection_Defaults_MVTec

**Détection, localisation et quantification de défauts industriels par apprentissage profond non supervisé (MVTec AD, PatchCore)**

## 1. Membres du groupe

| Nom | Prénom |
|---|---|
| Jaffel | Farah |
| Ounissi | Olfa |
| Zghal | Fatma |
| Chelbi | Mohamed Aziz |

## 2. Contexte et problématique

En contrôle qualité industriel, les pièces défectueuses sont rares et leurs défauts très variés (rayure, déformation, couleur, retournement...). Constituer un jeu d'entraînement annoté couvrant tous les défauts possibles est irréaliste. L'**apprentissage non supervisé** (entraînement sur des pièces saines uniquement) est donc adapté à ce contexte.

**Problématique :** comment détecter, localiser et *quantifier* (en % de surface de la pièce) les défauts à partir de pièces saines uniquement, avec une évaluation honnête (sans fuite de seuil) et une aide à la décision exploitable en usine (Conforme / Mineur / Non conforme) ?

## 3. Objectifs

1. Détecter les pièces défectueuses (niveau image).
2. Localiser les défauts (cartes de chaleur et masques de segmentation).
3. Quantifier la surface défectueuse : `% touché = pixels défectueux sur l'objet / pixels de l'objet × 100`.
4. Évaluer rigoureusement : AUROC, F1, IoU, Dice, erreur absolue moyenne (MAE) sur la surface, analyse par type de défaut.
5. Proposer une aide à la décision paramétrable et une démonstration interactive.

## 4. Solution envisagée

- **Détection et localisation :** PatchCore (backbone WideResNet-50, couches `layer2` et `layer3`) via la bibliothèque Anomalib.
- **Segmentation de la pièce :** Otsu + morphologie (OpenCV), avec détection de la polarité du fond, pour isoler la surface de la pièce et exclure le fond.
- **Calibration sans fuite de seuil :** le jeu de test est séparé en une partie de calibration (seuils image et pixel) et une partie d'évaluation jamais utilisée pour régler les seuils.
- **Quantification :** pourcentage de surface 2D touchée (la surface visible, pas le volume du défaut).
- **Décision :** règles paramétrables dans `configs/decision.yaml`.
- **Démonstration :** application Gradio, puis dashboard Streamlit alimenté par une API FastAPI.

## 5. Approches et architectures

| Rôle | Méthode |
|---|---|
| Modèle de référence | PatchCore (WideResNet-50) |
| Modèles de comparaison prévus | PaDiM, STFPM (autres familles d'Anomalib) |
| Post-traitement prévu | Lissage gaussien de la carte, nettoyage morphologique, seuil pixel calibré |
| Export prévu | ONNX / OpenVINO pour le déploiement |

PatchCore utilise un backbone préentraîné ; la contribution du projet porte donc sur le protocole d'évaluation, la calibration, le post-traitement, la quantification et la comparaison expérimentale.

## 6. Dataset

- **MVTec AD** (Bergmann et al., CVPR 2019), 15 catégories (10 objets, 5 textures), environ 5 300 images.
- Entraînement : images saines uniquement. Test : images saines et défectueuses, avec masques de vérité terrain pixel à pixel.
- **Catégorie principale du rapport : `metal_nut`** (défauts : `bent`, `color`, `flip`, `scratch`). Les effectifs exacts sont à confirmer à la lecture du dataset (`notebooks/01_exploration_dataset.ipynb`).
- **Source officielle :** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Miroir Kaggle utilisé :** https://www.kaggle.com/datasets/ipythonx/mvtec-ad
- **Licence :** CC BY-NC-SA 4.0 (usage non commercial), à vérifier sur la page officielle. Le dataset n'est **pas** déposé sur ce dépôt (voir `data/README.md`).

## 7. Étapes de réalisation, Avancement & Feuille de route (Sprints)

- [x] **Prototype Kaggle** : entraînement PatchCore (`wide_resnet50_2`, `layer2`/`layer3`, `coreset_sampling_ratio=0.05` via Anomalib), calcul des métriques, quantification de surface et export (`resume.json` + archive ZIP)
- [x] **Sprint 1 (en cours de finalisation)** : corrections méthodologiques déjà codées dans le notebook (segmentation objet Otsu + alignement `Resize`/`CenterCrop` + gestion des textures/trous, split stratifié calibration/évaluation sans fuite de seuils, métriques IoU/Dice bruts vs filtrés, latence forward propre) — *synchronisation des modules `src/` via `pousser()`*
- [x] **Sprint 2 — Visualisation Streamlit (prochaine étape)** : développement du tableau de bord interactif Streamlit sur un seul écran lisible, sans valeurs codées en dur, alimenté par les sorties exportées (`resume.json`, images alignées, heatmaps, masques filtrés vs GT, score d'anomalie et % de surface défectueuse)
- [ ] **Sprint 3 — Approfondissement & post-traitement** : raffinement du post-traitement des masques, comparaison PatchCore / PaDiM / STFPM et étude d'ablation
- [ ] **Sprint 4 — Industrialisation & inspection** : module d'inspection, journalisation SQLite, règles de décision YAML, export ONNX/OpenVINO et API FastAPI
- [ ] **Sprint 5 — Généralisation & monitoring** : exécution cyclique sur d'autres catégories MVTec AD et suivi/monitoring de dérive
- [ ] **Sprint 6 — Finalisation GitHub, Rapport & Démo** : organisation complète du dépôt GitHub (`src/`, `app/` Streamlit, `README.md`, `requirements.txt`), rédaction du rapport académique final et préparation de la démo live
## 8. Résultats préliminaires

Obtenus sur `metal_nut` avec le prototype initial, **avant** correction de la fuite de seuil : ils sont donc optimistes et seront remplacés par les résultats du protocole corrigé (voir `results/`).

| Métrique | Valeur (prototype) |
|---|---|
| AUROC image | 1.0000 |
| F1-score image | ~0.99 |
| AUROC pixel | ~0.987 |
| IoU moyen | ~0.558 |
| Dice moyen | ~0.695 |
| MAE surface | ~3.47 % |

## 9. Technologies et bibliothèques

Python 3.10+, PyTorch, Anomalib, OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, PyYAML, Gradio (puis Streamlit, FastAPI). Détail dans `requirements.txt`.

## 10. Installation et exécution (prévisionnel)

```bash
git clone https://github.com/<jaffelfarah0-creator>/Detection_Defaults_MVTec.git
cd Detection_Defaults_MVTec
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Télécharger le dataset en suivant `data/README.md`, puis :

```bash
# Expérience reproductible (catégorie et graine fixées)
python -m src.train --category metal_nut --seed 42 --model patchcore

# Catégorie tirée au hasard à chaque exécution (exploration)
python -m src.train --category random
```

Chaque exécution écrit un `resume.json` (graine, catégorie, seuils, métriques) dans `results/`. Un GPU est recommandé. Pour exécuter sans installation locale : importer `notebooks/02_demo_kaggle.ipynb` sur Kaggle (GPU et Internet activés) puis « Run All ».

## Organisation du dépôt

```
configs/    règles de décision (YAML)
data/       instructions de téléchargement (pas de données)
notebooks/  01_exploration_dataset.ipynb (exploration) et 02_demo_kaggle.ipynb (pipeline complet exécutable sur Kaggle)
scripts/    sprint1_pipeline_corrige.py (même pipeline, au format cellules # %%)
src/        code source (prétraitement, segmentation, calibration, évaluation, quantification, benchmark)
models/     points de contrôle (non versionnés)
results/    tableaux, figures, resume.json
docs/       références bibliographiques
app/        démonstration interactive
```

## Intégrité académique

Les travaux, codes, datasets et modèles préexistants sont cités dans `docs/references.md` (MVTec AD, PatchCore, Anomalib, WideResNet). Toute contribution externe y est mentionnée.
