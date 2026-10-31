# HAM10000 — Classification de lésions cutanées et Explainable AI
**Imagerie médicale · Vision par ordinateur · Classification d’images · Apprentissage profond · Explicabilité (XAI)**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![Torchvision](https://img.shields.io/badge/Torchvision-Computer_Vision-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html) [![Captum](https://img.shields.io/badge/XAI-Captum-284B9B)](https://captum.ai/) ![Statut](https://img.shields.io/badge/Statut-Exp%C3%A9rimentation_termin%C3%A9e-2ea44f)


> **Statut : expérimentation terminée.** Les cinq notebooks ont été exécutés sur Kaggle : exploration, préparation des données, entraînement, évaluation et explicabilité.
>
> **Avertissement :** ce projet est expérimental et pédagogique. Il ne constitue **pas** un outil de diagnostic médical.

## Vue d'ensemble

L'objectif est de comparer trois architectures pré-entraînées sur ImageNet pour classifier des images dermoscopiques issues de **HAM10000** en sept classes, puis d'examiner qualitativement leurs prédictions à l'aide de méthodes d'explicabilité (XAI).

Les priorités méthodologiques sont :

- éviter la fuite de données en séparant les exemples par `lesion_id`.
- gérer le déséquilibre entre les catégories.
- sélectionner le modèle **sur Validation uniquement**, avant de consulter le Test Set.
- analyser les performances par classe, en particulier le rappel du mélanome.
- confronter les bonnes et mauvaises prédictions à des cartes d'attribution.

## Résultats principaux

**ViT-B/16** est le modèle sélectionné sur la base du meilleur **macro-F1 de validation (78,55 %)**. Sur le Test Set de 1 502 images, il obtient **85,22 % d'accuracy** et **75,26 % de macro-F1**.

| Modèle | Accuracy Test | Macro-F1 Test | Balanced accuracy | Recall mélanome (`mel`) | Macro-F1 Validation |
|:--|--:|--:|--:|--:|--:|
| ResNet50 | 80,56 % | 70,12 % | 75,50 % | **69,48 %** | 70,71 % |
| EfficientNet-B0 | 79,69 % | 68,24 % | **75,53 %** | 66,88 % | 70,75 % |
| **ViT-B/16** | **85,22 %** | **75,26 %** | 75,36 % | 66,23 % | **78,55 %** |

**Point important :** le meilleur modèle selon le macro-F1 global n'est pas celui qui détecte le plus de mélanomes. Sur les **154 mélanomes du Test Set**, ResNet50 en identifie **107**, contre **102** pour ViT-B/16. En particulier, ViT-B/16 classe **28 mélanomes comme Naevus (`nv`)**. Ces résultats doivent être interprétés dans le contexte des erreurs par classe, pas seulement à partir de l'accuracy.

## Jeu de données

**HAM10000** (*Human Against Machine with 10,000 training images*) contient **10 015 images dermoscopiques**, représentant **7 470 lésions distinctes** et **7 catégories diagnostiques**.

| Code | Catégorie | Nombre d'images |
|:--|:--|--:|
| `akiec` | Kératoses actiniques et carcinomes intraépithéliaux | 327 |
| `bcc` | Carcinomes basocellulaires | 514 |
| `bkl` | Lésions bénignes de type kératose | 1 099 |
| `df` | Dermatofibromes | 115 |
| `mel` | Mélanomes | 1 113 |
| `nv` | Naevus mélanocytaires | 6 705 |
| `vasc` | Lésions vasculaires | 142 |

Source : [HAM10000 sur Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). [HAM10000 sur Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T). Article de référence : [Tschandl, Rosendahl & Kittler (2018)](https://doi.org/10.1038/sdata.2018.161).


## Méthodologie

### 1. Préparation et prévention de la fuite de données

Les images sont séparées par **identifiant de lésion (`lesion_id`)**, de façon à ce qu'aucune lésion ne figure dans plusieurs partitions. La graine du split retenu est `71`, la graine générale de reproductibilité utilisée dans les notebooks est `42`.

| Partition | Images | Lésions distinctes |
|:--|--:|--:|
| Train | 7 012 | 5 229 |
| Validation | 1 501 | 1 120 |
| Test | 1 502 | 1 121 |
| **Total** | **10 015** | **7 470** |

Les trois intersections de `lesion_id` ont été vérifiées vides. Les poids de classes sont calculés **sur Train uniquement**. Les images sont redimensionnées à **224 × 224**, puis normalisées avec les statistiques ImageNet. Les augmentations aléatoires ne concernent que Train.

> Le découpage par lésion empêche qu'une même lésion soit présente dans plusieurs partitions, mais **ne garantit pas** une séparation au niveau patient si l'identifiant du patient n'est pas disponible.

### 2. Fine-tuning des architectures

Trois modèles `torchvision` initialisés avec des poids ImageNet ont été adaptés aux sept classes : **ResNet50**, **EfficientNet-B0** et **ViT-B/16**.

- Fine-tuning de l'ensemble du réseau, avec `AdamW` et des taux d'apprentissage distincts : `1e-5` pour le backbone et `1e-4` pour la tête.
- `weight_decay = 1e-4` et perte `CrossEntropyLoss` pondérée par les fréquences de Train, perte non pondérée pour Validation.
- Scheduler `ReduceLROnPlateau` (`patience=2`) et `EarlyStopping` (`patience=5`), au plus 30 époques.
- Tailles de batch respectives : 16 (ResNet50), 24 (EfficientNet-B0), 8 (ViT-B/16).
- Les meilleurs checkpoints sont sauvegardés à partir du **macro-F1 de Validation**.

| Modèle | Meilleure époque | Macro-F1 Validation |
|:--|--:|--:|
| ResNet50 | 16 | 0,707088 |
| EfficientNet-B0 | 22 | 0,707474 |
| ViT-B/16 | 8 | **0.785512** |

### 3. Évaluation indépendante

Les trois checkpoints retenus sont évalués sur **le même Test Set**. Les notebooks exportent les prédictions individuelles, probabilités softmax, rapports de classification, matrices de confusion et un tableau comparatif (`test_comparison.csv`). La sélection finale est tracée dans `selected_model.json` avec la métrique `validation_macro_f1` : **aucun résultat Test n'est utilisé pour changer le modèle sélectionné**.

### 4. Explainable AI (XAI)

Deux méthodes de la bibliothèque officielle **[Captum](https://captum.ai/)** sont utilisées sur les modèles fine-tunés :

| Architecture expliquée | Méthode | Implémentation |
|:--|:--|:--|
| EfficientNet-B0 (meilleur CNN sur Validation) | **Grad-CAM** | `captum.attr.LayerGradCam`, dernière partie convolutionnelle |
| ViT-B/16 (modèle sélectionné) | **Integrated Gradients** | `captum.attr.IntegratedGradients`, attributions des pixels |

**Dix cartes** ont été produites : **cinq images Test par modèle**, choisies de manière reproductible pour représenter des prédictions correctes et des erreurs. Les figures comprennent l'image originale, la carte d'attribution et sa superposition. Les métadonnées figurent dans `xai_examples.csv`.

Les méthodes ont des interprétations différentes :

- Grad-CAM affiche ici les **contributions positives** au score prédit.
- Integrated Gradients affiche ici la **somme des valeurs absolues** des attributions sur les trois canaux RGB : le signe des contributions n'est donc pas conservé dans la carte visualisée. Une référence nulle **dans l'espace normalisé** et 24 étapes d'intégration sont utilisées.

**Exemple notable — `ISIC_0024351` :** les deux modèles prédisent à tort `nv` alors que la classe réelle est `mel`. ViT-B/16 attribue une confiance softmax de **98,87 %** à cette prédiction incorrecte, et EfficientNet-B0 **86,91 %**. Une probabilité élevée ne garantit donc pas une bonne prédiction ni une sécurité clinique.

Les dix cas XAI sont un **échantillon intentionnel**, comprenant six erreurs et quatre prédictions correctes : ils ne constituent pas un échantillon représentatif permettant d'estimer les performances générales.

## Structure du dépôt

```text
skin-lesion-Classification/
│
├── data/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── models/
│
├── notebooks/
│   ├── 01-dataexploration.ipynb
│   ├── 02-datapreprocessing.ipynb
│   ├── 03-modelstraining.ipynb
│   ├── 04-modelsevaluation.ipynb
│   └── 05-xai-gradcam-ig.ipynb
│
├── results/
│   ├── figures/
│   │   ├── confusion_efficientnet_b0.png
│   │   ├── confusion_resnet50.png
│   │   ├── confusion_vit_b_16.png
│   │   ├── eda_class_distribution.png
│   │   ├── eda_class_examples.png
│   │   ├── eda_metadata.png
│   │   ├── efficientnet_b0_learning_curves.png
│   │   ├── resnet50_learning_curves.png
│   │   ├── split_class_distribution.png
│   │   └── vit_b_16_learning_curves.png
│   │
│   ├── metrics/
│   │   ├── classification_report_efficientnet_b0.csv
│   │   ├── classification_report_resnet50.csv
│   │   ├── classification_report_vit_b_16.csv
│   │   ├── predictions_efficientnet_b0.csv
│   │   ├── predictions_resnet50.csv
│   │   ├── predictions_vit_b_16.csv
│   │   ├── selected_model.json
│   │   └── test_comparison.csv
│   │
│   ├── training/
│   │   ├── efficientnet_b0_history.csv
│   │   ├── efficientnet_b0_summary.json
│   │   ├── resnet50_history.csv
│   │   ├── resnet50_summary.json
│   │   ├── vit_b_16_history.csv
│   │   └── vit_b_16_summary.json
│   │
│   ├── xai/
│   │    ├── efficientnet_b0/
│   │    ├── vit_b_16/
│   │    └── xai_examples.csv
│   │
│   ├── data_config.json
│   └── training_summary.csv
│   
├── .gitignore
└── README.md
```

Les résultats exécutés et les checkpoints sont générés sur Kaggle. Les fichiers volumineux (`.pt`, images du dataset) et les sorties non publiées ne sont pas supposés disponibles directement dans ce dépôt.

## Reproduire les expériences sur Kaggle

Les notebooks ont été exécutés avec **PyTorch / torchvision**, **pandas**, **NumPy**, **scikit-learn**, **matplotlib**, **Pillow** et **Captum** (version observée pour l'étape XAI : `0.9.0`). Un **GPU Kaggle** est recommandé pour l'entraînement et Integrated Gradients.

1. **Obtenir HAM10000** depuis la source officielle et le joindre au notebook `01-dataexploration.ipynb` comme Input Kaggle.
2. **Exécuter `01-dataexploration.ipynb`** pour explorer les classes et la qualité des images.
3. **Exécuter `02_datapreprocessing.ipynb`** et sauvegarder les partitions `train.csv`, `val.csv`, `test.csv` et `data_config.json`.
4. **Exécuter `03_modelstraining.ipynb`**, avec HAM10000 et les fichiers de préparation en Inputs, sauvegarder les trois fichiers `*_best.pt` et les historiques.
5. **Exécuter `04_modelsevaluation.ipynb`**, avec les partitions et checkpoints en Inputs, sauvegarder `selected_model.json`, `test_comparison.csv`, les prédictions et les matrices de confusion.
6. **Exécuter `05-xai-gradcam-ig.ipynb`**, en joignant également les Outputs du notebook 04. Installer Captum si nécessaire : `%pip install captum`.

**Important :** les chemins Kaggle sont propres au compte et aux versions des notebooks. Mettre à jour les constantes `DATA_DIR`, `PREPROCESSING_DIR`, `MODELS_DIR` et `METRICS_DIR` au début des notebooks si les Inputs sont attachés à d'autres emplacements. Les fichiers créés dans `/kaggle/working` doivent être conservés dans les Outputs d'une version Kaggle, puis joints comme Input aux notebooks suivants.


## Limites et pistes futures

- Jeu de données **déséquilibré**, avec une forte majorité de `nv`, les petites classes comportent peu d'exemples Test.
- Le **rappel du mélanome** reste limité, y compris pour le modèle retenu.
- Aucune **validation externe multicentrique** ni évaluation en pratique clinique n'a été conduite.
- Les scores softmax ne sont **pas calibrés** ici, leur valeur ne représente pas une probabilité médicale validée.
- Les cartes XAI sont des **attributions dépendantes d'une méthode et de ses paramètres**, pas des segmentations vérifiées ou une démonstration de raisonnement médical.
- La convergence (Integrated Gradients) : Sur le modèle final (**ViT-B/16 atteignant 78.55% de Macro-F1**), le convergence_delta calculé avec **24 étapes** est de **0.30915** Cette valeur indique que l'approximation numérique est moins précise que sur les versions préliminaires, probablement due à la complexité accrue des gradients du modèle optimisé. **Perspective** : pour une analyse XAI plus fiable sur ce modèle final, il serait recommandé d'augmenter le nombre d'étapes d'intégration (ex: 128 étapes) afin de réduire cet écart.

Extensions pertinentes : validation externe, calibration, analyse de sensibilité des méthodes XAI, robustesse aux artefacts visuels, rappel du mélanome et comparaison des explications avec des annotations expertes lorsque disponibles.

## À propos

**Imane El Qoraychy**

Actuellement étudiante en double cursus de Master :

- **Master WISD — Web Intelligence et Sciences des Données**, Faculté des Sciences Dhar El Mahraz (FSDM), Université Sidi Mohamed Ben Abdellah, Fès, Maroc.
- **Master EID² — Exploration Informatique des Données et Décisionnel**, Université Sorbonne Paris Nord (USPN), Institut Galilée, France.

>Passionnée par la **Data Science, le Machine Learning, le Deep Learning et l'Intelligence Artificielle**, avec un intérêt particulier pour la **vision par ordinateur, l'IA médicale, l'explicabilité des modèles (XAI) et l'IA générative (Generative AI)**.
---

*Portfolio de recherche en classification d'images médicales et explicabilité, usage non clinique.*
