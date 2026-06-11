# Projet Deep Learning — EMSI Casablanca 2025–2026

> **Module :** Deep Learning | **Filière :** IAD | **Auteur :** Adam GHZAOUNI  
> **Établissement :** EMSI Mouley Youssef, Casablanca

---

## Idée générale du projet

Ce projet explore trois familles architecturales fondamentales du deep learning, chacune adaptée à une modalité de données différente :

| Partie | Architecture | Données | Tâche |
|--------|-------------|---------|-------|
| **I** | MLP (Perceptron Multicouche) | Breast Cancer Wisconsin (tabulaire) | Classification binaire (maligne/bénigne) |
| **II** | CNN (Réseau Convolutionnel) | MNIST (images 28×28) | Classification de chiffres manuscrits (10 classes) |
| **III** | RNN / LSTM / GRU + Seq2Seq | Corpus Tatoeba fra-eng (texte) | Modèle de langage + Traduction français→anglais |

L'ensemble des expériences est conduit sous **PyTorch** avec une graine aléatoire fixée (`SEED=42`) pour garantir la reproductibilité. Chaque partie articule fondements théoriques, implémentation commentée, expérimentation comparative et analyse critique.

---

## Résultats clés

### Partie I — MLP sur Breast Cancer Wisconsin
- **99% accuracy** sur le test set (86 échantillons), 1 seule erreur
- Comparaison de 3 stratégies d'initialisation : Gaussienne, Constante, **Xavier (Glorot)** ✓
- Xavier se distingue par une convergence plus stable (train loss finale 5.9× inférieure à la constante)
- F1-score pondéré : **0.99**

### Partie II — CNN (LeNet modernisé) sur MNIST
- **99.10% accuracy** de validation vs 98.05% pour le MLP — avec 2.14× moins de paramètres
- Étude d'ablation sur 5 facteurs : padding, pooling, nombre de filtres, stride, convolution 1×1
- Le nombre de filtres est le levier dominant (Δ = +2.37% entre filters=2 et filters=32)
- Visualisation des cartes de caractéristiques par hook PyTorch (`register_forward_hook`)

### Partie III — RNN/LSTM/GRU + Seq2Seq sur fra-eng
- GRU : perplexité **3.34** (meilleure), LSTM : 3.35, RNN : 3.39
- Système Seq2Seq encodeur–décodeur LSTM avec teacher forcing (ratio=0.5)
- Score BLEU : **0.5531** (Beam Search k=3) vs 0.4747 (Greedy) → +16.5%

---

## Structure du dépôt

```
deep-learning-emsi/
│
├── notebooks/
│   ├── partie1_2_MLP_CNN.ipynb        # Parties I & II : MLP (Breast Cancer) + CNN (MNIST)
│   └── partie3_RNN_Seq2Seq.ipynb      # Partie III : RNN/LSTM/GRU + Seq2Seq fra→eng
│
├── report/
│   └── Rapport_DeepLearning_EMSI.pdf  # Rapport complet (25 pages) — géré via Git LFS
│
├── requirements.txt                   # Dépendances Python
├── .gitattributes                     # Configuration Git LFS (fichiers lourds)
├── .gitignore
└── README.md
```

---

## Contenu détaillé des notebooks

### `partie1_2_MLP_CNN.ipynb`

**Partie I — MLP & ingénierie PyTorch**
- Setup reproductibilité (SEED=42, device detection)
- Chargement et exploration du dataset Breast Cancer Wisconsin (569 échantillons, 30 features)
- Préparation sans data leakage : split stratifié 70/15/15, StandardScaler fit sur train uniquement
- Implémentation MLP en deux variantes : `nn.Sequential` et classe personnalisée `nn.Module`
- Inspection des paramètres : `named_parameters()`, `state_dict()`
- Comparaison de 3 stratégies d'initialisation sur 100 époques
- Sauvegarde/rechargement du meilleur checkpoint (`best_mlp.pth`)
- Évaluation finale : matrice de confusion, rapport de classification

**Partie II — CNN & vision par ordinateur**
- Chargement MNIST avec `torchvision`, DataLoaders (batch=64)
- Implémentation manuelle validée numériquement : corrélation croisée 2D, max-pooling, average-pooling
- Visualisation de l'effet de noyaux (bord horizontal/vertical, flou) sur des images MNIST
- Architecture **LeNet modernisée** : BatchNorm2d, ReLU, convolution 1×1, 3 couches FC
- Entraînement 10 époques, best checkpoint à l'époque 8 (val acc = 99.10%)
- Étude d'ablation systématique (5 facteurs, un à la fois)
- Visualisation des cartes de caractéristiques Conv1 via `register_forward_hook`
- Comparaison MLP vs CNN : accuracy, nombre de paramètres, courbes de convergence

### `partie3_RNN_Seq2Seq.ipynb`

**Partie III — RNN, LSTM, GRU & Seq2Seq**
- Téléchargement et préprocessing du corpus Tatoeba fra-eng (filtrage ≤12 tokens, 20 000 paires)
- Construction de vocabulaires source/cible avec tokens spéciaux `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Implémentation des 3 architectures : `VanillaRNN`, `LSTM`, `GRU` (mêmes hyperparamètres)
- Entraînement comparatif 10 époques — métriques : loss masquée, perplexité
- Illustration expérimentale du gradient clipping (`max_norm=1.0`) sur 90 batchs
- Système Seq2Seq encodeur–décodeur LSTM avec teacher forcing
- Entraînement 15 époques, analyse de la convergence train/val
- Décodage glouton (Greedy) et Beam Search (k=3) avec normalisation par longueur (α=0.7)
- Évaluation BLEU sur 200 paires de test, analyse qualitative de 8 exemples

---

## Installation & utilisation

### Prérequis

- Python ≥ 3.9
- GPU recommandé (CUDA) mais CPU supporté

### Installation des dépendances

```bash
git clone https://github.com/<votre-username>/deep-learning-emsi.git
cd deep-learning-emsi
pip install -r requirements.txt
```

### Lancer les notebooks

```bash
jupyter notebook notebooks/
```

Ouvrir dans l'ordre :
1. `partie1_2_MLP_CNN.ipynb` — Parties I & II
2. `partie3_RNN_Seq2Seq.ipynb` — Partie III

> **Note :** Le notebook Partie III télécharge automatiquement le corpus Tatoeba fra-eng via `gdown`. Une connexion internet est requise lors de la première exécution.

### Fichiers de modèles sauvegardés

Les checkpoints entraînés (`.pth`) ne sont pas inclus dans le dépôt (voir `.gitignore`). Relancer les notebooks génère automatiquement les fichiers `best_mlp.pth` et `best_seq2seq.pth` dans le répertoire courant.

---

## Dépendances principales

| Librairie | Usage |
|-----------|-------|
| `torch`, `torchvision` | Modèles, entraînement, datasets MNIST |
| `scikit-learn` | Dataset Breast Cancer, métriques, StandardScaler |
| `matplotlib`, `seaborn` | Visualisations, courbes, matrices de confusion |
| `nltk` | Calcul du score BLEU |
| `gdown` | Téléchargement du corpus Tatoeba |
| `numpy`, `pandas` | Manipulation des données |

---

## Notes sur Git LFS

Le rapport PDF (≈1.4 Mo) est tracké via **Git LFS** (voir `.gitattributes`). Pour cloner le dépôt avec tous les fichiers lourds :

```bash
# Installer Git LFS si ce n'est pas déjà fait
git lfs install
git clone https://github.com/<votre-username>/deep-learning-emsi.git
```

---

## Références

- Goodfellow, Bengio, Courville (2016). *Deep Learning*. MIT Press.
- LeCun et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
- Hochreiter & Schmidhuber (1997). Long short-term memory. *Neural Computation*.
- Cho et al. (2014). Learning phrase representations using RNN encoder–decoder. *EMNLP*.
- Glorot & Bengio (2010). Understanding the difficulty of training deep feedforward networks. *AISTATS*.
- Paszke et al. (2019). PyTorch: An imperative style high-performance deep learning library. *NeurIPS*.

---

*Année universitaire 2025–2026 — EMSI Mouley Youssef, Casablanca*
