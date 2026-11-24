# DM-Detection-Intrusions-UNSW-NB15 — Détection d'intrusions réseau par Machine Learning

Mini-projet de Data Mining (année académique 2026/2027) : classification supervisée de connexions réseau (normales ou malveillantes) sur le jeu de données **UNSW-NB15**.

---

## 1. Présentation, objectifs et description du projet

### Contexte métier

Un **système de détection d'intrusions (IDS)** surveille le trafic d'un réseau pour repérer les connexions malveillantes. Ce projet, dans le domaine de la **cybersécurité**, construit des modèles qui décident, à partir des caractéristiques techniques d'une connexion (durée, volume d'octets, protocole, service...), si elle est **normale** ou si elle correspond à une **attaque**.

### Objectifs

**Objectifs scientifiques**
- Explorer et nettoyer les données (statistiques descriptives, distributions, corrélations).
- Réaliser des tests statistiques justifiés : normalité (Shapiro-Wilk), comparaison de deux groupes (Mann-Whitney U), comparaison de plusieurs groupes (Kruskal-Wallis), lien entre variables catégorielles (Chi² et V de Cramér) et corrélation de Spearman.
- Comparer les algorithmes de classification du cours : Zero-R (référence), K-NN, arbre de décision, Random Forest, Naïve Bayes et SVM.
- Optimiser les hyperparamètres par validation croisée (`GridSearchCV` ou `RandomizedSearchCV`, K ≥ 5) et évaluer avec des métriques adaptées au déséquilibre des classes (F1, ROC-AUC, matrices de confusion).

**Objectifs applicatifs**
- Fournir une application **Streamlit** interactive, en six pages : exploration des données avec filtres dynamiques, contrôle qualité des données, prédiction d'une connexion en temps réel et prédiction par lot (import/export CSV).

### Le jeu de données

| Caractéristique | Valeur |
|---|---|
| Nom | UNSW-NB15 (trafic réseau normal et attaques simulées) |
| Source | Kaggle : [`mrwellsdavid/unsw-nb15`](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) |
| Enregistrements bruts | 175 341 (entraînement) + 82 332 (test), soit plus de 257 000 lignes (≥ 10 000 exigées) |
| Attributs | 45 colonnes (≥ 10 exigées) : 42 variables décrivant la connexion, un identifiant (`id`) et 2 colonnes cibles |
| Cibles | `label` (0 = normal, 1 = attaque) et `attack_cat` (type de connexion) |
| Classes de `attack_cat` | Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms |

Répartition de la cible binaire dans le jeu d'entraînement : 56 000 connexions normales (31,9 %) et 119 341 attaques (68,1 %). Les types d'attaques sont fortement déséquilibrés (de 40 000 lignes pour `Generic` à 130 pour `Worms`). Le jeu de test a une composition différente : 55,1 % d'attaques.

> **Remarque sur les fichiers Kaggle.** Dans ce jeu de données, les noms de fichiers sont inversés : `UNSW_NB15_testing-set.csv` contient les 175 341 lignes utilisées pour l'entraînement et `UNSW_NB15_training-set.csv` les 82 332 lignes utilisées pour le test. Cette inversion est documentée dans le code pour éviter toute confusion.

### Structure du dépôt

```
DM-Detection-Intrusions-UNSW-NB15/
|-- README.md
|-- requirements.txt
|-- .streamlit/config.toml        # thème de l'application
|-- data/
|   |-- raw/                      # données brutes (téléchargées automatiquement, non versionnées)
|   |-- processed/                # données transformées (générées par le notebook 02)
|   `-- download_data.py          # script de téléchargement du jeu de données
|-- notebooks/
|   |-- 01_eda_and_cleaning.ipynb
|   |-- 02_preprocessing_encoding.ipynb
|   `-- 03_model_training_evaluation.ipynb
|-- src/
|   |-- app.py                    # application Streamlit (6 pages)
|   |-- preprocessing.py          # contrôle qualité des données
|   `-- models.py                 # chargement du modèle, prédiction unitaire et par lot
|-- models/                       # préprocesseur et modèles sauvegardés
`-- docs/
    |-- Rapport_MiniProject.pdf   # rapport scientifique
    |-- figures/                  # figures générées par les notebooks
    |-- resultats_modeles.csv
    |-- tableau_comparatif.csv
    `-- optimisation_cv.csv
```

---

## 2. Guide d'installation, exécution locale et accès aux données

### Prérequis

- **Git**
- **Python 3.11 ou plus** (projet testé avec Python 3.13 et 3.14)
- Une connexion Internet pour le premier lancement (téléchargement du jeu de données)

### Installation pas à pas

**1. Cloner le dépôt**

```bash
git clone https://github.com/barellina/DM-Detection-Intrusions-UNSW-NB15.git
cd DM-Detection-Intrusions-UNSW-NB15
```

**2. Créer et activer un environnement virtuel**

```bash
python -m venv venv
```

| Système / terminal | Commande d'activation |
|---|---|
| Windows, Git Bash | `source venv/Scripts/activate` |
| Windows, PowerShell | `venv\Scripts\Activate.ps1` |
| macOS / Linux | `source venv/bin/activate` |

Le nom de l'environnement `(venv)` doit apparaître au début de la ligne de commande.

**3. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**4. Lancer l'application**

```bash
streamlit run src/app.py
```

L'application s'ouvre dans le navigateur (par défaut sur `http://localhost:8501`). Pour utiliser un autre port : `streamlit run src/app.py --server.port 8600`.

### Accès aux données

Le jeu de données dépasse la taille conseillée pour GitHub : **il n'est pas versionné** (voir `.gitignore`). Il est téléchargé **automatiquement** depuis Kaggle au premier lancement, grâce à la bibliothèque `kagglehub`, et conservé dans le cache local. Le premier chargement peut donc prendre quelques minutes.

Pour télécharger les données sans lancer l'application, on peut aussi exécuter `data/download_data.py` ou la première cellule du notebook `01_eda_and_cleaning.ipynb`.

### Exécuter les notebooks

```bash
jupyter notebook
```

Les ouvrir dans l'ordre : `01_eda_and_cleaning` (exploration et tests statistiques), `02_preprocessing_encoding` (encodage One-Hot et standardisation, qui génère le préprocesseur `models/preprocessor.joblib`), puis `03_model_training_evaluation` (modèles et évaluation, qui génère les modèles et les tableaux de résultats du dossier `docs/`).

### Fonctionnalités de l'application

| Page | Contenu |
|---|---|
| **Accueil** | Présentation du projet, chiffres clés et description des types de connexion. |
| **Exploration des données** | Statistiques descriptives, valeurs manquantes, distributions des variables numériques, fréquences des variables catégorielles et matrice de corrélation, avec des **filtres interactifs** (type de connexion, durée). |
| **Contrôle qualité** | Types des colonnes, cases vides et valeurs inconnues, doublons, valeurs aberrantes (méthode de l'écart interquartile) et contrôles de cohérence ; fonctionne sur le jeu UNSW-NB15 ou sur un fichier CSV importé. |
| **Prédiction** | Formulaire de saisie d'une connexion (6 variables numériques et 3 catégorielles) ; affiche la probabilité d'attaque, un niveau de risque à 5 échelons et les variables les plus importantes. |
| **Prédiction par lot** | Import d'un fichier CSV, prédiction de toutes les lignes avec un seuil de décision réglable, histogramme des probabilités et export des prédictions ; si le fichier contient `label`, calcule aussi l'accuracy, la précision, le rappel, le F1 et la matrice de confusion. |
| **À propos** | Jeu de données, méthode et équipe. |

---

## 3. Rôles des membres et matrice des contributions

Conformément à la règle d'anonymat, seuls les **codes étudiants** sont utilisés.

| Code | Rôle | Responsabilités et contributions |
|---|---|---|
| **CS01** | Team Leader | Création du dépôt et de sa structure ; exploration des données et tests statistiques de base (`01_eda_and_cleaning`) ; prétraitement (`02_preprocessing_encoding`) ; entraînement, optimisation et évaluation des modèles (`03_model_training_evaluation`) ; moteur de prédiction (`models.py`, `valeurs_defaut.json`). |
| **CS19** | Membre | Application Streamlit (navigation, tableau de bord d'exploration, filtres) ; tests de Shapiro-Wilk et de Kruskal-Wallis et analyse des doublons (`01_eda_and_cleaning`) ; module de contrôle qualité et page associée ; pages de prédiction unitaire et de prédiction par lot ; rédaction du rapport scientifique et de ce README. |

L'historique des commits du dépôt (branches et Pull Requests par fonctionnalité) reflète cette répartition.

---

## 4. Résultats clés et synthèse des découvertes

### Résultats de l'exploration des données

- **Valeurs manquantes** : aucune case vide, mais 53,7 % des connexions ont un service non identifié (valeur `-` dans `service`), conservé comme catégorie à part entière.
- **Doublons** : 38,6 % des lignes d'entraînement et 32,0 % des lignes de test sont des copies d'une autre ligne du même jeu (identifiant ignoré). Seules 5,2 % des lignes du test existent à l'identique dans l'entraînement : le test n'est donc que faiblement contaminé.
- **Déséquilibre des classes** : 68,1 % d'attaques dans l'entraînement contre 55,1 % dans le test, avec des types d'attaques très inégaux. Le **F1-score** et le **ROC-AUC** sont donc privilégiés à la seule exactitude (accuracy).
- **Valeurs extrêmes** : les variables de volume (`sbytes`, `dbytes`, `rate`) sont très asymétriques. Les valeurs extrêmes correspondent souvent à des attaques : elles sont **conservées**.
- **Variables les plus liées à la cible** (corrélation de Spearman avec `label`) : `sttl` (+0,73), `ct_state_ttl` (+0,64), `dload` (-0,63), `dmean` (-0,55).
- **Prétraitement** : encodage One-Hot de `proto`, `service` et `state` (les valeurs rares sont regroupées) et standardisation des variables numériques, calculés sur le jeu d'entraînement uniquement pour éviter toute fuite de données (42 colonnes avant, 60 après).

### Tests statistiques et justification des choix

Le test de Shapiro-Wilk montre que les variables numériques ne suivent pas une loi normale (même après transformation logarithmique), ce qui impose des tests **non paramétriques**.

| Test | Pourquoi ce test | Résultat |
|---|---|---|
| **Shapiro-Wilk** (normalité, échantillon de 5 000 lignes) | Conditionne le choix de tests paramétriques ou non | H0 rejetée pour `dur`, `sbytes`, `dbytes`, `rate`, `sttl` (W de 0,03 à 0,64 ; p < 10⁻⁷²), même après log(1+x) |
| **Mann-Whitney U** (variables numériques contre `label`) | Compare deux groupes sans supposer de distribution normale | Différence très significative (p < 0,001) : médiane de `sttl` de 31 (normal) contre 254 (attaque) ; médiane de `dbytes` de 1 112 contre 0 |
| **Kruskal-Wallis** (variables selon les 10 types `attack_cat`) | Extension du précédent à plus de 2 groupes, alternative non paramétrique à l'ANOVA | H0 rejetée pour les 5 variables testées ; taille d'effet η² de 0,30 à 0,59 (`sttl` la plus discriminante) |
| **Chi² d'indépendance** (`service` contre `label`) et **V de Cramér** | Teste le lien entre deux variables catégorielles ; le V mesure la force du lien, car avec 175 000 lignes la p-value seule est toujours significative | χ² = 12 985,7 ; 12 ddl ; p < 0,001 ; V de Cramér = 0,272 (lien modéré) |
| **Corrélation de Spearman** | Mesure le lien entre variables sans supposer de relation linéaire, et résiste aux valeurs extrêmes | Voir les variables les plus liées à la cible ci-dessus |

### Comparaison des modèles

Résultats sur le jeu de test (82 332 lignes), triés par F1-score décroissant :

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0,8728 | 0,8196 | 0,9860 | 0,8951 | 0,9796 |
| Random Forest (optimisé) | 0,8708 | 0,8159 | 0,9882 | 0,8939 | 0,9824 |
| Arbre de décision (optimisé) | 0,8718 | 0,8297 | 0,9651 | 0,8923 | 0,9135 |
| Arbre de décision | 0,8629 | 0,8241 | 0,9549 | 0,8847 | 0,8550 |
| K-NN (optimisé) | 0,8460 | 0,7917 | 0,9777 | 0,8749 | 0,9651 |
| K-NN (k = 5) | 0,8448 | 0,7966 | 0,9644 | 0,8725 | 0,9315 |
| SVM (optimisé, 50 000 lignes) | 0,8152 | 0,7509 | 0,9941 | 0,8555 | 0,9583 |
| SVM RBF (50 000 lignes) | 0,8145 | 0,7501 | 0,9944 | 0,8551 | 0,9574 |
| SVM linéaire (LinearSVC) | 0,8111 | 0,7504 | 0,9842 | 0,8516 | 0,9537 |
| Naïve Bayes gaussien (défaut = optimisé) | 0,7693 | 0,8601 | 0,6939 | 0,7681 | 0,8698 |
| Naïve Bayes Bernoulli (optimisé) | 0,7478 | 0,8418 | 0,6674 | 0,7445 | 0,8184 |
| Naïve Bayes Bernoulli (défaut) | 0,7478 | 0,8418 | 0,6674 | 0,7445 | 0,8173 |
| Zero-R (référence) | 0,5506 | 0,5506 | 1,0000 | 0,7102 | 0,5000 |

### Choix d'optimisation des hyperparamètres

Chaque modèle réglable est optimisé par `GridSearchCV` (grille exhaustive) ou `RandomizedSearchCV` (tirage aléatoire), avec validation croisée stratifiée à **5 plis** et le F1-score comme critère (Zero-R est exempté, conformément au cahier des charges). Pour K-NN, le Random Forest et le SVM, dont le coût de calcul est élevé, la recherche est faite sur un sous-échantillon stratifié du jeu d'entraînement, puis le meilleur réglage est réentraîné sur l'ensemble du train (K-NN, Random Forest) ou sur 50 000 lignes (SVM) avant l'évaluation finale sur le test.

| Modèle | Données de recherche | Meilleurs hyperparamètres | F1 (validation croisée, 5 plis) |
|---|---|---|---|
| Arbre de décision | 175 341 lignes | `criterion=entropy, max_depth=20, min_samples_leaf=1` | 0,9638 |
| Naïve Bayes gaussien | 175 341 lignes | `var_smoothing=1e-05` | 0,8078 |
| Naïve Bayes Bernoulli | 175 341 lignes | `alpha=0.01` | 0,7813 |
| K-NN | sous-échantillon de 20 000 | `metric=manhattan, n_neighbors=21, weights=distance` | 0,9568 |
| Random Forest | sous-échantillon de 40 000 | `n_estimators=300, max_depth=30, min_samples_leaf=2, max_features=sqrt, class_weight=None` | 0,9639 |
| SVM | sous-échantillon de 6 000 | `kernel=rbf, C=10, gamma=0.01` | 0,9559 |

Ces scores de validation croisée (environ 0,96 pour les quatre meilleurs modèles) sont nettement supérieurs aux scores du test (environ 0,89) : cet écart est analysé dans le rapport scientifique (`docs/Rapport_MiniProject.pdf`, section 6.5) et rapproché de deux causes possibles, non exclusives : 38,6 % de lignes dupliquées dans l'entraînement (une ligne et sa copie peuvent se retrouver dans deux plis différents de la validation croisée) et le changement de proportion d'attaques entre l'entraînement et le test (68,1 % contre 55,1 %).

### Conclusion

Le **Random Forest optimisé** est le modèle retenu pour l'application : il obtient le meilleur ROC-AUC (0,9824), un rappel élevé (98,8 %) et un F1-score (0,8939) statistiquement équivalent au meilleur résultat, sans avoir besoin de parcourir les données d'entraînement à chaque prédiction (contrairement à K-NN et au SVM). Les modèles à base d'arbres (Random Forest, arbre de décision) dominent largement Naïve Bayes, dont l'hypothèse d'indépendance des variables est peu réaliste ici.

Deux limites principales ressortent, discutées en détail dans le rapport : un **taux de fausses alarmes élevé** (27,3 % des connexions normales classées à tort comme attaques pour le Random Forest optimisé) et un **score de validation croisée optimiste** par rapport au test. Les perspectives retenues sont le réglage du seuil de décision sur un jeu de validation séparé, la déduplication des données, la construction d'un unique `Pipeline` scikit-learn (préprocesseur + modèle) pour supprimer toute fuite dans la validation croisée, et l'extension du benchmark à des modèles de boosting (XGBoost) ainsi qu'à la classification multiclasse sur `attack_cat` pour mieux évaluer la détection des catégories d'attaques les plus rares (`Worms`, `Shellcode`, `Backdoor`).
