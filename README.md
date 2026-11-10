# DM-Detection-Intrusions-UNSW-NB15# Détection d'intrusions réseau par Machine Learning

Mini-projet de Data Mining (année académique 2026/2027) : classification supervisée de connexions réseau (normales ou malveillantes) sur le jeu de données **UNSW-NB15**.

---

## 1. Présentation, objectifs et description du projet

### Contexte métier

Un **système de détection d'intrusions (IDS)** surveille le trafic d'un réseau pour repérer les connexions malveillantes. Ce projet, dans le domaine de la **cybersécurité**, construit des modèles qui décident, à partir des caractéristiques techniques d'une connexion (durée, volume d'octets, protocole, service...), si elle est **normale** ou si elle correspond à une **attaque**.

### Objectifs

**Objectifs scientifiques**
- Explorer et nettoyer les données (statistiques descriptives, distributions, corrélations).
- Réaliser des tests statistiques justifiés (Mann-Whitney U, Chi² d'indépendance avec V de Cramér, corrélation de Spearman).
- Comparer les algorithmes de classification du cours : Zero-R (référence), K-NN, arbre de décision, Random Forest, Naïve Bayes et SVM.
- Optimiser les hyperparamètres par validation croisée (`GridSearchCV` ou `RandomizedSearchCV`, K ≥ 5) et évaluer avec des métriques adaptées au déséquilibre des classes (F1, ROC-AUC, matrices de confusion).

**Objectifs applicatifs**
- Fournir une application **Streamlit** interactive : tableau de bord d'exploration des données avec filtres dynamiques.

### Le jeu de données

| Caractéristique | Valeur |
|---|---|
| Nom | UNSW-NB15 (trafic réseau normal et attaques simulées) |
| Source | Kaggle : [`mrwellsdavid/unsw-nb15`](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) |
| Enregistrements bruts | 175 341 (entraînement) + 82 332 (test), soit plus de 257 000 lignes (≥ 10 000 exigées) |
| Attributs | 45 colonnes (≥ 10 exigées) : 42 variables décrivant la connexion, un identifiant (`id`) et 2 colonnes cibles |
| Cibles | `label` (0 = normal, 1 = attaque) et `attack_cat` (type de connexion) |
| Classes de `attack_cat` | Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms |

Répartition de la cible binaire dans le jeu d'entraînement : 56 000 connexions normales (31,9 %) et 119 341 attaques (68,1 %). Les types d'attaques sont fortement déséquilibrés (de 40 000 lignes pour `Generic` à 130 pour `Worms`).

> **Remarque sur les fichiers Kaggle.** Dans ce jeu de données, les noms de fichiers sont inversés : `UNSW_NB15_testing-set.csv` contient les 175 341 lignes utilisées pour l'entraînement et `UNSW_NB15_training-set.csv` les 82 332 lignes utilisées pour le test.

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
|   |-- app.py                    # application Streamlit
|   |-- preprocessing.py
|   `-- models.py
|-- models/                       # préprocesseur et modèles sauvegardés
`-- docs/
    `-- Rapport_MiniProject.pdf   # rapport scientifique
```

---

## 2. Guide d'installation, exécution locale et accès aux données

### Prérequis

- **Git**
- **Python 3.14** (projet testé avec Python 3.14.7)
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

Les ouvrir dans l'ordre : `01_eda_and_cleaning` (exploration et tests statistiques), `02_preprocessing_encoding` (encodage One-Hot et standardisation, qui génère le préprocesseur `models/preprocessor.joblib`), puis `03_model_training_evaluation` (modèles et évaluation).

### Fonctionnalités de l'application

- **Accueil** : présentation du projet, chiffres clés et description des types de connexion.
- **Exploration des données** : statistiques descriptives, valeurs manquantes, distributions des variables numériques, fréquences des variables catégorielles et matrice de corrélation, avec des **filtres interactifs** (type de connexion, durée) dans la barre latérale.
- **À propos** : jeu de données, méthode et équipe.

---

## 3. Rôles des membres et matrice des contributions

Conformément à la règle d'anonymat, seuls les **codes étudiants** sont utilisés.

| Code | Rôle | Responsabilités et contributions |
|---|---|---|
| **CS01** | Team Leader | Création du dépôt et de sa structure ; exploration des données et tests statistiques (`01_eda_and_cleaning`) ; prétraitement (`02_preprocessing_encoding`) ; entraînement, optimisation et évaluation des modèles (`03_model_training_evaluation`) |
| **CS19** | Membre | Application Streamlit (`src/app.py`) : navigation, tableau de bord d'exploration, filtres interactifs ; rédaction du README |

L'historique des commits du dépôt reflète ces contributions.

---

## 4. Résultats clés et synthèse des découvertes

### Résultats de l'exploration des données

- **Valeurs manquantes** : aucune case vide, mais 53,7 % des connexions ont un service non identifié (valeur `-` dans `service`), conservé comme catégorie à part entière.
- **Déséquilibre des classes** : 68,1 % d'attaques en binaire, avec des types d'attaques très inégaux. Le **F1-score** est donc privilégié à la seule exactitude (accuracy).
- **Valeurs extrêmes** : les variables de volume (`sbytes`, `dbytes`, `rate`) sont très asymétriques. Les valeurs extrêmes correspondent souvent à des attaques : elles sont **conservées**.
- **Variables les plus liées à la cible** (corrélation de Spearman avec `label`) : `sttl` (+0,73), `ct_state_ttl` (+0,64), `dload` (-0,63), `dmean` (-0,55).
- **Prétraitement** : encodage One-Hot de `proto`, `service` et `state` (les valeurs rares sont regroupées) et standardisation des variables numériques, calculés sur le jeu d'entraînement uniquement pour éviter toute fuite de données (42 colonnes avant, 60 après).

### Tests statistiques et justification des choix

Les distributions n'étant pas normales (histogrammes asymétriques et multimodaux, même après transformation logarithmique), des tests **non paramétriques** sont utilisés.

| Test | Pourquoi ce test | Résultat |
|---|---|---|
| **Mann-Whitney U** (variables numériques contre `label`) | Compare deux groupes sans supposer de distribution normale | Différence très significative (p < 0,001) : médiane de `sttl` de 31 (normal) contre 254 (attaque) ; médiane de `dbytes` de 1 112 contre 0 ; médiane de `dur` de 0,0386 contre 0 |
| **Chi² d'indépendance** (`service` contre `label`) et **V de Cramér** | Teste le lien entre deux variables catégorielles ; le V mesure la force du lien, car avec 175 000 lignes la p-value seule est toujours significative | χ² = 12 985,7 ; 12 ddl ; p < 0,001 ; V de Cramér = 0,272 (lien modéré) |
| **Corrélation de Spearman** | Mesure le lien entre variables sans supposer de relation linéaire, et résiste aux valeurs extrêmes | Voir les variables les plus liées à la cible ci-dessus |

### Comparaison des modèles

> **À compléter** avec les résultats du notebook `03_model_training_evaluation` (évaluation sur le jeu de test de 82 332 lignes).

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC | Meilleurs hyperparamètres |
|---|---|---|---|---|---|---|
| Zero-R (référence) | | | | | | non applicable |
| K-NN | | | | | | |
| Arbre de décision | | | | | | |
| Random Forest | | | | | | |
| Naïve Bayes | | | | | | |
| SVM | | | | | | |

### Choix d'optimisation des hyperparamètres

> **À compléter** : méthode utilisée (`GridSearchCV` ou `RandomizedSearchCV`, K ≥ 5 plis), grilles de recherche pour chaque modèle, et justification (par exemple, réglage de K-NN et du SVM sur un sous-échantillon à cause du temps de calcul).

### Conclusion

> **À compléter** : modèle retenu, interprétation des résultats et limites.