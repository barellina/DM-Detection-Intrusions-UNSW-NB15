
import os
 
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
 
st.set_page_config(
    page_title="Détection d'intrusions UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ---------------------------------------------------------------------------
# Palette de couleurs (comme dans les outils de supervision : vert = normal,
# rouge/orange = attaque). Chaque type de connexion garde la même couleur
# dans tous les graphiques.
# ---------------------------------------------------------------------------
COULEUR_NORMAL = "#2ecc71"
COULEUR_ATTAQUE = "#e74c3c"
COULEUR_NEUTRE = "#1f77b4"
 
PALETTE_CLASSES = {"Normal": COULEUR_NORMAL, "Attaque": COULEUR_ATTAQUE}
 
PALETTE_TYPES = {
    "Normal": "#2ecc71",
    "Generic": "#e74c3c",
    "Exploits": "#e67e22",
    "Fuzzers": "#f1c40f",
    "DoS": "#c0392b",
    "Reconnaissance": "#9b59b6",
    "Analysis": "#1abc9c",
    "Backdoor": "#ff6f91",
    "Shellcode": "#00a8cc",
    "Worms": "#8d6e63",
}
 
 
# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement du dataset (le premier lancement peut être long)...")
def load_data():
    path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
    # Dans ce dataset Kaggle, les noms de fichiers sont inversés :
    # "testing-set" contient le gros fichier (175 341 lignes)
    return pd.read_csv(os.path.join(path, "UNSW_NB15_testing-set.csv"))
 
 
def style_graphiques():
    """Couleurs des graphiques matplotlib/seaborn adaptées au thème clair ou sombre."""
    try:
        sombre = st.context.theme.type == "dark"
    except Exception:
        sombre = False  # version de Streamlit trop ancienne : on reste en clair
    couleur = "#e8ecf4" if sombre else "#1c2333"
    return {
        "text.color": couleur,
        "axes.labelcolor": couleur,
        "axes.edgecolor": couleur,
        "xtick.color": couleur,
        "ytick.color": couleur,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
    }
 
 
@st.cache_data
def calculer_correlation(data, methode):
    num = data.select_dtypes(include="number").drop(columns=["id"])
    echantillon = num.sample(min(len(num), 30000), random_state=42)
    return echantillon.corr(method=methode)
 
 
def format_nombre(n):
    """Écrit 175341 sous la forme 175 341."""
    return f"{int(n):,}".replace(",", " ")
 
 
def astuce_theme():
    """Rappel visible : Streamlit ne permet de changer de thème que par son menu."""
    st.caption("🎨 Thème clair / sombre : menu ⋮ en haut à droite → Settings → Theme")
 
 
# ===========================================================================
# PAGE 1 : ACCUEIL
# ===========================================================================
def page_accueil():
    df_complet = load_data()
 
    st.title("🛡️ Détection d'intrusions réseau par Machine Learning")
    st.caption("Mini-projet Data Mining 2026/2027")
    astuce_theme()
 
    st.markdown(
        """
        Un **système de détection d'intrusions (IDS)** surveille le trafic d'un réseau
        pour repérer les connexions malveillantes. Ce projet entraîne des modèles de
        **classification supervisée** qui décident, à partir des caractéristiques
        techniques d'une connexion (durée, volume d'octets, protocole, service...),
        si elle est **normale** ou si elle correspond à une **attaque**.
        """
    )
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Connexions étudiées", format_nombre(len(df_complet)))
    c2.metric("Colonnes", df_complet.shape[1])
    c3.metric("Types de connexion", df_complet["attack_cat"].nunique())
 
    st.subheader("Ce que contient l'application")
    st.markdown(
        """
        - **📊 Exploration des données** : statistiques, valeurs manquantes, distributions,
          fréquences, corrélations, avec des filtres interactifs.
        - **ℹ️ À propos** : le dataset, la méthode et l'équipe.
 
        👉 Utilise le **menu à gauche** pour naviguer entre les pages.
        """
    )
 
    st.subheader("Les types de connexion du dataset")
    descriptions = pd.DataFrame(
        {
            "Type": [
                "Normal",
                "Generic",
                "Exploits",
                "Fuzzers",
                "DoS",
                "Reconnaissance",
                "Analysis",
                "Backdoor",
                "Shellcode",
                "Worms",
            ],
            "Description": [
                "Trafic légitime, sans attaque.",
                "Attaque générique visant les mécanismes de chiffrement.",
                "Exploitation d'une faille connue d'un logiciel ou d'un système.",
                "Envoi de données aléatoires pour faire planter un service.",
                "Déni de service : saturer une machine pour la rendre indisponible.",
                "Collecte d'informations sur le réseau avant une attaque.",
                "Analyse du réseau : balayage de ports, intrusion dans des fichiers.",
                "Accès caché qui contourne l'authentification normale.",
                "Petit code malveillant injecté pour prendre le contrôle d'une machine.",
                "Programme qui se propage seul d'une machine à l'autre.",
            ],
        }
    )
    effectifs = df_complet["attack_cat"].value_counts()
    descriptions["Connexions"] = descriptions["Type"].map(effectifs).apply(format_nombre)
    st.dataframe(descriptions, hide_index=True, use_container_width=True)
 
 
# ===========================================================================
# PAGE 2 : EXPLORATION DES DONNÉES (EDA)
# ===========================================================================
def page_exploration():
    df_complet = load_data()
 
    # -----------------------------------------------------------------------
    # Barre latérale : filtres interactifs
    # Le DataFrame filtré s'appelle "df" : tous les onglets se mettent à jour
    # automatiquement quand on change les filtres.
    # -----------------------------------------------------------------------
    st.sidebar.header("🔎 Filtres")
 
    types = sorted(df_complet["attack_cat"].unique())
    choix_types = st.sidebar.multiselect("Type de connexion :", types, default=types)
 
    duree_max_totale = float(df_complet["dur"].max())
    duree_min, duree_max = st.sidebar.slider(
        "Durée de la connexion (secondes) :",
        min_value=0.0,
        max_value=duree_max_totale,
        value=(0.0, duree_max_totale),
    )
 
    masque = df_complet["attack_cat"].isin(choix_types) & df_complet["dur"].between(
        duree_min, duree_max
    )
    df = df_complet[masque]
 
    if df.empty:
        st.warning("Aucune ligne ne correspond à ces filtres.")
        st.stop()
 
    st.sidebar.success(
        f"{format_nombre(len(df))} lignes sélectionnées sur {format_nombre(len(df_complet))}"
    )
 
    # -----------------------------------------------------------------------
    # En-tête et cartes de chiffres clés
    # -----------------------------------------------------------------------
    st.title("📊 Exploration des données")
    st.caption("Dashboard d'exploration (EDA) - dataset UNSW-NB15")
    astuce_theme()
 
    nb_attaques = int((df["label"] == 1).sum())
    nb_normales = int((df["label"] == 0).sum())
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Connexions", format_nombre(len(df)))
    c2.metric("🟢 Normales", format_nombre(nb_normales))
    c3.metric("🔴 Attaques", format_nombre(nb_attaques))
    c4.metric("Part d'attaques", f"{df['label'].mean() * 100:.1f} %")
 
    # -----------------------------------------------------------------------
    # Onglets
    # -----------------------------------------------------------------------
    onglet0, onglet1, onglet2, onglet3, onglet4, onglet5 = st.tabs(
        [
            "🏠 Vue d'ensemble",
            "📊 Statistiques",
            "❓ Valeurs manquantes",
            "📈 Distributions",
            "🏷️ Catégories",
            "🔗 Corrélation",
        ]
    )
 
    # --- Vue d'ensemble : anneau normal/attaque et classement des types -----
    with onglet0:
        st.subheader("Vue d'ensemble")
 
        gauche, droite = st.columns([1, 2])
 
        with gauche:
            st.write("Répartition normal / attaque :")
            with plt.rc_context(style_graphiques()):
                fig, ax = plt.subplots(figsize=(4.5, 4.5))
                ax.pie(
                    [nb_normales, nb_attaques],
                    labels=["Normal", "Attaque"],
                    colors=[COULEUR_NORMAL, COULEUR_ATTAQUE],
                    autopct="%1.1f%%",
                    startangle=90,
                    wedgeprops={"width": 0.4},
                    textprops={"color": plt.rcParams["text.color"]},
                )
                ax.set_aspect("equal")
                st.pyplot(fig)
                plt.close(fig)
 
        with droite:
            st.write("Nombre de connexions par type :")
            effectifs = df["attack_cat"].value_counts().sort_values()
            couleurs = [PALETTE_TYPES.get(t, "#888888") for t in effectifs.index]
            with plt.rc_context(style_graphiques()):
                fig, ax = plt.subplots(figsize=(7, 4.5))
                barres = ax.barh(effectifs.index, effectifs.values, color=couleurs)
                ax.bar_label(barres, padding=3)
                ax.set_xlabel("Nombre de connexions")
                ax.margins(x=0.12)
                st.pyplot(fig)
                plt.close(fig)
 
    # --- Élément 1 : statistiques descriptives -----------------------------
    with onglet1:
        st.subheader("Statistiques descriptives")
        st.write("Aperçu des premières lignes :")
        st.dataframe(df.head(10), use_container_width=True)
 
        stats = df.describe().T
        st.write("Résumé de chaque variable numérique :")
        st.dataframe(stats, use_container_width=True)
        st.download_button(
            "Télécharger les statistiques (CSV)",
            stats.to_csv().encode("utf-8"),
            file_name="statistiques_descriptives.csv",
            mime="text/csv",
        )
 
    # --- Élément 2 : valeurs manquantes ------------------------------------
    with onglet2:
        st.subheader("Valeurs manquantes")
 
        manquantes = df.isnull().sum()
        tableau_manquantes = pd.DataFrame(
            {
                "Valeurs manquantes": manquantes,
                "Taux de présence (%)": ((1 - manquantes / len(df)) * 100).round(2),
            }
        )
        st.write(f"Total de cases vides : {int(manquantes.sum())}")
 
        gauche, droite = st.columns(2)
        with gauche:
            st.write("Taux de présence par colonne :")
            st.dataframe(tableau_manquantes, use_container_width=True)
        with droite:
            # Valeurs inconnues cachées sous forme de tiret "-"
            tirets = (df.select_dtypes(exclude="number") == "-").sum()
            st.write("Valeurs inconnues notées « - » (colonnes texte) :")
            st.bar_chart(tirets, color=COULEUR_NEUTRE)
 
    # --- Élément 3 : distributions des variables numériques ----------------
    with onglet3:
        st.subheader("Distributions des variables numériques")
 
        colonnes_num = [
            c for c in df.select_dtypes(include="number").columns if c not in ["id", "label"]
        ]
 
        gauche, droite = st.columns([2, 1])
        with gauche:
            col = st.selectbox("Choisis une variable numérique :", colonnes_num)
        with droite:
            echelle_log = st.checkbox("Échelle logarithmique", value=True)
            st.caption("Conseillée pour les variables très étalées.")
 
        donnees = df[col]
        if echelle_log:
            donnees = np.log1p(donnees)
 
        tableau_dist = pd.DataFrame(
            {"valeur": donnees, "Classe": df["label"].map({0: "Normal", 1: "Attaque"})}
        )
        tableau_dist = tableau_dist.sample(min(len(tableau_dist), 20000), random_state=42)
 
        with plt.rc_context(style_graphiques()):
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.histplot(
                data=tableau_dist,
                x="valeur",
                hue="Classe",
                palette=PALETTE_CLASSES,
                bins=50,
                kde=True,
                stat="density",
                common_norm=False,
                element="step",
                ax=ax,
            )
            ax.set_xlabel(f"log(1 + {col})" if echelle_log else col)
            ax.set_ylabel("Densité")
            st.pyplot(fig)
            plt.close(fig)
 
    # --- Élément 4 : fréquences des variables catégorielles ----------------
    with onglet4:
        st.subheader("Fréquences des variables catégorielles")
 
        colonnes_cat = ["proto", "service", "state", "attack_cat"]
 
        gauche, droite = st.columns(2)
        with gauche:
            col_cat = st.selectbox("Choisis une variable catégorielle :", colonnes_cat)
        with droite:
            nb_max = st.slider("Nombre de catégories à afficher :", 3, 20, 10)
 
        frequences = df[col_cat].value_counts()
        st.write(f"La colonne « {col_cat} » contient {frequences.shape[0]} valeurs différentes.")
 
        tableau_freq = pd.DataFrame(
            {
                "Effectif": frequences,
                "Pourcentage (%)": (frequences / len(df) * 100).round(2),
            }
        )
 
        gauche, droite = st.columns([2, 1])
        with gauche:
            st.bar_chart(frequences.head(nb_max), color=COULEUR_NEUTRE)
        with droite:
            st.dataframe(tableau_freq.head(nb_max), use_container_width=True)
 
    # --- Élément 5 : matrice de corrélation --------------------------------
    with onglet5:
        st.subheader("Matrice de corrélation")
 
        methode = st.radio("Méthode de corrélation :", ["spearman", "pearson"], horizontal=True)
        corr = calculer_correlation(df, methode)
 
        gauche, droite = st.columns([3, 1])
        with gauche:
            with plt.rc_context(style_graphiques()):
                fig, ax = plt.subplots(figsize=(13, 10))
                sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
                ax.set_title(f"Matrice de corrélation ({methode})")
                st.pyplot(fig)
                plt.close(fig)
        with droite:
            st.write("Les 10 variables les plus corrélées avec la cible (label) :")
            top = (
                corr["label"].drop("label").dropna().sort_values(key=abs, ascending=False).head(10)
            )
            st.dataframe(top.rename("Corrélation"), use_container_width=True)
 
 
# ===========================================================================
# PAGE 3 : À PROPOS
# ===========================================================================
def page_apropos():
    st.title("ℹ️ À propos du projet")
    astuce_theme()
 
    st.subheader("Le dataset")
    st.markdown(
        """
        - **Nom** : UNSW-NB15 (trafic réseau normal et attaques simulées)
        - **Source** : Kaggle, `mrwellsdavid/unsw-nb15`
        - **Taille** : 175 341 connexions pour l'entraînement et 82 332 pour le test
        - **Colonnes** : 45, dont 42 variables décrivant la connexion et 2 cibles
        - **Cibles** : `label` (0 = normal, 1 = attaque) et `attack_cat` (type de connexion)
        """
    )
 
    st.subheader("La méthode")
    st.markdown(
        """
        - **Parcours** : classification supervisée (cible binaire : normal ou attaque)
        - **Exploration** : statistiques descriptives, distributions, corrélations de Spearman
        - **Tests statistiques** : Mann-Whitney U (variables numériques) et Chi² d'indépendance
          avec V de Cramér (variables catégorielles)
        - **Prétraitement** : encodage One-Hot des variables texte et standardisation des variables
          numériques, calculés sur le jeu d'entraînement uniquement pour éviter toute fuite de données
        """
    )
 
    st.subheader("L'équipe")
    equipe = pd.DataFrame(
        {
            "Code étudiant": ["CS01", "CS19"],
            "Rôle": ["Team Leader", "Membre"],
        }
    )
    st.dataframe(equipe, hide_index=True)
 
 
# ===========================================================================
# NAVIGATION
# ===========================================================================
pages = [
    st.Page(page_accueil, title="Accueil", icon="🏠", default=True),
    st.Page(page_exploration, title="Exploration des données", icon="📊", url_path="exploration"),
    st.Page(page_apropos, title="À propos", icon="ℹ️", url_path="apropos"),
]
 
navigation = st.navigation(pages, position="top")
navigation.run()
 

