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
)


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


df_complet = load_data()

# ---------------------------------------------------------------------------
# Barre latérale : filtres interactifs
# Le DataFrame filtré s'appelle "df" : tous les onglets se mettent à jour
# automatiquement quand on change les filtres.
# ---------------------------------------------------------------------------
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

st.sidebar.success(f"{len(df):,} lignes sélectionnées sur {len(df_complet):,}".replace(",", " "))

# ---------------------------------------------------------------------------
# En-tête et cartes de chiffres clés
# ---------------------------------------------------------------------------
st.title("🛡️ Détection d'intrusions réseau")
st.caption("Dashboard d'exploration des données (EDA) - dataset UNSW-NB15")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Connexions", f"{len(df):,}".replace(",", " "))
c2.metric("Colonnes", df.shape[1])
c3.metric("Part d'attaques", f"{df['label'].mean() * 100:.1f} %")
c4.metric("Types de connexion", df["attack_cat"].nunique())

# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------
onglet1, onglet2, onglet3, onglet4, onglet5 = st.tabs(
    [
        "📊 Statistiques",
        "❓ Valeurs manquantes",
        "📈 Distributions",
        "🏷️ Catégories",
        "🔗 Corrélation",
    ]
)

# --- Élément 1 : statistiques descriptives ---------------------------------
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

# --- Élément 2 : valeurs manquantes ----------------------------------------
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
        st.bar_chart(tirets)

# --- Élément 3 : distributions des variables numériques --------------------
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
    echantillon = donnees.sample(min(len(donnees), 20000), random_state=42)

    with plt.rc_context(style_graphiques()):
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(echantillon, bins=50, kde=True, ax=ax, color="#1f77b4")
        ax.set_xlabel(f"log(1 + {col})" if echelle_log else col)
        ax.set_ylabel("Nombre de lignes")
        st.pyplot(fig, transparent=True)
        plt.close(fig)

# --- Élément 4 : fréquences des variables catégorielles --------------------
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
        st.bar_chart(frequences.head(nb_max))
    with droite:
        st.dataframe(tableau_freq.head(nb_max), use_container_width=True)

# --- Élément 5 : matrice de corrélation ------------------------------------
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
            st.pyplot(fig, transparent=True)
            plt.close(fig)
    with droite:
        st.write("Les 10 variables les plus corrélées avec la cible (label) :")
        top = corr["label"].drop("label").dropna().sort_values(key=abs, ascending=False).head(10)
        st.dataframe(top.rename("Corrélation"), use_container_width=True)

st.divider()
st.caption("Mini-projet Data Mining 2026/2027 - Détection d'intrusions réseau par Machine Learning")
