import os
import pandas as pd
import streamlit as st
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Détection d'intrusions UNSW-NB15", layout="wide")


@st.cache_data
def load_data():
    path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
    # Dans ce dataset Kaggle, les noms de fichiers sont inversés
    return pd.read_csv(os.path.join(path, "UNSW_NB15_testing-set.csv"))


df = load_data()

st.title("Dashboard EDA - Détection d'intrusions réseau")
st.write(f"{df.shape[0]} lignes et {df.shape[1]} colonnes")

st.subheader("1. Statistiques descriptives")
st.dataframe(df.describe().T)

st.subheader("2. Valeurs manquantes")


manquantes = df.isnull().sum()
tableau = pd.DataFrame({
    "Valeurs manquantes": manquantes,
    "Taux de présence (%)": ((1 - manquantes / len(df)) * 100).round(2),
})
st.write(f"Total de cases vides : {int(manquantes.sum())}")
st.dataframe(tableau)


tirets = (df.select_dtypes(exclude="number") == "-").sum()
st.write("Valeurs inconnues notées « - » (colonnes texte) :")
st.bar_chart(tirets)

st.subheader("3. Distributions des variables numériques")

colonnes_num = [c for c in df.select_dtypes(include="number").columns
                if c not in ["id", "label"]]

col = st.selectbox("Choisis une variable numérique :", colonnes_num)
echelle_log = st.checkbox("Échelle logarithmique (conseillée pour les variables très étalées)",
                          value=True)

donnees = df[col]
if echelle_log:
    donnees = np.log1p(donnees)

echantillon = donnees.sample(min(len(donnees), 20000), random_state=42)

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(echantillon, bins=50, kde=True, ax=ax)
ax.set_xlabel(f"log(1 + {col})" if echelle_log else col)
ax.set_ylabel("Nombre de lignes")
st.pyplot(fig)
plt.close(fig)

st.subheader("4. Fréquences des variables catégorielles")

colonnes_cat = ["proto", "service", "state", "attack_cat"]

col_cat = st.selectbox("Choisis une variable catégorielle :", colonnes_cat)
nb_max = st.slider("Nombre de catégories à afficher :", min_value=3, max_value=20, value=10)

frequences = df[col_cat].value_counts()
st.write(f"La colonne « {col_cat} » contient {frequences.shape[0]} valeurs différentes.")

st.bar_chart(frequences.head(nb_max))

tableau = pd.DataFrame({
    "Effectif": frequences,
    "Pourcentage (%)": (frequences / len(df) * 100).round(2),
})
st.dataframe(tableau.head(nb_max))

st.subheader("5. Matrice de corrélation")

methode = st.radio("Méthode de corrélation :", ["spearman", "pearson"], horizontal=True)


@st.cache_data
def calculer_correlation(methode):
    num = df.select_dtypes(include="number").drop(columns=["id"])
    echantillon = num.sample(min(len(num), 30000), random_state=42)
    return echantillon.corr(method=methode)


corr = calculer_correlation(methode)

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
ax.set_title(f"Matrice de corrélation ({methode})")
st.pyplot(fig)
plt.close(fig)

st.write("Les 10 variables les plus corrélées avec la cible (label) :")
top = corr["label"].drop("label").sort_values(key=abs, ascending=False).head(10)
st.dataframe(top.rename("Corrélation"))