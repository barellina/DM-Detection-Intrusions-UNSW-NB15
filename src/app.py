import os
import pandas as pd
import streamlit as st
import kagglehub

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
