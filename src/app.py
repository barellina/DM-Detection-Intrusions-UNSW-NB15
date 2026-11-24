import os
 
import altair as alt
import kagglehub
import numpy as np
import pandas as pd
import streamlit as st

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import exemples_doublons, rapport_qualite  # noqa: E402
 
st.set_page_config(
    page_title="Détection d'intrusions UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 

COULEUR_NORMAL = "#2ecc71"
COULEUR_ATTAQUE = "#e74c3c"
COULEUR_NEUTRE = "#1f77b4"
COULEUR_ORANGE = "#e67e22"
COULEUR_VIOLET = "#9b59b6"
COULEUR_TURQUOISE = "#1abc9c"
 
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
 
ECHELLE_CLASSES = alt.Scale(
    domain=list(PALETTE_CLASSES.keys()), range=list(PALETTE_CLASSES.values())
)
ECHELLE_TYPES = alt.Scale(
    domain=list(PALETTE_TYPES.keys()), range=list(PALETTE_TYPES.values())
)
 

CSS = """
<style>
.carte {
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 0.6rem;
    border-left: 6px solid var(--c);
    background: linear-gradient(135deg,
        color-mix(in srgb, var(--c) 24%, transparent),
        color-mix(in srgb, var(--c) 6%, transparent));
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.carte:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px color-mix(in srgb, var(--c) 35%, transparent);
}
.carte-titre { font-size: 0.9rem; opacity: 0.85; }
.carte-valeur { font-size: 2.1rem; font-weight: 700; line-height: 1.25; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
 
 

@st.cache_data(show_spinner="Chargement du dataset (le premier lancement peut être long)...")
def load_data():
    path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
   
    return pd.read_csv(os.path.join(path, "UNSW_NB15_testing-set.csv"))
 
 
def couleur_texte():
    """Couleur du texte dessiné dans les graphiques, adaptée au thème clair ou sombre."""
    try:
        sombre = st.context.theme.type == "dark"
    except Exception:
        sombre = False
    return "#e8ecf4" if sombre else "#1c2333"
 
 
def carte(titre, valeur, couleur):
    """Carte de chiffre clé colorée."""
    st.markdown(
        f"""
        <div class="carte" style="--c: {couleur};">
            <div class="carte-titre">{titre}</div>
            <div class="carte-valeur">{valeur}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
 
def format_nombre(n):
    """Écrit 175341 sous la forme 175 341."""
    return f"{int(n):,}".replace(",", " ")
 
 
def courbe_densite(valeurs, grille):
    """Estimation de densité (KDE gaussien) calculée avec numpy."""
    valeurs = np.asarray(valeurs, dtype=float)
    n = len(valeurs)
    ecart = valeurs.std()
    if n < 2 or ecart == 0:
        return np.zeros_like(grille)
    h = 1.06 * ecart * n ** (-0.2)
    dens = np.array([np.exp(-0.5 * ((g - valeurs) / h) ** 2).sum() for g in grille])
    return dens / (n * h * np.sqrt(2 * np.pi))
 
 
@st.cache_data
def calculer_correlation(data, methode):
    num = data.select_dtypes(include="number").drop(columns=["id"])
    echantillon = num.sample(min(len(num), 30000), random_state=42)
    return echantillon.corr(method=methode)
 
 

def page_accueil():
    df_complet = load_data()
 
    st.title("🛡️ Détection d'intrusions réseau par Machine Learning")
 
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
    with c1:
        carte("Connexions étudiées", format_nombre(len(df_complet)), COULEUR_NEUTRE)
    with c2:
        carte("Colonnes", df_complet.shape[1], COULEUR_VIOLET)
    with c3:
        carte("Types de connexion", df_complet["attack_cat"].nunique(), COULEUR_TURQUOISE)
 
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
    st.dataframe(descriptions, hide_index=True)
 

def page_exploration():
    df_complet = load_data()
 

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
 

    st.title("📊 Exploration des données")
 
    nb_attaques = int((df["label"] == 1).sum())
    nb_normales = int((df["label"] == 0).sum())
 
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        carte("Connexions", format_nombre(len(df)), COULEUR_NEUTRE)
    with c2:
        carte("🟢 Normales", format_nombre(nb_normales), COULEUR_NORMAL)
    with c3:
        carte("🔴 Attaques", format_nombre(nb_attaques), COULEUR_ATTAQUE)
    with c4:
        carte("Part d'attaques", f"{df['label'].mean() * 100:.1f} %", COULEUR_ORANGE)
 

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

    with onglet0:
        st.subheader("Vue d'ensemble")
        st.caption(
            "Passe la souris sur les graphiques pour voir les valeurs exactes. "
            "Clique sur la légende de l'anneau pour isoler une classe."
        )
 
        gauche, droite = st.columns([1, 2])
 
        with gauche:
            with st.container(border=True):
                st.write("Répartition normal / attaque")
                total = nb_normales + nb_attaques
                df_anneau = pd.DataFrame(
                    {
                        "Classe": ["Normal", "Attaque"],
                        "Connexions": [nb_normales, nb_attaques],
                    }
                )
                df_anneau["Part"] = df_anneau["Connexions"] / total
 
                selection = alt.selection_point(fields=["Classe"], bind="legend")
                anneau = (
                    alt.Chart(df_anneau)
                    .mark_arc(innerRadius=70, outerRadius=130)
                    .encode(
                        theta=alt.Theta("Connexions:Q"),
                        color=alt.Color(
                            "Classe:N",
                            scale=ECHELLE_CLASSES,
                            legend=alt.Legend(orient="bottom", title=None),
                        ),
                        opacity=alt.condition(selection, alt.value(1), alt.value(0.25)),
                        tooltip=[
                            alt.Tooltip("Classe:N", title="Classe"),
                            alt.Tooltip("Connexions:Q", title="Connexions", format=","),
                            alt.Tooltip("Part:Q", title="Part", format=".1%"),
                        ],
                    )
                    .add_params(selection)
                    .properties(height=320)
                )
                centre = (
                    alt.Chart(pd.DataFrame({"texte": [f"{nb_attaques / total * 100:.1f} %"]}))
                    .mark_text(size=30, fontWeight="bold", color=couleur_texte())
                    .encode(text="texte:N")
                )
                st.altair_chart(anneau + centre)
 
        with droite:
            with st.container(border=True):
                st.write("Nombre de connexions par type")
                effectifs = df["attack_cat"].value_counts()
                df_types = pd.DataFrame(
                    {
                        "Type": effectifs.index.astype(str),
                        "Connexions": effectifs.values,
                    }
                )
                df_types["Part"] = df_types["Connexions"] / len(df)
 
                barres = (
                    alt.Chart(df_types)
                    .mark_bar()
                    .encode(
                        y=alt.Y("Type:N", sort="-x", title=None),
                        x=alt.X("Connexions:Q", title="Nombre de connexions"),
                        color=alt.Color("Type:N", scale=ECHELLE_TYPES, legend=None),
                        tooltip=[
                            alt.Tooltip("Type:N", title="Type"),
                            alt.Tooltip("Connexions:Q", title="Connexions", format=","),
                            alt.Tooltip("Part:Q", title="Part", format=".1%"),
                        ],
                    )
                )
                valeurs = barres.mark_text(
                    align="left", dx=4, color=couleur_texte()
                ).encode(text=alt.Text("Connexions:Q", format=","))
                st.altair_chart((barres + valeurs).properties(height=330))
 

    with onglet1:
        st.subheader("Statistiques descriptives")
        st.write("Aperçu des premières lignes :")
        st.dataframe(df.head(10))
 
        stats = df.describe().T
        st.write("Résumé de chaque variable numérique :")
        st.dataframe(stats)
        st.download_button(
            "Télécharger les statistiques (CSV)",
            stats.to_csv().encode("utf-8"),
            file_name="statistiques_descriptives.csv",
            mime="text/csv",
        )
 

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
            with st.container(border=True):
                st.write("Taux de présence par colonne")
                st.dataframe(tableau_manquantes)
        with droite:
            with st.container(border=True):
                
                tirets = (df.select_dtypes(exclude="number") == "-").sum()
                st.write("Valeurs inconnues notées « - » (colonnes texte)")
                st.bar_chart(tirets, color=COULEUR_ORANGE)
 

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
 

        xmin = float(tableau_dist["valeur"].min())
        xmax = float(tableau_dist["valeur"].max())
        if xmax == xmin:
            xmax = xmin + 1.0
        bornes = np.linspace(xmin, xmax, 51)
        grille = np.linspace(xmin, xmax, 200)
 
        lignes_hist, lignes_kde = [], []
        for classe in ["Normal", "Attaque"]:
            vals = tableau_dist.loc[tableau_dist["Classe"] == classe, "valeur"].to_numpy()
            if len(vals) == 0:
                continue
            dens, _ = np.histogram(vals, bins=bornes, density=True)
            lignes_hist.append(
                pd.DataFrame(
                    {"debut": bornes[:-1], "fin": bornes[1:], "densite": dens, "Classe": classe}
                )
            )
            sous = vals if len(vals) <= 5000 else np.random.default_rng(42).choice(
                vals, 5000, replace=False
            )
            lignes_kde.append(
                pd.DataFrame(
                    {
                        "valeur": grille,
                        "densite": courbe_densite(sous, grille),
                        "Classe": classe,
                    }
                )
            )
        df_hist = pd.concat(lignes_hist, ignore_index=True)
        df_kde = pd.concat(lignes_kde, ignore_index=True)
 
        titre_x = f"log(1 + {col})" if echelle_log else col
        histogramme = (
            alt.Chart(df_hist)
            .mark_bar(opacity=0.35)
            .encode(
                x=alt.X("debut:Q", title=titre_x),
                x2="fin:Q",
                y=alt.Y("densite:Q", title="Densité"),
                color=alt.Color("Classe:N", scale=ECHELLE_CLASSES),
                tooltip=[
                    alt.Tooltip("Classe:N", title="Classe"),
                    alt.Tooltip("debut:Q", title="De", format=".2f"),
                    alt.Tooltip("fin:Q", title="À", format=".2f"),
                    alt.Tooltip("densite:Q", title="Densité", format=".4f"),
                ],
            )
        )
        courbes = (
            alt.Chart(df_kde)
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X("valeur:Q"),
                y=alt.Y("densite:Q"),
                color=alt.Color("Classe:N", scale=ECHELLE_CLASSES),
            )
        )
 
        with st.container(border=True):
            st.caption(
                "Molette pour zoomer, glisser pour déplacer, double-clic pour revenir à la vue initiale."
            )
            st.altair_chart((histogramme + courbes).properties(height=380).interactive())
 

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
                "Catégorie": frequences.index.astype(str),
                "Effectif": frequences.values,
                "Pourcentage (%)": (frequences.values / len(df) * 100).round(2),
            }
        ).head(nb_max)
 
        gauche, droite = st.columns([2, 1])
        with gauche:
            with st.container(border=True):
                graphique_freq = (
                    alt.Chart(tableau_freq)
                    .mark_bar(color=COULEUR_NEUTRE)
                    .encode(
                        y=alt.Y("Catégorie:N", sort="-x", title=None),
                        x=alt.X("Effectif:Q", title="Effectif"),
                        tooltip=[
                            alt.Tooltip("Catégorie:N", title="Catégorie"),
                            alt.Tooltip("Effectif:Q", title="Effectif", format=","),
                            alt.Tooltip("Pourcentage (%):Q", title="Pourcentage (%)"),
                        ],
                    )
                    .properties(height=max(200, 32 * len(tableau_freq)))
                )
                st.altair_chart(graphique_freq)
        with droite:
            st.dataframe(tableau_freq, hide_index=True)
 

    with onglet5:
        st.subheader("Matrice de corrélation")
 
        methode = st.radio("Méthode de corrélation :", ["spearman", "pearson"], horizontal=True)
        corr = calculer_correlation(df, methode)
 
        corr_longue = corr.stack().rename_axis(["x", "y"]).reset_index(name="correlation")
 
        gauche, droite = st.columns([3, 1])
        with gauche:
            with st.container(border=True):
                st.caption("Passe la souris sur une case pour voir les deux variables et leur corrélation.")
                heatmap = (
                    alt.Chart(corr_longue)
                    .mark_rect()
                    .encode(
                        x=alt.X(
                            "x:N",
                            sort=list(corr.columns),
                            title=None,
                            axis=alt.Axis(labelAngle=-90),
                        ),
                        y=alt.Y("y:N", sort=list(corr.index), title=None),
                        color=alt.Color(
                            "correlation:Q",
                            scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
                            legend=alt.Legend(title="Corrélation"),
                        ),
                        tooltip=[
                            alt.Tooltip("x:N", title="Variable 1"),
                            alt.Tooltip("y:N", title="Variable 2"),
                            alt.Tooltip("correlation:Q", title="Corrélation", format=".2f"),
                        ],
                    )
                    .properties(height=700)
                )
                st.altair_chart(heatmap)
        with droite:
            with st.container(border=True):
                st.write("Les 10 variables les plus corrélées avec la cible (label) :")
                top = (
                    corr["label"]
                    .drop("label")
                    .dropna()
                    .sort_values(key=abs, ascending=False)
                    .head(10)
                )
                st.dataframe(top.rename("Corrélation"))
 
 

@st.cache_data(show_spinner="Analyse de la qualité des données...")
def analyser_qualite(data):
    return rapport_qualite(data)


@st.cache_data
def lignes_dupliquees(data):
    return exemples_doublons(data)


def page_controle_qualite():
    st.title("🧪 Contrôle qualité des données")
    st.write(
        "Avant de modéliser, on vérifie que les données sont saines : types des colonnes, "
        "valeurs manquantes ou inconnues, doublons, valeurs aberrantes et cohérence entre les colonnes. "
        "Tu peux contrôler le jeu UNSW-NB15 ou importer ton propre fichier CSV."
    )

    source = st.radio(
        "Données à contrôler :",
        ["Jeu UNSW-NB15 (entraînement)", "Importer un fichier CSV"],
        horizontal=True,
    )
    if source.startswith("Jeu"):
        df = load_data()
    else:
        fichier = st.file_uploader("Choisis un fichier CSV", type="csv")
        if fichier is None:
            st.info("Importe un fichier CSV pour lancer le contrôle.")
            st.stop()
        try:
            df = pd.read_csv(fichier)
        except Exception as erreur:
            st.error(f"Impossible de lire ce fichier : {erreur}")
            st.stop()
        if df.empty:
            st.warning("Le fichier ne contient aucune ligne.")
            st.stop()

    rapport = analyser_qualite(df)
    resume = rapport["resume"]
    types = rapport["types"]
    coherence = rapport["coherence"]

    nb_types_inattendus = int((types["Conforme"] != "✅").sum())
    nb_controles_defaut = int((coherence["Verdict"] != "✅ OK").sum())
    nb_alertes = (
        nb_types_inattendus
        + nb_controles_defaut
        + int(resume["manquantes"] > 0)
        + int(resume["doublons"] > 0)
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        carte("Lignes × colonnes", f"{format_nombre(resume['lignes'])} × {resume['colonnes']}", COULEUR_NEUTRE)
    with c2:
        carte("Valeurs manquantes", format_nombre(resume["manquantes"]), COULEUR_TURQUOISE)
    with c3:
        carte("Lignes dupliquées", format_nombre(resume["doublons"]), COULEUR_ORANGE)
    with c4:
        carte("Alertes", str(nb_alertes), COULEUR_ATTAQUE if nb_alertes else COULEUR_NORMAL)

    if nb_alertes == 0:
        st.success("Aucun problème détecté sur les types, les valeurs manquantes, les doublons et la cohérence.")
    else:
        st.warning(
            f"{nb_alertes} point(s) à examiner : {nb_types_inattendus} type(s) de colonne inattendu(s), "
            f"{nb_controles_defaut} contrôle(s) de cohérence en défaut, "
            f"{format_nombre(resume['manquantes'])} case(s) vide(s), "
            f"{format_nombre(resume['doublons'])} ligne(s) dupliquée(s)."
        )

    onglet1, onglet2, onglet3, onglet4, onglet5 = st.tabs(
        [
            "🔤 Types",
            "❓ Manquantes et inconnues",
            "♊ Doublons",
            "📏 Valeurs aberrantes",
            "✅ Cohérence",
        ]
    )

    with onglet1:
        st.subheader("Types des colonnes")
        st.caption(
            "Chaque colonne est comparée à son rôle attendu : variable numérique, variable "
            "catégorielle (proto, service, state), cible (label, attack_cat) ou identifiant."
        )
        st.dataframe(types, hide_index=True)
        st.download_button(
            "Télécharger le tableau des types (CSV)",
            types.to_csv(index=False).encode("utf-8"),
            file_name="controle_types.csv",
            mime="text/csv",
        )

    with onglet2:
        st.subheader("Valeurs manquantes et valeurs inconnues")
        gauche, droite = st.columns(2)
        with gauche:
            with st.container(border=True):
                st.write("Cases vides par colonne")
                st.dataframe(
                    types[["Colonne", "Manquantes", "Taux manquant (%)"]], hide_index=True
                )
        with droite:
            with st.container(border=True):
                st.write("Valeurs inconnues codées « - » (colonnes texte)")
                inconnues = rapport["inconnues"]
                st.dataframe(inconnues, hide_index=True)
                if not inconnues.empty:
                    st.bar_chart(inconnues.set_index("Colonne")["Part (%)"], color=COULEUR_ORANGE)
        st.caption(
            "Une valeur « - » n'est pas une case vide pour pandas, mais c'est une information "
            "manquante en pratique : elle doit être traitée explicitement (par exemple comme une catégorie à part)."
        )

    with onglet3:
        st.subheader("Doublons")
        st.write(
            f"**{format_nombre(resume['doublons'])}** ligne(s) identique(s) à une autre "
            f"({resume['doublons'] / resume['lignes'] * 100:.2f} % du jeu). "
            "L'identifiant (`id`) est ignoré dans la comparaison, car il est unique par construction."
        )
        st.caption(
            "Des lignes identiques réparties entre l'entraînement et la validation peuvent gonfler "
            "artificiellement les scores d'un modèle."
        )
        if resume["doublons"] > 0:
            st.write("Exemples (toutes les copies des premières lignes concernées) :")
            st.dataframe(lignes_dupliquees(df), hide_index=True)

    with onglet4:
        st.subheader("Valeurs aberrantes (méthode de l'écart interquartile)")
        st.caption(
            "Une valeur est signalée si elle sort de l'intervalle [Q1 - 1,5 × IQR ; Q3 + 1,5 × IQR]. "
            "Pour UNSW-NB15, ces valeurs extrêmes sont souvent des attaques : elles sont conservées."
        )
        aberrantes = rapport["aberrantes"]
        if aberrantes.empty:
            st.info("Aucune variable numérique à analyser.")
        else:
            st.dataframe(aberrantes, hide_index=True)
            st.write("Part de valeurs hors bornes (15 variables les plus touchées)")
            st.bar_chart(aberrantes.head(15).set_index("Variable")["Part (%)"], color=COULEUR_VIOLET)

    with onglet5:
        st.subheader("Contrôles de cohérence")
        st.dataframe(coherence, hide_index=True)
        st.download_button(
            "Télécharger les contrôles (CSV)",
            coherence.to_csv(index=False).encode("utf-8"),
            file_name="controle_coherence.csv",
            mime="text/csv",
        )


def page_apropos():
    st.title("ℹ️ À propos du projet")
 
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
 
 

page_accueil_obj = st.Page(page_accueil, title="Accueil", icon="🏠", default=True)
page_exploration_obj = st.Page(
    page_exploration, title="Exploration des données", icon="📊", url_path="exploration"
)
page_qualite_obj = st.Page(
    page_controle_qualite, title="Contrôle qualité", icon="🧪", url_path="qualite"
)
page_apropos_obj = st.Page(page_apropos, title="À propos", icon="ℹ️", url_path="apropos")
 
navigation = st.navigation(
    [page_accueil_obj, page_exploration_obj, page_qualite_obj, page_apropos_obj],
    position="hidden",
)
 
with st.container(border=True):
    colonnes = st.columns([2.6, 1.2, 2.3, 1.9, 1.2])
    colonnes[0].markdown("#### 🛡️ IDS · UNSW-NB15")
    with colonnes[1]:
        st.page_link(page_accueil_obj, label="Accueil", icon="🏠")
    with colonnes[2]:
        st.page_link(page_exploration_obj, label="Exploration des données", icon="📊")
    with colonnes[3]:
        st.page_link(page_qualite_obj, label="Contrôle qualité", icon="🧪")
    with colonnes[4]:
        st.page_link(page_apropos_obj, label="À propos", icon="ℹ️")
 
navigation.run()
 

