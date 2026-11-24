
import numpy as np
import pandas as pd

COLONNES_CATEGORIELLES = ["proto", "service", "state"]
COLONNES_CIBLES = ["label", "attack_cat"]
COLONNE_ID = "id"


def colonnes_numeriques(df):
    """Variables numériques utiles à l'analyse (sans l'identifiant ni les cibles)."""
    exclues = {COLONNE_ID, *COLONNES_CIBLES}
    return [c for c in df.select_dtypes(include="number").columns if c not in exclues]


def _sans_id(df):
    return [c for c in df.columns if c != COLONNE_ID]


def resume_global(df):
    """Chiffres clés de qualité du jeu de données."""
    colonnes = _sans_id(df)
    return {
        "lignes": len(df),
        "colonnes": df.shape[1],
        "manquantes": int(df.isnull().sum().sum()),
        "doublons": int(df.duplicated(subset=colonnes).sum()),
        "constantes": int((df.nunique(dropna=False) <= 1).sum()),
        "memoire_mo": float(round(df.memory_usage(deep=True).sum() / 1e6, 1)),
    }


def type_attendu(colonne):
    """Rôle attendu d'une colonne dans le jeu UNSW-NB15."""
    if colonne == COLONNE_ID:
        return "identifiant"
    if colonne in COLONNES_CIBLES:
        return "cible"
    if colonne in COLONNES_CATEGORIELLES:
        return "catégoriel"
    return "numérique"


def tableau_types(df):
    """Une ligne par colonne : type lu, type attendu, valeurs distinctes, manquantes."""
    lignes = []
    for c in df.columns:
        attendu = type_attendu(c)
        est_num = pd.api.types.is_numeric_dtype(df[c])
        if attendu == "numérique":
            conforme = est_num
        elif attendu == "catégoriel":
            conforme = not est_num
        else:
            conforme = True
        lignes.append(
            {
                "Colonne": c,
                "Type lu": str(df[c].dtype),
                "Rôle attendu": attendu,
                "Conforme": "✅" if conforme else "⚠️ type inattendu",
                "Valeurs distinctes": int(df[c].nunique(dropna=True)),
                "Manquantes": int(df[c].isnull().sum()),
                "Taux manquant (%)": round(df[c].isnull().mean() * 100, 2),
                "Exemple": str(df[c].dropna().iloc[0]) if df[c].notna().any() else "",
            }
        )
    return pd.DataFrame(lignes)


def valeurs_inconnues(df):
    """Valeurs « inconnues » codées par un symbole dans les colonnes texte (« - » ou chaîne vide).

    Ce ne sont pas des cases vides pour pandas, mais ce sont des informations manquantes en pratique.
    """
    colonnes_texte = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    lignes = []
    for c in colonnes_texte:
        texte = df[c].astype(str).str.strip()
        n = int(texte.isin(["-", ""]).sum())
        lignes.append({"Colonne": c, "Valeurs « - » ou vides": n, "Part (%)": round(n / len(df) * 100, 2)})
    return pd.DataFrame(lignes, columns=["Colonne", "Valeurs « - » ou vides", "Part (%)"])


def exemples_doublons(df, nombre=20):
    """Premières lignes dupliquées (l'identifiant est ignoré pour la comparaison)."""
    masque = df.duplicated(subset=_sans_id(df), keep=False)
    return df[masque].sort_values(_sans_id(df)[:3]).head(nombre)


def valeurs_aberrantes(df):
    """Valeurs hors de [Q1 - 1,5 IQR ; Q3 + 1,5 IQR] pour chaque variable numérique."""
    lignes = []
    for c in colonnes_numeriques(df):
        s = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        bas, haut = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < bas) | (s > haut)).sum())
        lignes.append(
            {
                "Variable": c,
                "Q1": q1,
                "Q3": q3,
                "Borne basse": bas,
                "Borne haute": haut,
                "Valeurs hors bornes": n,
                "Part (%)": round(n / len(s) * 100, 2),
            }
        )
    resultat = pd.DataFrame(lignes)
    if not resultat.empty:
        resultat = resultat.sort_values("Part (%)", ascending=False).reset_index(drop=True)
    return resultat


def controles_coherence(df):
    """Contrôles de cohérence : chaque ligne du résultat est un test avec son verdict."""
    controles = []

    def ajouter(nom, nb_problemes, detail=""):
        controles.append(
            {
                "Contrôle": nom,
                "Lignes en défaut": int(nb_problemes),
                "Verdict": "✅ OK" if nb_problemes == 0 else "⚠️ À vérifier",
                "Détail": detail,
            }
        )

    if COLONNE_ID in df.columns:
        ajouter("Identifiants uniques", df[COLONNE_ID].duplicated().sum(), "chaque connexion a un id distinct")

    if {"label", "attack_cat"}.issubset(df.columns):
        incoherent = (df["label"] == 0) != (df["attack_cat"].astype(str) == "Normal")
        ajouter("label cohérent avec attack_cat", incoherent.sum(), "label = 0 exactement quand attack_cat = Normal")

    num = df[colonnes_numeriques(df)]
    ajouter("Aucune valeur négative", (num < 0).any(axis=1).sum(), "durées, octets et paquets sont positifs")
    ajouter(
        "Aucune valeur infinie",
        np.isinf(num.to_numpy(dtype=float)).any(axis=1).sum(),
        "pas de division par zéro dans les variables dérivées",
    )
    constantes = [c for c in num.columns if num[c].nunique(dropna=False) <= 1]
    ajouter("Aucune colonne constante", len(constantes), ", ".join(constantes) if constantes else "toutes les variables varient")
    return pd.DataFrame(controles)


def rapport_qualite(df):
    
    return {
        "resume": resume_global(df),
        "types": tableau_types(df),
        "inconnues": valeurs_inconnues(df),
        "aberrantes": valeurs_aberrantes(df),
        "coherence": controles_coherence(df),
    }
