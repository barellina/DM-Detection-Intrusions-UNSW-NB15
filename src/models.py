
import json
from pathlib import Path

import joblib
import pandas as pd

DOSSIER_MODELES = Path(__file__).resolve().parent.parent / "models"


def charger_modele():
    
    preprocesseur = joblib.load(DOSSIER_MODELES / "preprocessor.joblib")
    modele = joblib.load(DOSSIER_MODELES / "rf_optimise.joblib")
    with open(DOSSIER_MODELES / "valeurs_defaut.json", encoding="utf-8") as fichier:
        infos = json.load(fichier)
    return preprocesseur, modele, infos


def predire(saisie, preprocesseur, modele, infos):
    
    
    ligne = dict(infos["defaut"])
    ligne.update(saisie)
    donnees = pd.DataFrame([ligne], columns=infos["colonnes"])
    transformees = preprocesseur.transform(donnees)
    
    return float(modele.predict_proba(transformees)[0, 1])


COLONNES_CATEGORIELLES = ["proto", "service", "state"]


def predire_lot(donnees, preprocesseur, modele, infos):
    """Probabilité d'attaque pour chaque ligne d'un tableau (par exemple un CSV importé).

    - Les colonnes absentes du fichier reçoivent la valeur typique de `valeurs_defaut.json`.
    - Les valeurs numériques illisibles ou infinies sont remplacées par cette même valeur typique.
    - Les colonnes en trop (id, label, attack_cat...) sont ignorées par le modèle.

    Retourne (probabilites, rapport) ; le rapport liste ce qui a dû être complété ou corrigé.
    """
    colonnes = infos["colonnes"]
    defaut = infos["defaut"]
    entrees = pd.DataFrame(index=donnees.index)
    colonnes_absentes = []
    valeurs_corrigees = {}

    for c in colonnes:
        if c not in donnees.columns:
            colonnes_absentes.append(c)
            entrees[c] = defaut[c]
        elif c in COLONNES_CATEGORIELLES:
            texte = donnees[c].astype(str).str.strip()
            vide = donnees[c].isna() | (texte == "")
            valeurs_corrigees_c = int(vide.sum())
            entrees[c] = texte.where(~vide, str(defaut[c]))
            if valeurs_corrigees_c:
                valeurs_corrigees[c] = valeurs_corrigees_c
        else:
            nombres = pd.to_numeric(donnees[c], errors="coerce")
            nombres = nombres.replace([float("inf"), float("-inf")], float("nan"))
            nb = int(nombres.isna().sum())
            entrees[c] = nombres.fillna(defaut[c])
            if nb:
                valeurs_corrigees[c] = nb

    transformees = preprocesseur.transform(entrees[colonnes])
    probabilites = modele.predict_proba(transformees)[:, 1]
    rapport = {"colonnes_absentes": colonnes_absentes, "valeurs_corrigees": valeurs_corrigees}
    return probabilites, rapport
