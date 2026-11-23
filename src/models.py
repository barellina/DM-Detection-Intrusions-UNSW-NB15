

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
