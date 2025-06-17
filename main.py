import numpy as np
import pandas as pd
import os

from VisualtionsRedDim import *
from nettoyage import *
from reductionDim import *
from courbes import *
from VisualisationCluster import *

# Chargement multi-départements
files = {
    "29": "D29/H_29_2020-2023.csv.gz",
    "21": "D21/H_21_previous-2020-2023.csv.gz",
    # "2A": "H_2A_xxxx.csv.gz"  # optionnel
}

# Chargement du fichier (gzip)
def charger_donnees(path_csv):
    print(f"[INFO] Chargement : {path_csv}")
    df = pd.read_csv(path_csv, sep=';', compression='infer', low_memory=False)
    df.columns = df.columns.str.strip()  # Enlève les espaces autour des noms de colonnes
    print(f"[INFO] Dimensions brutes : {df.shape}")
    return df

# Génération d’un échantillon de test
def jeuDeTest():
    fichier = "D29/H_29_2024-2025.csv.gz"
    df = charger_donnees(fichier)
    df_test = df.copy()
    df_test.to_csv("jeu_de_test_100.csv", index=False, sep=';')
    print("[INFO] Jeu de test sauvegardé : jeu_de_test_100.csv")

# Programme principal
if __name__ == "__main__":
    fichier = "D29/H_29_2024-2025.csv.gz"  # ou "jeu_de_test_100.csv"
    df_brut = charger_donnees(fichier)

    # Nettoyage centralisé
    df = nettoyer_donnees(df_brut, verbose=True)
    print(df.columns.tolist())

    # Corrélation / Réduction de dimension (si nécessaire)
    correlation(df, seuil_corr=0.5)
    df = supprimer_colonnes_correlees(df, seuil=0.95)

    # Affichages / Visualisations
    NuagePointsTemperature(df)
    courbe_temperature_par_departement(df)
    courbes_variables(df)
    boiteAMoustache(df, verbose=True)
    hist_temperature(df)
    boxplot_temperature(df)
    courbe_moyenne_par_mois(df, colonne="T")
