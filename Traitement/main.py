import numpy as np
import pandas as pd
import os

from Traitement.nettoyage import *
from Traitement.reductionDim import *


# Chargement du fichier (gzip)
def charger_donnees(path_csv):
    print(f"Chargement : {path_csv}")
    df = pd.read_csv(path_csv, sep=';', compression='infer', low_memory=False)
    print(f"Dimensions brutes : {df.shape}")
    return df


# Programme principal
if __name__ == "__main__":
    fichier = "D29/H_29_2024-2025.csv.gz"  # À adapter si besoin
    df = charger_donnees(fichier)
    df = conversion_virgules(df)
    df = nettoyer_lignes(df)
    df = supprimer_colonnes_constantes(df)

    print(f"Dimensions finales : {df.shape}")
    
    df = supprimer_colonnes_correlees(df, seuil=0.98)
    
    print(f"Dimensions finales corrélées : {df.shape}")
    df.to_csv("donnees_meteo_nettoyees.csv", index=False)
    print("Fichier nettoyé exporté : donnees_meteo_nettoyees.csv")
