# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import os

from VisualtionsRedDim import *
from nettoyage import *
from reductionDim import *
from courbes import *
from VisualisationCluster import *


# Chargement du fichier (gzip)
def charger_donnees(path_csv):
    print(f"Chargement : {path_csv}")
    df = pd.read_csv(path_csv, sep=';', compression='infer', low_memory=False)
    df.columns = df.columns.str.strip()  # <-- Ajoute cette ligne
    print(f"Dimensions brutes : {df.shape}")
    # return df[:100]
    return df

def jeuDeTest():
    fichier = "D29/H_29_2024-2025.csv.gz"
    df = charger_donnees(fichier)

    df_test = df.copy()
    df_test.to_csv("jeu_de_test_100.csv", index=False, sep=';')
    print("Jeu de test sauvegardé : jeu_de_test_100.csv")


# Programme principal
if __name__ == "__main__":
    # fichier = "jeu_de_test_100.csv"  
    fichier = "D29/H_29_2024-2025.csv.gz"
    df = charger_donnees(fichier)
    print(df.columns.tolist())
    
    df = conversion_virgules(df)
    df.replace([' ', '', 'nan', 'None', 'NONE'], np.nan, inplace=True)
    
    # 1. Supprime colonnes avec < 5 valeurs
    df = supprimer_colonnes_peu_remplies(df, min_non_nan=5, verbose=True)
    
    # 2. Supprime les lignes trop vides
    seuil_lignes = 0.5
    print(df.isna().sum(axis=1).sort_values(ascending=False).head(10))
    print(f"Taille seuil NaN (lignes) : {int(seuil_lignes * df.shape[1])}")
    
    df = nettoyer_lignes(df, seuil=seuil_lignes)
    print(f"Dimensions après nettoyage lignes : {df.shape}")
    
    df = dateRewrite(df)   
    
    # 3. Supprime colonnes constantes
    df = supprimer_colonnes_constantes(df, seuil_variation=0.1, verbose=True)
    
    df = supprimer_colonnes_peu_remplies(df, min_non_nan=5, verbose=True)
    
    df = supprimer_colonnes_correlees(df, seuil=0.95)
    
    correlation(df,seuil_corr=0.5)

    # Affichage des courbes
    boiteAMoustache(df, verbose=True)
    
    NuagePointsTemperature(df)
    
    courbe_moyenne_par_mois(df, colonne="T", label="Température")
    courbe_moyenne_par_mois(df, colonne="RR1", label="Précipitations")
