import numpy as np
import pandas as pd
import os

# Chargement du fichier (gzip)
def charger_donnees(path_csv):
    print(f"Chargement : {path_csv}")
    df = pd.read_csv(path_csv, sep=';', compression='infer', low_memory=False)
    print(f"Dimensions brutes : {df.shape}")
    return df

# Conversion des colonnes numériques avec virgules en float
def conversion_virgules(df):
    for col in df.columns[2:]:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Suppression des lignes avec trop de NaN
def nettoyer_lignes(df, seuil=0.5):
    seuil_abs = int(seuil * df.shape[1])
    df = df.dropna(thresh=seuil_abs)
    df = df.reset_index(drop=True)
    return df

# Suppression des colonnes sans variation
def supprimer_colonnes_constantes(df, seuil_variation=0.1):
    colonnes_a_supprimer = []
    for col in df.columns[2:]:
        try:
            min_val, max_val = df[col].min(), df[col].max()
            if pd.isna(min_val) or pd.isna(max_val):
                continue
            if max_val - min_val < seuil_variation:
                colonnes_a_supprimer.append(col)
        except Exception:
            continue
    df = df.drop(columns=colonnes_a_supprimer)
    return df

def supprimer_colonnes_correlees(df, seuil=0.98):
    colonnes = df.columns[2:]  # On ignore les 2 premières colonnes (ex : identifiants)
    df_numeric = df[colonnes]

    # Calcul de la matrice de corrélation absolue
    corr_matrix = df_numeric.corr().abs()

    # On ignore la diagonale
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Colonnes à supprimer
    colonnes_a_supprimer = [
        column for column in upper.columns if any(upper[column] > seuil)
    ]

    print(f"Colonnes supprimées car corrélées à ≥ {seuil*100:.0f}% : {colonnes_a_supprimer}")

    # Suppression dans le dataframe complet
    df = df.drop(columns=colonnes_a_supprimer)
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
