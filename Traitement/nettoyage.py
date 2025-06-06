import numpy as np
import pandas as pd

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