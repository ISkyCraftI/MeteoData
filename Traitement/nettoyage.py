import numpy as np
import pandas as pd

def conversion_virgules(df):
    for col in df.columns[2:]:
        try:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"[WARN] Problème avec la colonne {col} : {e}")
    return df

def nettoyer_lignes(df, seuil=0.5):
    seuil_abs = int(seuil * df.shape[1])
    df = df.dropna(thresh=seuil_abs)
    df = df.reset_index(drop=True)
    return df

def supprimer_colonnes_peu_remplies(df, min_non_nan=5, verbose=False):
    valeurs_non_nulles = df.count()
    colonnes_a_supprimer = valeurs_non_nulles[valeurs_non_nulles < min_non_nan].index.tolist()

    if verbose:
        print(f"[INFO] Colonnes avec moins de {min_non_nan} valeurs non nulles : {colonnes_a_supprimer}")

    return df.drop(columns=colonnes_a_supprimer).reset_index(drop=True)


def supprimer_colonnes_constantes(df, seuil_variation=0.1, verbose=False):
    colonnes_a_supprimer = []
    for col in df.columns[2:]:
        try:
            if df[col].isna().all():
                colonnes_a_supprimer.append(col)
                continue
            min_val, max_val = df[col].min(), df[col].max()
            if pd.isna(min_val) or pd.isna(max_val):
                continue
            if max_val - min_val < seuil_variation:
                colonnes_a_supprimer.append(col)
        except Exception as e:
            if verbose:
                print(f"[WARN] Erreur sur {col} : {e}")
            continue

    if verbose:
        print(f"[INFO] Colonnes supprimées ({len(colonnes_a_supprimer)}) : {colonnes_a_supprimer}")

    df = df.drop(columns=colonnes_a_supprimer)
    df.reset_index(drop=True, inplace=True)
    return df

def dateRewrite(df):
    # Vérifie que la colonne existe
    if 'AAAAMMJJHH' not in df.columns:
        raise KeyError("La colonne 'AAAAMMJJHH' est absente du DataFrame.")

    # Conversion en datetime (format AAAAMMJJHH)
    df['AAAAMMJJHH'] = pd.to_datetime(df['AAAAMMJJHH'].astype(str), format='%Y%m%d%H', errors='coerce')

    # Transformation en chaîne ISO 8601 (ex. 2024-01-01T13:00:00)
    df['AAAAMMJJHH'] = df['AAAAMMJJHH'].dt.strftime('%Y-%m-%dT%H:00:00')
    return df
