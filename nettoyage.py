import numpy as np
import pandas as pd


# ✅
def conversion_virgules(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    df = df.copy()
    cols_concernees = []

    for col in df.columns[2:]:
        try:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                # Vérifie rapidement s'il y a des virgules dans les 100 premiqères lignes
                if df[col].astype(str).head(100).str.contains(',', regex=False).any():
                    cols_concernees.append(col)
        except Exception as e:
            if verbose:
                print(f"[WARN] Impossible d'analyser {col} : {e}")
    
    if verbose:
        print(f"[INFO] Colonnes avec virgules détectées ({len(cols_concernees)}) : {cols_concernees}")

    for col in cols_concernees:
        try:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            if verbose:
                print(f"[WARN] Conversion échouée pour {col} : {e}")

    return df

# ✅
def nettoyer_lignes(df, seuil=0.1):
    seuil_abs = max(1, int(seuil * df.shape[1]))
    df = df.dropna(thresh=seuil_abs)
    df = df.reset_index(drop=True)
    return df

# ✅
def supprimer_colonnes_peu_remplies(df, min_non_nan=5, verbose=False):
    valeurs_non_nulles = df.count()
    colonnes_a_supprimer = valeurs_non_nulles[valeurs_non_nulles < min_non_nan].index.tolist()

    if verbose:
        print(f"[INFO] Colonnes avec moins de {min_non_nan} valeurs non nulles : {colonnes_a_supprimer}")

    return df.drop(columns=colonnes_a_supprimer).reset_index(drop=True)

# ✅
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

# ✅
def dateRewrite(df):
    # Vérifie que la colonne existe
    if 'AAAAMMJJHH' not in df.columns:
        raise KeyError("La colonne 'AAAAMMJJHH' est absente du DataFrame.")

    # Conversion en datetime (format AAAAMMJJHH)
    df['AAAAMMJJHH'] = pd.to_datetime(df['AAAAMMJJHH'].astype(str), format='%Y%m%d%H', errors='coerce')

    # Transformation en chaîne ISO 8601 (ex. 2024-01-01T13:00:00)
    df['AAAAMMJJHH'] = df['AAAAMMJJHH'].dt.strftime('%Y-%m-%dT%H:00:00')
    return df

# ❌
def nettoyer_donnees(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    """
    Nettoie un DataFrame météo pour une analyse horaire (clustering, ACP, ML).

    Étapes :
    - Nettoyage classique (lignes vides, colonnes peu remplies, constantes, etc.)
    - Conversion des formats (virgules, date ISO)
    - Moyenne unique par heure pour chaque 'dep'
    """
    df = df.copy()

    if verbose: print("[INFO] Nettoyage des lignes trop incomplètes...")
    df = nettoyer_lignes(df, seuil=0.1)

    if verbose: print("[INFO] Suppression des colonnes peu remplies...")
    df = supprimer_colonnes_peu_remplies(df, min_non_nan=5, verbose=verbose)

    if verbose: print("[INFO] Suppression des colonnes constantes...")
    df = supprimer_colonnes_constantes(df, seuil_variation=0.1, verbose=verbose)

    if verbose: print("[INFO] Conversion des virgules en points...")
    df = conversion_virgules(df)

    if verbose: print("[INFO] Réécriture des dates au format ISO...")
    df = dateRewrite(df)

    # Conversion en datetime
    df['date'] = pd.to_datetime(df['AAAAMMJJHH'], format='%Y-%m-%dT%H:00:00', errors='coerce')

    # On garde uniquement les lignes avec une date valide
    df = df.dropna(subset=['date'])

    # Sélectionne uniquement les colonnes numériques à moyenner
    colonnes_a_moyenner = df.select_dtypes(include=[np.number]).columns.tolist()
    colonnes_a_moyenner = [col for col in colonnes_a_moyenner if col not in ["NUM_POSTE"]]  # exclure ID si besoin

    # Agrégation par dep + date (1 valeur par heure max)
    if verbose:
        print(f"[INFO] Colonnes agrégées par heure : {colonnes_a_moyenner}")

    df = df.groupby(['dep', 'date'])[colonnes_a_moyenner].mean().reset_index()

    df = df.sort_values(by=['dep', 'date']).reset_index(drop=True)

    return df

