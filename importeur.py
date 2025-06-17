import pandas as pd
# === Chargement multi-départements ===
files = {
    "29": "D29/H_29_2020-2023.csv.gz",
    "21": "D21/H_21_previous-2020-2023.csv.gz",
}

# Palette personnalisée des départements
COULEURS_DEPS = {
    "21": "#8f2035",   # Bordeaux (effet vin/cassis)
    "29": "#bbd500"    # Vert vif
}

def charger_donnees_departements(files_dict):
    dfs = []
    for dep, file in files_dict.items():
        df = pd.read_csv(file, compression='gzip', sep=';', low_memory=False)
        df["dep"] = dep
        df.columns = df.columns.str.strip()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def filtrer_colonnes_utiles(df):
    colonnes = ["date", "T", "U", "RR1", "FF", "DD", "PSTAT", "P", "dep"]
    colonnes_presentes = [col for col in colonnes if col in df.columns]
    return df[colonnes_presentes].copy()