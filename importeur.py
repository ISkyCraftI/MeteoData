import os
import glob
import pandas as pd

# === Chargement multi-départements ===
files = {
    "29": "D29/H_29_2020-2023.csv.gz",
    "21": "D21/H_21_2020-2023.csv.gz",
    "06": "D06/H_06_2020-2023.csv.gz",
}


# === Palette personnalisée des départements ===
COULEURS_DEPS = {
    "21": "#8f2035",   # Bordeaux (effet vin/cassis)
    "29": "#bbd500",    # Vert vif
    "06": "#87ceeb"  # Bleu ciel

    # Ajoute ici d'autres départements si tu veux des couleurs spécifiques

}

def detecter_fichiers_par_departement(racine=".", extension="*.csv.gz"):
    """
    Détecte tous les fichiers .csv.gz dans les sous-dossiers (ex: D29, D21…).
    Retourne un dict : { "29": [chemin1, chemin2], "21": [...], ... }
    """
    fichiers = glob.glob(os.path.join(racine, "**", extension), recursive=True)
    fichiers_par_dep = {}

    for fichier in fichiers:
        dossier = os.path.basename(os.path.dirname(fichier))
        dep = dossier.replace("D", "") if dossier.startswith("D") else dossier

        if dep not in fichiers_par_dep:
            fichiers_par_dep[dep] = []
        fichiers_par_dep[dep].append(fichier)

    return fichiers_par_dep

# def charger_donnees_departements(fichiers_par_dep):
#     """
#     Charge tous les fichiers détectés et ajoute une colonne 'dep'.
#     """
#     dfs = []
#     for dep, fichiers in fichiers_par_dep.items():
#         for file in fichiers:
#             df = pd.read_csv(file, compression='gzip', sep=';', low_memory=False)
#             df["dep"] = dep
#             df.columns = df.columns.str.strip()
#             dfs.append(df)
#     return pd.concat(dfs, ignore_index=True)


def charger_donnees_departements(files_dict):
    """
    Charge tous les fichiers d’un dictionnaire {dep: fichier_path}
    et concatène les DataFrame avec ajout d’une colonne 'dep'.

def charger_donnees_departements(fichiers_par_dep):
    """
    #Charge tous les fichiers détectés et ajoute une colonne 'dep'.

    
    dfs = []
    for dep, filepath in files_dict.items():
        try:
            print(f"[INFO] Chargement {dep} depuis {filepath}...")
            df = pd.read_csv(filepath, compression='gzip', sep=';', low_memory=False)
            df["dep"] = dep
            df.columns = df.columns.str.strip()  # Nettoyage des noms de colonnes
            dfs.append(df)
        except Exception as e:
            print(f"[ERREUR] Impossible de charger le fichier pour le département {dep} : {e}")
    
    if not dfs:
        raise ValueError("Aucune donnée chargée. Vérifiez vos fichiers.")
    
    return pd.concat(dfs, ignore_index=True)


def filtrer_colonnes_utiles(df):
    colonnes = ["date", "T", "U", "RR1", "FF", "DD", "P", "dep"]
    colonnes_presentes = [col for col in colonnes if col in df.columns]
    return df[colonnes_presentes].copy()
