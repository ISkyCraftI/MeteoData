import pandas as pd
import numpy as np

from nettoyage import nettoyer_donnees
from moyMedEcTyp import statistiques
from courbes import *
from VisualtionsRedDim import *
from reductionDim import *
from methodeCoude import *
from VisualisationCluster import *
from regression import regression_lineaire

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# === Chargement multi-départements ===
files = {
    "29": "D29/H_29_2020-2023.csv.gz",
    "21": "D21/H_21_previous-2020-2023.csv.gz",
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

# === Programme principal ===
if __name__ == "__main__":
    # Chargement brut
    data = charger_donnees_departements(files)
    print(f"[INFO] Dimensions brutes : {data.shape}")

    # Nettoyage : retourne uniquement les données horaires
    data = nettoyer_donnees(data, verbose=True)

    # Conversion unités
    data["P"] = data["PSTAT"]

    # === Visualisations météo (par heure) ===
    courbe_temperature_par_departement(data)
    boiteAMoustache(data)
    courbes_variables(data)
    boxplot_temperature(data)
    hist_temperature(data)

    # Statistiques descriptives (sur les données horaires)
    stats = data.groupby("dep").apply(lambda x: statistiques(x[["T", "U", "P", "FF"]]))
    print("\n[INFO] Statistiques descriptives :\n", stats)

    # Corrélation par département
    for dep in data["dep"].unique():
        print(f"\n[INFO] Corrélation - Département {dep}")
        heatmap_correlation(data[data["dep"] == dep], dep=dep)

    # === ACP ===
    features = ["T", "U", "P", "FF"]
    X = StandardScaler().fit_transform(data[features])
    data_pca, explained_var = appliquer_pca(data, features)
    data_pca["dep"] = data["dep"].values

    print(f"\n[INFO] Variance PC1 + PC2 : {explained_var[:2].sum():.2%}")
    print(f"[INFO] Variance PC3 + PC4 : {explained_var[2:4].sum():.2%}")

    # === Clustering ===
    methode_du_coude(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    data_pca["cluster"] = kmeans.fit_predict(X)
    centres = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("\n[INFO] Centres des clusters :\n", centres)

    for dep in data_pca["dep"].unique():
        subset = data_pca[data_pca["dep"] == dep]
        visualisation_clusters_pair(subset, dep)
        visualisation_clusters_3D(subset, dep)

    # === Régression linéaire sur PCA ===
    for var in ["T", "U", "P", "FF"]:
        data_pca[var] = data[var].values

    print("\n[INFO] Régression linéaire : prédiction des composantes principales")
    for i in range(1, 5):
        regression_lineaire(data_pca, explicatives=["T", "U"], cible=f"PC{i}")

    # Optionnel : réduire les colonnes
    data = filtrer_colonnes_utiles(data)
