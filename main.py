import numpy as np
import pandas as pd
import os

from nettoyage import *
from courbes import *
from VisualtionsRedDim import *
from reductionDim import *
from methodeCoude import *
from regression import regression_lineaire
from VisualisationCluster import *

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Dictionnaire des fichiers multi-départements
files = {
    "29": "D29/H_29_2020-2023.csv.gz",
    "21": "D21/H_21_previous-2020-2023.csv.gz",
}

def charger_donnees_departements(files_dict):
    dfs = []
    for dep, path in files_dict.items():
        print(f"[INFO] Chargement : {path}")
        df = pd.read_csv(path, sep=';', compression='infer', low_memory=False)
        df["dep"] = dep
        df.columns = df.columns.str.strip()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ======== Programme principal ========
if __name__ == "__main__":
    # Chargement brut
    data = charger_donnees_departements(files)
    print(f"[INFO] Dimensions brutes : {data.shape}")

    # Nettoyage
    data = nettoyer_donnees(data, verbose=True)

    # Conversion unités
    data["T"] /= 10
    data["FF"] /= 10
    data["P"] = data["PSTAT"] / 10

    # Colonnes utiles
    data = data[["date", "T", "U", "RR1", "FF", "DD", "P", "dep"]].dropna()

    # === Visualisations ===
    boiteAMoustache(data)
    NuagePointsTemperature(data)
    courbe_temperature_par_departement(data)
    courbes_variables(data)
    hist_temperature(data)
    boxplot_temperature(data)
    courbe_moyenne_par_mois(data, colonne="T", label="Température", group_by_dep=True)

    # === Corrélation ===
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
    print("\nCentres des clusters :\n", centres)

    for dep in data_pca["dep"].unique():
        subset = data_pca[data_pca["dep"] == dep]
        visualisation_clusters_pair(subset, dep)
        visualisation_clusters_3D(subset, dep)

    # === Régression linéaire ===
    data_pca["T"] = data["T"].values
    data_pca["U"] = data["U"].values
    data_pca["P"] = data["P"].values
    data_pca["FF"] = data["FF"].values

    print("\n[INFO] Régression T+U sur PC1 → PC4")
    for i in range(1, 5):
        regression_lineaire(data_pca, explicatives=["T", "U"], cible=f"PC{i}")
