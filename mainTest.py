import pandas as pd
import numpy as np
from nettoyage import conversion_virgules
from moyMedEcTyp import *
from courbes import *
from VisualtionsRedDim import *
from reductionDim import *
from methodeCoude import *

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Chargement multi-départements
files = {
    "29": "D29/H_29_2020-2023.csv.gz",
    "21": "D21/H_21_previous-2020-2023.csv.gz",
    # "2A": "H_2A_xxxx.csv.gz"  # optionnel
}

def charger_donnees_departements(files_dict):
    dfs = {}
    for dep, file in files_dict.items():
        df = pd.read_csv(file, compression='gzip', sep=';')
        df["dep"] = dep
        dfs[dep] = df
    return pd.concat(dfs.values(), ignore_index=True)

# Programme principal
if __name__ == "__main__":
    # Chargement brut
    data = charger_donnees_departements(files)
    print(f"Données brutes : {data.shape}")
    
    # Conversion de date et nettoyage
    data["date"] = pd.to_datetime(data["AAAAMMJJHH"], format="%Y%m%d%H", errors="coerce")
    data = data.dropna(subset=["date", "T", "U", "RR1", "FF", "DD", "PSTAT"])

    # Préparation des colonnes utiles
    data = data[["date", "T", "U", "FF", "DD", "PSTAT", "dep"]].copy()
    courbe_moyenne_par_mois(data, colonne="T", label="Température")
    courbe_moyenne_par_mois(data, colonne="RR1", label="Précipitations")  # Pas trouvé
    courbe_moyenne_par_mois(data, colonne="U", label="Humidité")
    courbe_moyenne_par_mois(data, colonne="FF", label="Vent moyen")
    
    data["T"] /= 10
    data["FF"] /= 10
    data["P"] = data["PSTAT"] / 10  # pression en hPa

    # Statistiques
    stats = data.groupby("dep").apply(lambda x: statistiques(x[["T", "U", "P", "FF"]]))
    print("\nStatistiques descriptives :\n", stats)

    # Courbes temporelles
    courbe_temperature_par_departement(data)
    courbes_variables(data)
    boxplot_temperature(data)
    hist_temperature(data)

    # Corrélation
    heatmap_correlation(data)

    # PCA
    features = ["T", "U", "P", "FF"]
    X = StandardScaler().fit_transform(data[features])
    data_pca, explained_var = appliquer_pca(data, features)
    data_pca["dep"] = data["dep"].values

    print(f"\nVariance expliquée par les 2 premières composantes : {explained_var[0]:.2%} + {explained_var[1]:.2%} = {explained_var.sum():.2%}")

    # Clustering
    methode_du_coude(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    data_pca["cluster"] = kmeans.fit_predict(X)
    centres = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("\nCentres des clusters :\n", centres)

    # visualisation_clusters(data_pca, X, features)

