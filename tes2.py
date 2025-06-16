import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nettoyage import conversion_virgules
from moyMedEcTyp import *
from courbes import *
from VisualtionsRedDim import *
from reductionDim import *
from methodeCoude import *

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from regression import regression_lineaire

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

# Fonction intégrée pour les courbes de moyennes mensuelles
def courbe_moyenne_par_mois(df, colonne, label="", group_by_dep=False):
    df["mois"] = df["date"].dt.to_period("M").dt.to_timestamp()

    plt.figure(figsize=(10, 5))
    if group_by_dep:
        for dep in df["dep"].unique():
            df_dep = df[df["dep"] == dep]
            serie = df_dep.groupby("mois")[colonne].mean()
            plt.plot(serie.index, serie.values, label=f"Dép {dep}")
        plt.title(f"{label} moyenne par mois (par département)")
    else:
        serie = df.groupby("mois")[colonne].mean()
        plt.plot(serie.index, serie.values, label=label)
        plt.title(f"{label} moyenne par mois")

    plt.xlabel("Mois")
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Programme principal
if __name__ == "__main__":
    # Chargement brut
    data = charger_donnees_departements(files)
    print(f"Données brutes : {data.shape}")
    
    # Conversion de date et nettoyage
    data["date"] = pd.to_datetime(data["AAAAMMJJHH"], format="%Y%m%d%H", errors="coerce")
    data = data.dropna(subset=["date", "T", "U", "RR1", "FF", "DD", "PSTAT"])

    # Préparation des colonnes utiles
    data = data[["date", "T", "U", "RR1", "FF", "DD", "PSTAT", "dep"]].copy()
    
    # Courbes moyennes mensuelles pour chaque variable
    courbe_moyenne_par_mois(data, colonne="T", label="Température", group_by_dep=True)
    courbe_moyenne_par_mois(data, colonne="RR1", label="Précipitations", group_by_dep=True)
    courbe_moyenne_par_mois(data, colonne="U", label="Humidité", group_by_dep=True)
    courbe_moyenne_par_mois(data, colonne="FF", label="Vent moyen", group_by_dep=True)
    
    # Conversion unités
    data["T"] /= 10
    data["FF"] /= 10
    data["P"] = data["PSTAT"] / 10  # pression en hPa

    # Statistiques par département
    stats = data.groupby("dep").apply(lambda x: statistiques(x[["T", "U", "P", "FF"]]))
    print("\nStatistiques descriptives :\n", stats)

    # Courbes temporelles
    courbe_temperature_par_departement(data)
    courbes_variables(data)
    boxplot_temperature(data)
    hist_temperature(data)

    # Corrélations par département
    for dep in data["dep"].unique():    
        print(f"\nCorrélation pour le département {dep} :")
        df_dep = data[data["dep"] == dep]
        heatmap_correlation(df_dep, dep = dep)

    # PCA
    features = ["T", "U", "P", "FF"]
    X = StandardScaler().fit_transform(data[features])
    data_pca, explained_var = appliquer_pca(data, features)
    data_pca["dep"] = data["dep"].values

    print(f"\nVariance expliquée par les 2 premières composantes : {explained_var[0]:.2%} + {explained_var[1]:.2%} = {explained_var[:2].sum():.2%}")
    print(f"Variance expliquée par les composantes 3 et 4 : {explained_var[2]:.2%} + {explained_var[3]:.2%} = {explained_var[2:4].sum():.2%}")

    # Clustering
    methode_du_coude(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    data_pca["cluster"] = kmeans.fit_predict(X)
    centres = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("\nCentres des clusters :\n", centres)


    features_pca = ["PC1", "PC2", "PC3", "PC4"]
    departements = data_pca["dep"].unique()

    for dep in departements:
        subset = data_pca[data_pca["dep"] == dep].copy()

        visualisation_clusters_pair(subset, dep)
        
        visualisation_clusters_3D(subset, dep)  
        
    print(data.columns)   
    print(data_pca.columns)  
    data_pca['FF'] = data['FF'].values
    data_pca['U'] = data['U'].values
    data_pca['T'] = data['T'].values
    data_pca['P'] = data['P'].values
    regression_lineaire(data_pca, explicatives =['FF','T'], cible='PC1')