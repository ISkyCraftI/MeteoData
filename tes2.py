import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nettoyage import conversion_virgules, nettoyer_donnees
from moyMedEcTyp import *
from courbes import boxplot_variable
from courbes import *
from VisualtionsRedDim import *
from reductionDim import *
from methodeCoude import *
from VisualisationCluster import classifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

def filtrer_colonnes_utiles(df):
    colonnes = ["date", "T", "U", "RR1", "FF", "DD", "PSTAT", "P", "dep"]
    colonnes_presentes = [col for col in colonnes if col in df.columns]
    return df[colonnes_presentes].copy()

#  Programme principal 
if __name__ == "__main__":
    # Chargement brut
    data = charger_donnees_departements(files)
    print(f"[INFO] Dimensions brutes : {data.shape}")

    # Nettoyage : retourne uniquement les données horaires
    data = nettoyer_donnees(data, verbose=True)

    # Conversion unités
    data["P"] = data["PSTAT"]
    
    # Courbes moyennes mensuelles pour chaque variable
    # courbe_moyenne_par_mois(data, colonne="T", label="Température", group_by_dep=True)
    # courbe_moyenne_par_mois(data, colonne="RR1", label="Précipitations", group_by_dep=True)
    # courbe_moyenne_par_mois(data, colonne="U", label="Humidité", group_by_dep=True)
    # courbe_moyenne_par_mois(data, colonne="FF", label="Vent moyen", group_by_dep=True)
    
    # Conversion des unités
    data["P"] = data["PSTAT"] # pression en hPa

    # Statistiques par département
    stats = data.groupby("dep").apply(lambda x: statistiques(x[["T", "U", "P", "FF"]]))
    print("\nStatistiques descriptives :\n", stats)

    # Courbes temporelles
    boiteAMoustache(data)
    correlation(data,seuil_corr=0.5)
    courbe_temperature_par_departement(data)
    courbes_variables(data)
    boxplot_temperature(data)
    boxplot_variable(data)
    hist_temperature(data)
    hist_variable(data)

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
    
combinations = [
    (['FF', 'T'], 'PC1'), (['FF', 'T'], 'PC2'), (['FF', 'T'], 'PC3'), (['FF', 'T'], 'PC4'),
    (['FF', 'U'], 'PC1'), (['FF', 'U'], 'PC2'), (['FF', 'U'], 'PC3'), (['FF', 'U'], 'PC4'),
    (['FF', 'P'], 'PC1'), (['FF', 'P'], 'PC2'), (['FF', 'P'], 'PC3'), (['FF', 'P'], 'PC4'),
    (['T', 'U'], 'PC1'),  (['T', 'U'], 'PC2'),  (['T', 'U'], 'PC3'),  (['T', 'U'], 'PC4'),
    (['T', 'P'], 'PC1'),  (['T', 'P'], 'PC2'),  (['T', 'P'], 'PC3'),  (['T', 'P'], 'PC4'),
    (['U', 'P'], 'PC1'),  (['U', 'P'], 'PC2'),  (['U', 'P'], 'PC3'),  (['U', 'P'], 'PC4'),
]

for dep in data_pca['dep'].unique():
    sous_ensemble = data_pca[data_pca['dep'] == dep]

    if len(sous_ensemble) < 10:
        continue

    print(f"\nDÉPARTEMENT : {dep}\n")
    
    for explicatives, cible in combinations:
        print(f"\nRégression : {cible} ~ {' + '.join(explicatives)}")
        regression_lineaire(sous_ensemble, explicatives=explicatives, cible=cible)
    

# Classification KNN et LDA via la fonction utilitaire
resultats_par_dep = classifier(
    data_pca,
    features=["PC1", "PC2", "PC3", "PC4"],
    target="cluster",
    n_neighbors=5,
    test_size=0.3
)
