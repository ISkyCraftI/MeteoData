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

#  Programme principal 
if __name__ == "__main__":
    # Chargement brut
    data = charger_donnees_departements(files)
    print(f"[INFO] Dimensions brutes : {data.shape}")

    # Nettoyage : retourne uniquement les données horaires
    data = nettoyer_donnees(data, verbose=True)

    # Conversion unités
    data["P"] = data["PSTAT"]

    #  Visualisations météo (par heure) A DECOMMENTER
    # boiteAMoustache(data)
    # correlation(data,seuil_corr=0.5)
    # courbe_temperature_par_departement(data)
    # courbes_variables(data)
    # boxplot_temperature(data)
    # boxplot_variable(data)
    # hist_temperature(data)
    # hist_variable(data)
    
    # Statistiques descriptives (sur les données horaires)
    stats = data.groupby("dep").apply(lambda x: statistiques(x[["T", "U", "P", "FF"]]))
    print("\n[INFO] Statistiques descriptives :\n", stats)

    # # Corrélation par département A DECOMMENTER
    # for dep in data["dep"].unique():
    #     print(f"\n[INFO] Corrélation - Département {dep}")
    #     heatmap_correlation(data[data["dep"] == dep], dep=dep)
    print("COLONNES\n : ")
    data.columns 
    #  ACP 
    features = ["T", "U", "P", "FF"]
    data_clean = data.dropna(subset=features).copy()
    X = StandardScaler().fit_transform(data_clean[features])
    data_pca, explained_var = appliquer_pca(data_clean, features)
    data_pca["dep"] = data_clean["dep"].values

    print(f"\n[INFO] Variance PC1 + PC2 : {explained_var[:2].sum():.2%}")
    print(f"[INFO] Variance PC3 + PC4 : {explained_var[2:4].sum():.2%}")

    #  Clustering 
    # methode_du_coude(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    data_pca["cluster"] = kmeans.fit_predict(X)
    centres = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("\n[INFO] Centres des clusters :\n", centres)

print(data.columns)   
print(data_pca.columns)  
data_pca['FF'] = data_clean['FF'].values
data_pca['U'] = data_clean['U'].values
data_pca['T'] = data_clean['T'].values
data_pca['P'] = data_clean['P'].values
            
combinations = [
    (['FF', 'T'], 'PC1'), (['FF', 'T'], 'PC2'), (['FF', 'T'], 'PC3'), (['FF', 'T'], 'PC4'),
    (['FF', 'U'], 'PC1'), (['FF', 'U'], 'PC2'), (['FF', 'U'], 'PC3'), (['FF', 'U'], 'PC4'),
    (['FF', 'P'], 'PC1'), (['FF', 'P'], 'PC2'), (['FF', 'P'], 'PC3'), (['FF', 'P'], 'PC4'),
    (['T', 'U'], 'PC1'),  (['T', 'U'], 'PC2'),  (['T', 'U'], 'PC3'),  (['T', 'U'], 'PC4'),
    (['T', 'P'], 'PC1'),  (['T', 'P'], 'PC2'),  (['T', 'P'], 'PC3'),  (['T', 'P'], 'PC4'),
    (['U', 'P'], 'PC1'),  (['U', 'P'], 'PC2'),  (['U', 'P'], 'PC3'),  (['U', 'P'], 'PC4'),
]

# for dep in data_pca['dep'].unique():
#     sous_ensemble = data_pca[data_pca['dep'] == dep]

#     if len(sous_ensemble) < 10:
#         continue

#     print(f"\nDÉPARTEMENT : {dep}\n")
    
#     for explicatives, cible in combinations:
#         print(f"\nRégression : {cible} ~ {' + '.join(explicatives)}")
#         regression_lineaire(sous_ensemble, explicatives=explicatives, cible=cible)

#     # Optionnel : réduire les colonnes
#     data = filtrer_colonnes_utiles(data)

# Classification KNN et LDA via la fonction utilitaire
resultats_par_dep = classifier(
    data_pca,
    features=["PC1", "PC2", "PC3", "PC4"],
    target="cluster",
    n_neighbors=5,
    test_size=0.2
)
