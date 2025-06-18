import pandas as pd
import numpy as np

from importeur import detecter_fichiers_par_departement
from sequenceur import generer_sequences
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

# === Programme principal ===
if __name__ == "__main__":
    fichiers_detectes = detecter_fichiers_par_departement(racine=".")

    all_data = []

    for dep, fichiers in fichiers_detectes.items():
        print(f"\n Analyse du département {dep} ")

        # Lecture et concaténation des fichiers
        if isinstance(fichiers, list):
            frames = []
            for chemin in fichiers:
                df_part = pd.read_csv(chemin, compression='gzip', sep=';', low_memory=False)
                df_part["dep"] = dep
                frames.append(df_part)
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.read_csv(fichiers, compression='gzip', sep=';', low_memory=False)
            df["dep"] = dep

        all_data.append(df)

    # Fusion de tous les départements
    data = pd.concat(all_data, ignore_index=True)
    print(f"[INFO] Dimensions brutes : {data.shape}")

    # Nettoyage
    data = nettoyer_donnees(data, verbose=True)

    # Conversion d’unités
    data["P"] = data["PSTAT"]

    # Séquençage (optionnel, désactivé si tu veux juste faire du clustering)
    sequences = generer_sequences(data, colonnes=["T", "U", "P", "FF"], taille=24)
    print(f"[INFO] Séquences générées : {sequences.shape}")

    # Statistiques descriptives
    stats = data.groupby("dep").apply(lambda x: statistiques(x[["T", "U", "P", "FF"]]))
    print("\n[INFO] Statistiques descriptives :\n", stats)

    # ACP
    features = ["T", "U", "P", "FF"]
    data_clean = data.dropna(subset=features).copy()
    X = StandardScaler().fit_transform(data_clean[features])
    data_pca, explained_var = appliquer_pca(data_clean, features)
    data_pca["dep"] = data_clean["dep"].values

    print(f"\n[INFO] Variance PC1 + PC2 : {explained_var[:2].sum():.2%}")
    print(f"[INFO] Variance PC3 + PC4 : {explained_var[2:4].sum():.2%}")

    # Clustering
    methode_du_coude(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    data_pca["cluster"] = kmeans.fit_predict(X)
    centres = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("\n[INFO] Centres des clusters :\n", centres)

    # Copie des colonnes utiles pour les régressions
    for var in features:
        data_pca[var] = data_clean[var].values

    # Régressions par département
    combinations = [
        (['FF', 'T'], 'PC1'), (['FF', 'T'], 'PC2'), (['FF', 'T'], 'PC3'), (['FF', 'T'], 'PC4'),
        (['FF', 'U'], 'PC1'), (['FF', 'U'], 'PC2'), (['FF', 'U'], 'PC3'), (['FF', 'U'], 'PC4'),
        (['FF', 'P'], 'PC1'), (['FF', 'P'], 'PC2'), (['FF', 'P'], 'PC3'), (['FF', 'P'], 'PC4'),
        (['T', 'U'], 'PC1'),  (['T', 'U'], 'PC2'),  (['T', 'U'], 'PC3'),  (['T', 'U'], 'PC4'),
        (['T', 'P'], 'PC1'),  (['T', 'P'], 'PC2'),  (['T', 'P'], 'PC3'),  (['T', 'P'], 'PC4'),
        (['U', 'P'], 'PC1'),  (['U', 'P'], 'PC2'),  (['U', 'P'], 'PC3'),  (['U', 'P'], 'PC4'),
    ]

    for dep in data_pca["dep"].unique():
        sous_ensemble = data_pca[data_pca["dep"] == dep]
        if len(sous_ensemble) < 10:
            continue
        print(f"\nDÉPARTEMENT : {dep}")
        for explicatives, cible in combinations:
            regression_lineaire(sous_ensemble, explicatives=explicatives, cible=cible)

    # Classification KNN / LDA (à partir des PC)
    resultats_par_dep = classifier(
        data_pca,
        features=["PC1", "PC2", "PC3", "PC4"],
        target="cluster",
        n_neighbors=5,
        test_size=0.2
    )
    print("\n[INFO] Classification KNN/LDA terminée.")
    print("\n[INFO] Fin de l'analyse.")