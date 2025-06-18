import pandas as pd
import numpy as np

from importeur import *
from sequenceur import traiter_par_blocs
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
    # Chargement automatique
    fichiers_detectes = detecter_fichiers_par_departement(racine=".")
    df_global = []

    for dep, fichiers in fichiers_detectes.items():
        for fichier in fichiers:
            print(f"\n===== Traitement par blocs : {fichier} =====")
            resultat = traiter_par_blocs(fichier, dep, chunk_size=10000)
            df_global.append(resultat)

    data_pca = pd.concat(df_global, ignore_index=True)

    # Statistiques descriptives
    stats = data_pca.groupby("dep").apply(lambda x: statistiques(x[["T", "U", "P", "FF"]]))
    print("\n[INFO] Statistiques descriptives :\n", stats)

    # Régressions multiples
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
        test_size=0.2
    )
