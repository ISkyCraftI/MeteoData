import pandas as pd
import numpy as np

from importeur import *
from sequenceur2 import traiter_par_blocs
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
    df_global = []
    resume_chunks = {}

    for dep, fichiers in fichiers_detectes.items():
        total_chunks = 0
        valid_chunks = 0

        for fichier in fichiers:
            print(f"\n===== Traitement par blocs : {fichier} =====")
            resultat, total, valides = traiter_par_blocs(fichier, dep, chunk_size=10000)
            df_global.append(resultat)
            total_chunks += total
            valid_chunks += valides

        resume_chunks[dep] = (valid_chunks, total_chunks)

    # Résumé global
    print("\n===== Résumé du traitement par département =====")
    for dep, (valides, total) in resume_chunks.items():
        print(f"[{dep}] Chunks valides : {valides}/{total} ({(valides/total)*100:.1f}%)")

    # Fusion des résultats
    data_pca = pd.concat(df_global, ignore_index=True)

    # === Visualisations météo ===
    boiteAMoustache(data_pca)
    correlation(data_pca, seuil_corr=0.5)
    courbe_temperature_par_departement(data_pca)
    courbes_variables(data_pca)
    boxplot_temperature(data_pca)
    boxplot_variable(data_pca)
    hist_temperature(data_pca)
    hist_variable(data_pca)

    # Statistiques descriptives
    stats = data_pca.groupby("dep").apply(lambda x: statistiques(x[["T", "U", "P", "FF"]]))
    print("\n[INFO] Statistiques descriptives :\n", stats)

    # ACP
    features = ["T", "U", "P", "FF"]
    data_clean = data_pca.dropna(subset=features).copy()
    X = StandardScaler().fit_transform(data_clean[features])
    data_pca, explained_var = appliquer_pca(data_clean, features)
    data_pca["dep"] = data_clean["dep"].values

    print(f"\n[INFO] Variance PC1 + PC2 : {explained_var[:2].sum():.2%}")
    print(f"[INFO] Variance PC3 + PC4 : {explained_var[2:4].sum():.2%}")

    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    data_pca["cluster"] = kmeans.fit_predict(X)
    centres = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("\n[INFO] Centres des clusters :\n", centres)

    for var in features:
        data_pca[var] = data_clean[var].values

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

    # Classification finale
    resultats_par_dep = classifier(
        data_pca,
        features=["PC1", "PC2", "PC3", "PC4"],
        target="cluster",
        n_neighbors=5,
        test_size=0.2
    )
