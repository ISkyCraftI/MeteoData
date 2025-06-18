
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def correlation(df, seuil_corr):
    # Ne garder que les colonnes numériques
    df_notes = df.select_dtypes(include=[np.number])

    print("[DEBUG] Colonnes numériques sélectionnées :", df_notes.columns.tolist())
    print("[DEBUG] Types de colonnes :")
    print(df_notes.dtypes)

    # Matrice de corrélation
    matrice = df_notes.corr()

    # Mise à zéro des corrélations faibles
    matrice_filtrée = matrice.where(matrice >= 0.40, 0)

    # Compter les corrélations fortes (≥ seuil), sans compter la diagonale
    seuil_nb = 3
    matrice_sans_diag = matrice.copy()
    np.fill_diagonal(matrice_sans_diag.values, 0)
    ue_a_conserver = (matrice_sans_diag >= seuil_corr).sum(axis=1) >= seuil_nb

    # Filtrage final
    colonnes_a_garder = ue_a_conserver[ue_a_conserver].index
    matrice_finale = matrice.loc[colonnes_a_garder, colonnes_a_garder]

    # Affichage
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrice_finale,
        annot=True,
        fmt=".2f",
        cmap="bwr",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor='white'
    )
    plt.title(f"Heatmap de nos colonnes")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()


def heatmap_correlation(data, dep=None):
    corr = data[["T", "U", "P", "FF"]].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    
    # Titre dynamique avec numéro de département
    titre = f"Matrice de corrélation - Département {dep}" if dep else "Matrice de corrélation"
    plt.title(titre)
    plt.tight_layout()
    plt.show()

def visualisation_clusters_pair(data_pca, dep, n_clusters=4):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Clustering sur PC1 et PC2
    features_12 = ["PC1", "PC2"]
    kmeans_12 = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    data_pca["cluster_12"] = kmeans_12.fit_predict(data_pca[features_12])
    centers_12 = kmeans_12.cluster_centers_

    palette_12 = sns.color_palette("tab10", n_colors=n_clusters)
    for cluster in range(n_clusters):
        subset = data_pca[data_pca["cluster_12"] == cluster]
        axes[0].scatter(subset["PC1"], subset["PC2"], s=10, color=palette_12[cluster], label=f"Clust {cluster}")
    # Ajout des centres
    axes[0].scatter(centers_12[:, 0], centers_12[:, 1], 
                    c='black', marker='X', s=100, label='Centre')

    axes[0].set_title(f"Clustering sur PC1 vs PC2 - Dép {dep}")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    # Clustering sur PC3 et PC4
    features_34 = ["PC3", "PC4"]
    kmeans_34 = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    data_pca["cluster_34"] = kmeans_34.fit_predict(data_pca[features_34])
    centers_34 = kmeans_34.cluster_centers_

    palette_34 = sns.color_palette("tab20", n_colors=n_clusters)
    for cluster in range(n_clusters):
        subset = data_pca[data_pca["cluster_34"] == cluster]
        axes[1].scatter(subset["PC3"], subset["PC4"], s=10, color=palette_34[cluster], label=f"Clust {cluster}")
    # Ajout des centres
    axes[1].scatter(centers_34[:, 0], centers_34[:, 1], 
                    c='black', marker='X', s=100, label='Centre')

    axes[1].set_title(f"Clustering sur PC3 vs PC4 - Dép {dep}")
    axes[1].set_xlabel("PC3")
    axes[1].set_ylabel("PC4")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def visualisation_clusters_3D(data_pca, dep):
    unique_clusters = sorted(data_pca["cluster"].unique())
    palette = sns.color_palette("hsv", len(unique_clusters))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for i, cluster in enumerate(unique_clusters):
        subset = data_pca[data_pca["cluster"] == cluster]
        ax.scatter(
            subset["PC1"], subset["PC2"], subset["PC3"],
            color=palette[i], s=10, label=f"Cluster {cluster}"
        )

    ax.set_title(f"Clustering météorologique 3D - Dép {dep}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()
    plt.show()
