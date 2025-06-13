
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def visualiser_clusters(df, n_clusters):
    # 1. Sélection des colonnes numériques
    df_numeric = df.select_dtypes(include=[np.number])
    print(f"[DEBUG] Colonnes numériques : {df_numeric.columns.tolist()}")
    print(f"[DEBUG] Taille avant dropna : {df_numeric.shape}")

    # 🛑 Vérifie si NaNs sont trop nombreux
    df_numeric = df_numeric.fillna(df_numeric.mean(numeric_only=True))

    print(f"[DEBUG] Taille après dropna : {df_numeric.shape}")

    if df_numeric.empty:
        print("[ERREUR] Aucune ligne complète sans NaN pour le clustering.")
        print("[SOLUTION] Essayez avec `fillna()` à la place de `dropna()`.")
        return

    # 2. Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # 3. Clustering KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    # 4. Réduction dimensionnelle (PCA)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 5. Visualisation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='tab10', s=60)
    plt.title("Visualisation des clusters (PCA)")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

