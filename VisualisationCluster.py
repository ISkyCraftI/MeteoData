import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

def visualisation_clusters(df_pca, X, features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X)
    df_pca["cluster"] = clusters

    centres = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    print("\n Centres des clusters :\n", centres)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="cluster",
        style="dep",
        palette="Set2",
        s=20
    )
    plt.title(f"Clustering météorologique par KMeans ({n_clusters} clusters)")
    plt.xlabel("Composante principale 1 (PC1)")
    plt.ylabel("Composante principale 2 (PC2)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()
