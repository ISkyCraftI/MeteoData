import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Chargement des fichiers
files = {
    "29": "D29/H_29_2020-2023.csv.gz",
    "21": "D21/H_21_previous-2020-2023.csv.gz",
    # "2A": "H_2A_xxxx.csv.gz"  # à ajouter si dispo
}

dfs = {}
for dep, file in files.items():
    df = pd.read_csv(file, compression='gzip', sep=';')
    print(f"Colonnes du fichier {file} :\n", df.columns.tolist())  # debug
    df["dep"] = dep
    dfs[dep] = df

data = pd.concat(dfs.values(), ignore_index=True)

# Conversion des types et nettoyage
data["date"] = pd.to_datetime(data["AAAAMMJJHH"], format="%Y%m%d%H", errors="coerce")
data = data.dropna(subset=["date", "T", "U", "FF", "DD", "PSTAT"])

# Sélection et mise à l'échelle
data = data[["date", "T", "U", "FF", "DD", "PSTAT", "dep"]].copy()
data["T"] = data["T"] / 10
data["FF"] = data["FF"] / 10
data["P"] = data["PSTAT"] / 10  # pression en hPa

# Statistiques descriptives
def statistiques(df):
    return df.describe().loc[["mean", "50%", "std"]].rename(index={"50%": "median"})

stats = data.groupby("dep").apply(lambda x: statistiques(x[["T", "U", "P", "FF"]]))
print("\n📊 Statistiques descriptives :\n", stats)

#  Visualisation Température avec matplotlib
plt.figure(figsize=(12, 6))
for dep in data["dep"].unique():
    subset = data[data["dep"] == dep]
    plt.plot(subset["date"], subset["T"], label=f"Dép {dep}", linewidth=0.5)
plt.legend()
plt.title("Température (°C) par département")
plt.xlabel("Date")
plt.ylabel("Température")
plt.tight_layout()
plt.show()

#  Autres courbes : U, FF, P avec matplotlib
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for dep in data["dep"].unique():
    subset = data[data["dep"] == dep]
    axs[0].plot(subset["date"], subset["U"], label=f"Dép {dep}", linewidth=0.5)
    axs[1].plot(subset["date"], subset["FF"], label=f"Dép {dep}", linewidth=0.5)
    axs[2].plot(subset["date"], subset["P"], label=f"Dép {dep}", linewidth=0.5)

axs[0].set_title("Humidité (%)")
axs[1].set_title("Vent moyen (m/s)")
axs[2].set_title("Pression (hPa)")
axs[2].set_xlabel("Date")
for ax in axs:
    ax.legend()
    ax.set_ylabel("Valeur")
plt.tight_layout()
plt.show()

#  Boxplot Température (garde seaborn ici, plus simple)
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="dep", y="T")
plt.title("Répartition des températures par département")
plt.show()

#  Histogramme (matplotlib)
plt.figure(figsize=(12, 5))
for dep in data["dep"].unique():
    subset = data[data["dep"] == dep]
    plt.hist(subset["T"], bins=50, alpha=0.5, label=f"Dép {dep}")
plt.legend()
plt.title("Histogramme des températures")
plt.xlabel("Température (°C)")
plt.ylabel("Fréquence")
plt.show()

# Matrice de corrélation (on garde seaborn pour heatmap)
corr = data[["T", "U", "P", "FF"]].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()

# PCA
features = ["T", "U", "P", "FF"]
X = StandardScaler().fit_transform(data[features])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
data_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
data_pca["dep"] = data["dep"].values

#  Affichage variance expliquée
explained_var = pca.explained_variance_ratio_
print(f"\n Variance expliquée par les 2 premières composantes : {explained_var[0]:.2%} + {explained_var[1]:.2%} = {explained_var.sum():.2%}")

#  KMeans – méthode du coude
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 10), inertias, marker='o')
plt.title("Méthode du coude - KMeans")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.grid()
plt.show()

#  KMeans final (ex: k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
data_pca["cluster"] = kmeans.fit_predict(X)

#  Centres des clusters
centres = pd.DataFrame(kmeans.cluster_centers_, columns=features)
print("\n Centres des clusters KMeans :\n", centres)

#  Visualisation des clusters (matplotlib)
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for cluster in sorted(data_pca["cluster"].unique()):
    subset = data_pca[data_pca["cluster"] == cluster]
    plt.scatter(subset["PC1"], subset["PC2"], s=10, color=colors[cluster], label=f"Cluster {cluster}")
plt.title("Clustering météorologique")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.show()
