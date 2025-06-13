import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

# Chargement des fichiers
files = {
    "29": "D29/H_29_2020-2023.csv.gz",
    "21": "D21/H_21_previous-2020-2023.csv.gz",
}

dfs = {}
for dep, file in files.items():
    df = pd.read_csv(file, compression='gzip', sep=';')
    df["dep"] = dep
    dfs[dep] = df

data = pd.concat(dfs.values(), ignore_index=True)

# Conversion et nettoyage
data["date"] = pd.to_datetime(data["AAAAMMJJHH"], format="%Y%m%d%H", errors="coerce")
data = data.dropna(subset=["date", "RR1", "T", "U", "FF", "DD", "PSTAT"])
data["T"] = data["T"] / 10
data["FF"] = data["FF"] / 10
data["P"] = data["PSTAT"] / 10

# Variances
print("\nVariances des variables :")
print(data[["RR1", "T", "U", "P", "FF"]].var())

# Statistiques descriptives
def statistiques(df):
    return df.describe().loc[["mean", "50%", "std"]].rename(index={"50%": "median"})
stats = data.groupby("dep").apply(lambda x: statistiques(x[["RR1", "T", "U", "P", "FF"]]))
print("\nStatistiques descriptives :\n", stats)

# Matrice de corrélation
corr = data[["RR1", "T", "U", "P", "FF"]].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()

# Standardisation et PCA 3 composantes
features = ["RR1", "T", "U", "P", "FF"]
X = data[features].dropna()
dep_labels = data.loc[X.index, "dep"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
print(f"\nVariance expliquée : PC1 = {explained_var[0]:.2%}, PC2 = {explained_var[1]:.2%}, PC3 = {explained_var[2]:.2%}, Total = {explained_var.sum():.2%}")

# Méthode du coude
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)
plt.figure(figsize=(6, 4))
plt.plot(range(1, 10), inertias, marker='o')
plt.title("Méthode du coude (ACP 3 composantes)")
plt.xlabel("Nb clusters")
plt.ylabel("Inertie")
plt.grid()
plt.tight_layout()
plt.show()

# KMeans final k=4
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Centres des clusters
centres_pca = pd.DataFrame(kmeans.cluster_centers_, columns=["PC1", "PC2", "PC3"])
print("\nCentres des clusters dans l'espace ACP :")
print(centres_pca)

# DataFrame PCA pour affichage
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
df_pca["cluster"] = clusters
df_pca["dep"] = dep_labels

# Visualisation 2D (PC1 / PC2) avec ronds et carrés
plt.figure(figsize=(9, 7))
for dep, marker in zip(["29", "21"], ['o', 's']):
    subset = df_pca[df_pca["dep"] == dep]
    plt.scatter(subset["PC1"], subset["PC2"],
                c=subset["cluster"], cmap='tab10', marker=marker, s=10, label=f'Dép {dep}')
plt.title("Clustering KMeans (PC1 vs PC2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.show()

# Visualisation 3D (PC1 / PC2 / PC3) avec ronds, carrés et centres différenciés
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# Points par département
for dep, marker in zip(["29", "21"], ['o', 's']):
    subset = df_pca[df_pca["dep"] == dep]
    ax.scatter(subset["PC1"], subset["PC2"], subset["PC3"],
               c=subset["cluster"], cmap='tab10', s=10, marker=marker, label=f'Dép {dep}')

# Centres des clusters : croix pour Dép 29 et losanges pour Dép 21 (globalement car même clustering)
ax.scatter(centres_pca["PC1"], centres_pca["PC2"], centres_pca["PC3"],
           c='black', s=150, marker='x', label='Centres Dép 29')
ax.scatter(centres_pca["PC1"], centres_pca["PC2"], centres_pca["PC3"],
           c='black', s=120, marker='D', label='Centres Dép 21')

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Clustering KMeans (ACP 3 composantes)")
ax.legend()
plt.tight_layout()
plt.show()

# Régression linéaire
y = data["T"]
X_reg = data[["U", "FF", "P", "RR1"]]
X_reg = X_reg.dropna()
y = y.loc[X_reg.index]
X_train, X_test, y_train, y_test = train_test_split(X_reg, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\n Coefficients de la régression :\n{pd.Series(model.coef_, index=X_reg.columns)}")
print(f"Intercept : {model.intercept_:.2f}")
print(f"R² sur le test : {r2_score(y_test, y_pred):.2f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
