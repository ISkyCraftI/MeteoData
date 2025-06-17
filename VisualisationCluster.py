import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

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

def classifier(data_pca, features, target, n_neighbors=5, test_size=0.3):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    for dep in data_pca["dep"].unique():
        print(f"\n Département : {dep} ")

        subset = data_pca[data_pca["dep"] == dep]

        if len(subset) < 10:
            print("Pas assez de données pour ce département.")
            continue

        X = subset[features]
        y = subset[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        # KNN
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        knn_score = knn.score(X_test, y_test)
        y_knn_pred = knn.predict(X_test)

        print("\nKNN Classification Report :")
        print(classification_report(y_test, y_knn_pred))
        print("KNN Accuracy :", knn_score)

        # LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)
        lda_score = lda.score(X_test, y_test)
        y_lda_pred = lda.predict(X_test)

        print("\nLDA Classification Report :")
        print(classification_report(y_test, y_lda_pred))
        print("LDA Accuracy :", lda_score)

        # Matrice de confusion LDA
        cm = confusion_matrix(y_test, y_lda_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title(f"Matrice de confusion LDA — Dép. {dep}")
        plt.xlabel("Prédictions")
        plt.ylabel("Réel")
        plt.tight_layout()
        plt.show()

