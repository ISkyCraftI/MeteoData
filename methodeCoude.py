from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def methode_du_coude(X, max_k=10):
    inertias = []
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, max_k), inertias, marker='o')
    plt.title("Méthode du coude - KMeans")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie")
    plt.grid()
    plt.show()
