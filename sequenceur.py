import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

from nettoyage import nettoyer_donnees
from reductionDim import appliquer_pca

def traiter_par_blocs(fichier, dep, chunk_size=1000):
    reader = pd.read_csv(fichier, compression='gzip', sep=';', low_memory=False, chunksize=chunk_size)
    resultats = []

    for i, chunk in enumerate(reader):
        print(f"[{dep}] Bloc {i} en cours...")
        chunk['dep'] = dep
        try:
            chunk = nettoyer_donnees(chunk, verbose=False)

            # On vérifie que la colonne "date" est bien présente après nettoyage
            if "date" not in chunk.columns:
                raise ValueError("Colonne 'date' manquante après nettoyage.")

            # Filtrage minimal pour éviter les NaN massifs
            features = ["T", "U", "P", "FF"]
            chunk = chunk.dropna(subset=features)

            # PCA locale
            X = StandardScaler().fit_transform(chunk[features])
            pca_df, _ = appliquer_pca(chunk, features)
            pca_df["dep"] = dep
            pca_df["cluster"] = KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(X)

            # Réinjecte les variables d'origine + date
            for var in features + ["date"]:
                pca_df[var] = chunk[var].values

            resultats.append(pca_df)
        except Exception as e:
            print(f"[{dep}] Chunk {i} ignoré : {e}")

    return pd.concat(resultats, ignore_index=True)

