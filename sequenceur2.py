import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

from nettoyage import nettoyer_donnees
from reductionDim import appliquer_pca


def traiter_par_blocs(fichier, dep, chunk_size=10000):
    reader = pd.read_csv(fichier, compression='gzip', sep=';', low_memory=False, chunksize=chunk_size)
    resultats = []
    erreurs = {}
    total_chunks = 0
    chunks_valides = 0

    for i, chunk in enumerate(reader):
        total_chunks += 1
        print(f"[{dep}] Bloc {i} en cours...")
        chunk['dep'] = dep
        try:
            chunk = nettoyer_donnees(chunk, verbose=False)
            chunk["P"] = chunk["PSTAT"]
            chunk = chunk.dropna(subset=["T", "U", "P", "FF"])

            X = StandardScaler().fit_transform(chunk[["T", "U", "P", "FF"]])
            pca_df, _ = appliquer_pca(chunk, ["T", "U", "P", "FF"])
            pca_df["dep"] = dep
            pca_df["cluster"] = KMeans(n_clusters=3, random_state=42, n_init='auto').fit_predict(X)

            for var in ["T", "U", "P", "FF"]:
                pca_df[var] = chunk[var].values
            pca_df["date"] = chunk["date"]

            resultats.append(pca_df)
            chunks_valides += 1

        except Exception as e:
            err_type = str(e).split(":")[0].strip()
            erreurs[err_type] = erreurs.get(err_type, 0) + 1
            print(f"[{dep}] Chunk {i} ignoré : {e}")

    if erreurs:
        print(f"\n[{dep}] ▶ Résumé des erreurs par type :")
        for typ, count in erreurs.items():
            print(f"[{dep}] - {typ!r} : {count} chunks ignorés")

    df_concat = pd.concat(resultats, ignore_index=True) if resultats else pd.DataFrame()
    return df_concat, total_chunks, chunks_valides
