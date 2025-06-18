import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

from nettoyage import nettoyer_donnees
from reductionDim import appliquer_pca

def generer_sequences(df, colonnes_features, colonne_cible, n_pas=100, step=1):
    """
    Génère des séquences glissantes de longueur n_pas pour du ML/RNN/LSTM.

    Paramètres :
        df : DataFrame trié par date
        colonnes_features : colonnes d'entrée (ex: ['T', 'U', 'FF', 'P'])
        colonne_cible : variable à prédire (ex: 'T')
        n_pas : taille de la séquence (en lignes)
        step : intervalle de saut (1 = glissant, 24 = journalier)

    Retour :
        X : tableau (n_samples, n_pas, n_features)
        y : tableau (n_samples,)
    """
    df = df.copy().dropna(subset=colonnes_features + [colonne_cible])
    X, y = [], []

    donnees = df[colonnes_features].values
    cibles = df[colonne_cible].values

    for i in range(0, len(df) - n_pas, step):
        seq_X = donnees[i:i+n_pas]
        seq_y = cibles[i+n_pas]
        X.append(seq_X)
        y.append(seq_y)

    return np.array(X), np.array(y)

def traiter_par_blocs(fichier, dep, chunk_size=1000):
    reader = pd.read_csv(fichier, compression='gzip', sep=';', low_memory=False, chunksize=chunk_size)
    resultats = []

    for i, chunk in enumerate(reader):
        print(f"[{dep}] Bloc {i} en cours...")
        chunk['dep'] = dep
        try:
            chunk = nettoyer_donnees(chunk, verbose=False)
            chunk["P"] = chunk["PSTAT"]

            # Filtrage minimal pour éviter les NaN massifs
            features = ["T", "U", "P", "FF"]
            chunk = chunk.dropna(subset=features)

            # PCA locale
            X = StandardScaler().fit_transform(chunk[features])
            pca_df, _ = appliquer_pca(chunk, features)
            pca_df["dep"] = dep
            pca_df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init='auto').fit_predict(X)

            # Réinjecte les variables d'origine si besoin
            for var in features:
                pca_df[var] = chunk[var].values

            resultats.append(pca_df)
        except Exception as e:
            print(f"[{dep}] Chunk {i} ignoré : {e}")
    
    return pd.concat(resultats, ignore_index=True)
