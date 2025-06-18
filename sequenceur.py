import numpy as np

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