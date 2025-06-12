import numpy as np


def supprimer_colonnes_correlees(df, seuil=0.98):
    colonnes = df.columns[2:]  # On ignore les 2 premières colonnes (ex : identifiants)
    df_numeric = df[colonnes]

    # Calcul de la matrice de corrélation absolue
    corr_matrix = df_numeric.corr().abs()

    # On ignore la diagonale
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Colonnes à supprimer
    colonnes_a_supprimer = [
        column for column in upper.columns if any(upper[column] > seuil)
    ]

    print(f"Colonnes supprimées car corrélées à ≥ {seuil*100:.0f}% : {colonnes_a_supprimer}")

    # Suppression dans le dataframe complet
    df = df.drop(columns=colonnes_a_supprimer)
    return df