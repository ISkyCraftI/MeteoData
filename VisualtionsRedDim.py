
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def correlation(df, seuil_corr):
    # Ne garder que les colonnes numériques
    df_notes = df.select_dtypes(include=[np.number])

    print("[DEBUG] Colonnes numériques sélectionnées :", df_notes.columns.tolist())
    print("[DEBUG] Types de colonnes :")
    print(df_notes.dtypes)

    # Matrice de corrélation
    matrice = df_notes.corr()

    # Mise à zéro des corrélations faibles
    matrice_filtrée = matrice.where(matrice >= 0.40, 0)

    # Compter les corrélations fortes (≥ seuil), sans compter la diagonale
    seuil_nb = 3
    matrice_sans_diag = matrice.copy()
    np.fill_diagonal(matrice_sans_diag.values, 0)
    ue_a_conserver = (matrice_sans_diag >= seuil_corr).sum(axis=1) >= seuil_nb

    # Filtrage final
    colonnes_a_garder = ue_a_conserver[ue_a_conserver].index
    matrice_finale = matrice.loc[colonnes_a_garder, colonnes_a_garder]

    # Affichage
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrice_finale,
        annot=True,
        fmt=".2f",
        cmap="bwr",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor='white'
    )
    plt.title(f"Precipitations  ≥ {seuil_nb} corrélations fortes (temperatures) (≥ {seuil_corr})")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()
