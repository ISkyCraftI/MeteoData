import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

def supprimer_colonnes_correlees(df, seuil=0.98):
    colonnes_a_exclure = ['NUM_POSTE', 'NOM_USUEL', 'AAAAMMJJHH']

    # Étape 1 : on exclut les colonnes non numériques
    df_temp = df.drop(columns=colonnes_a_exclure, errors='ignore')
    df_numeric = df_temp.loc[:, df_temp.apply(pd.api.types.is_numeric_dtype)]

    # Vérification debug
    print("[DEBUG] Colonnes numériques retenues :", df_numeric.columns.tolist())
    print("[DEBUG] Types :", df_numeric.dtypes)

    # Étape 2 : Matrice de corrélation
    corr_matrix = df_numeric.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    colonnes_a_supprimer = [
        column for column in upper.columns if any(upper[column] > seuil)
    ]

    print(f"[INFO] Colonnes supprimées car corrélées ≥ {seuil*100:.0f}% :", colonnes_a_supprimer)

    df_clean = df.drop(columns=colonnes_a_supprimer, errors='ignore')
    return df_clean

def appliquer_pca(df, features, n_components=4):
    # Étape 1 : Vérifier que les colonnes existent
    for col in features:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le DataFrame : {col}")

    # Étape 2 : Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(df[features])

    # Étape 3 : Vérification que tout est bien imputé
    if pd.DataFrame(X_imputed).isna().any().any():
        raise ValueError("Des NaN sont toujours présents après imputation.")

    # Étape 4 : Standardisation
    X_scaled = StandardScaler().fit_transform(X_imputed)

    # Étape 5 : PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Résultat
    columns = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=columns, index=df.index)

    return df_pca, pca.explained_variance_ratio_


