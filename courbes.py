import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def boiteAMoustache(df, verbose=False):
    # Colonnes à exclure manuellement (identifiants, horodatages, etc.)
    colonnes_exclues = ['NUM_POSTE', 'AAAAMMJJHH']

    # Sélectionne uniquement les colonnes numériques
    colonnes_numeriques = df.select_dtypes(include=[np.number]).columns

    # Supprime celles à exclure
    colonnes_utiles = [col for col in colonnes_numeriques if col not in colonnes_exclues]

    if verbose:
        print(f"[INFO] Colonnes numériques conservées : {colonnes_utiles}")

    # Calcul des moyennes pour tri
    moyennes = {col: df[col].mean() for col in colonnes_utiles}
    colonnes_triees = sorted(moyennes, key=moyennes.get)

    # Préparation des données
    data = [df[col].dropna() for col in colonnes_triees]

    # Affichage
    plt.figure(figsize=(16, 8))
    plt.boxplot(data, labels=colonnes_triees, patch_artist=True, vert=False)
    plt.title("Boîtes à moustaches des variables météo")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def NuagePointsTemperature(df):
    # Vérifications préalables
    if 'AAAAMMJJHH' not in df.columns or 'T' not in df.columns:
        raise ValueError("Le DataFrame doit contenir les colonnes 'AAAAMMJJHH' et 'T'.")

    # Filtrage des données valides
    df_plot = df[['AAAAMMJJHH', 'T']].dropna()

    # Conversion de AAAAMMJJHH en datetime si ce n'est pas déjà fait
    if not pd.api.types.is_datetime64_any_dtype(df_plot['AAAAMMJJHH']):
        try:
            df_plot['AAAAMMJJHH'] = pd.to_datetime(df_plot['AAAAMMJJHH'], errors='coerce')
        except Exception as e:
            print(f"[ERREUR] Conversion datetime impossible : {e}")
            return

    # Tracer le nuage de points
    plt.figure(figsize=(14, 6))
    plt.scatter(df_plot['AAAAMMJJHH'], df_plot['T'], s=10, alpha=0.6)
    plt.xlabel("Date")
    plt.ylabel("Température (°C)")
    plt.title("Nuage de points : Température en fonction du temps")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
