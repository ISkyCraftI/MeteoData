import matplotlib.pyplot as plt
import numpy as np

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

def NuagePointsTemperature(df):
    col = ['T']