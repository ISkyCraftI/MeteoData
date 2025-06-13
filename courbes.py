import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    # plt.figure(figsize=(4, 2))
    plt.boxplot(data, labels=colonnes_triees, patch_artist=True, vert=False)
    plt.title("Boîtes à moustaches des variables météo")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()



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


def courbe_temperature_par_departement(data):
    plt.figure(figsize=(12, 6))
    for dep in data["dep"].unique():
        subset = data[data["dep"] == dep]
        plt.plot(subset["date"], subset["T"], label=f"Dép {dep}", linewidth=0.5)
    plt.legend()
    plt.title("Température (°C) par département")
    plt.xlabel("Date")
    plt.ylabel("Température")
    plt.tight_layout()
    plt.show()


def courbes_variables(data):
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    for dep in data["dep"].unique():
        subset = data[data["dep"] == dep]
        axs[0].plot(subset["date"], subset["U"], label=f"Dép {dep}", linewidth=0.5)
        axs[1].plot(subset["date"], subset["FF"], label=f"Dép {dep}", linewidth=0.5)
        axs[2].plot(subset["date"], subset["P"], label=f"Dép {dep}", linewidth=0.5)

    axs[0].set_title("Humidité (%)")
    axs[1].set_title("Vent moyen (m/s)")
    axs[2].set_title("Pression (hPa)")
    axs[2].set_xlabel("Date")
    for ax in axs:
        ax.legend()
        ax.set_ylabel("Valeur")
    plt.tight_layout()
    plt.show()


def boxplot_temperature(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x="dep", y="T")
    plt.title("Répartition des températures par département")
    plt.show()


def hist_temperature(data):
    plt.figure(figsize=(12, 5))
    for dep in data["dep"].unique():
        subset = data[data["dep"] == dep]
        plt.hist(subset["T"], bins=50, alpha=0.5, label=f"Dép {dep}")
    plt.legend()
    plt.title("Histogramme des températures")
    plt.xlabel("Température (°C)")
    plt.ylabel("Fréquence")
    plt.show()
