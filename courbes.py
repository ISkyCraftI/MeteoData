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


def courbe_temperature_par_departement(data):
    plt.figure(figsize=(15, 6))

    # Ajout de la colonne date_jour si elle n'existe pas déjà
    if "date_jour" not in data.columns:
        data["date_jour"] = data["date"].dt.date

    # Moyenne journalière par département
    data_jour = data.groupby(["dep", "date_jour"])["T"].mean().reset_index()

    for dep in data_jour["dep"].unique():
        subset = data_jour[data_jour["dep"] == dep]
        plt.plot(subset["date_jour"], subset["T"], label=f"Dép {dep}", linewidth=0.8)

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


def boxplot_temperature(data, departement_highlight="29"):
    plt.figure(figsize=(10, 6))

    # Ordre inversé des départements (affiché de droite à gauche)
    ordre_deps = sorted(data["dep"].unique(), reverse=True)
    data["dep"] = pd.Categorical(data["dep"], categories=ordre_deps, ordered=True)

    # Palette personnalisée
    palette = {
        dep: ("crimson" if dep == departement_highlight else "skyblue")
        for dep in ordre_deps
    }

    sns.boxplot(data=data, x="dep", y="T", palette=palette)

    plt.title("Répartition des températures par département")
    plt.xlabel("Département")
    plt.ylabel("Température (°C)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
def boxplot_variable(data):
    variables = ["U", "FF", "P"]
    titres = ["Humidité (%)", "Vent moyen (m/s)", "Pression (hPa)"]

    ordre_deps = sorted(data["dep"].unique(), reverse=True)
    data["dep"] = pd.Categorical(data["dep"], categories=ordre_deps, ordered=True)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for i, variable in enumerate(variables):
        sns.boxplot(data=data, x="dep", y=variable, ax=axs[i], palette="pastel")
        axs[i].set_title(titres[i])
        axs[i].set_xlabel("Département")
        axs[i].set_ylabel(variable)
        axs[i].tick_params(axis='x', rotation=90)
        axs[i].grid(True, axis='y')

    plt.tight_layout()
    plt.show()



def hist_temperature(data):
    plt.figure(figsize=(12, 5))
    for dep in data["dep"].unique():
        subset = data[data["dep"] == dep]
        plt.hist(subset["T"], bins=50, alpha=0.5, label=f"Dép {dep}")
    ticks = plt.xticks()[0]
    plt.xticks(ticks, [f"{tick:.0f}" for tick in ticks]) 
    plt.legend()
    plt.title("Histogramme des températures")
    plt.xlabel("Température (°C)")
    plt.ylabel("Fréquence")
    plt.show()


import matplotlib.pyplot as plt
import calendar
import pandas as pd

def courbe_moyenne_par_mois(df: pd.DataFrame, colonne: str = "RR1", label=None):
    if colonne not in df.columns or "date" not in df.columns:
        print("Colonnes manquantes : il faut 'date' et", colonne)
        return

    # Copie et nettoyage
    df = df.copy()
    df[colonne] = pd.to_numeric(df[colonne], errors="coerce")
    df = df.dropna(subset=[colonne, "date"])

    # Extraire le mois
    df["mois"] = df["date"].dt.month

    # Moyenne mensuelle
    moyennes = df.groupby("mois")[colonne].mean()

    mois_labels = [calendar.month_name[m] for m in moyennes.index]

    # Tracé
    plt.figure(figsize=(10, 5))
    plt.plot(moyennes.index, moyennes.values, marker='o', label=label or f"{colonne} par mois")
    plt.xticks(moyennes.index, mois_labels, rotation=45)
    plt.xlabel("Mois")
    plt.ylabel(f"Moyenne de {colonne}")
    plt.title(f"Moyenne mensuelle de {colonne}")
    if label:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()