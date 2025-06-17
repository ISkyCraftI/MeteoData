import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from importeur import COULEURS_DEPS


def boiteAMoustache(df, verbose=False):
    colonnes_exclues = ['NUM_POSTE', 'AAAAMMJJHH']
    colonnes_numeriques = df.select_dtypes(include=[np.number]).columns
    colonnes_utiles = [col for col in colonnes_numeriques if col not in colonnes_exclues]

    if verbose:
        print(f"[INFO] Colonnes numériques conservées : {colonnes_utiles}")

    moyennes = {col: df[col].mean() for col in colonnes_utiles}
    colonnes_triees = sorted(moyennes, key=moyennes.get)
    data = [df[col].dropna() for col in colonnes_triees]

    plt.boxplot(data, labels=colonnes_triees, patch_artist=True, vert=False)
    plt.title("Boîtes à moustaches des variables météo")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

def courbe_temperature_par_departement(data):
    plt.figure(figsize=(15, 6))
    if "date_jour" not in data.columns:
        data["date_jour"] = data["date"].dt.date

    data_jour = data.groupby(["dep", "date_jour"])["T"].mean().reset_index()

    for dep in data_jour["dep"].unique():
        subset = data_jour[data_jour["dep"] == dep]
        plt.plot(subset["date_jour"], subset["T"], label=f"Dép {dep}", linewidth=0.8,
                 color=COULEURS_DEPS.get(dep, None))

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
        color = COULEURS_DEPS.get(dep, None)
        axs[0].plot(subset["date"], subset["U"], label=f"Dép {dep}", linewidth=0.5, color=color)
        axs[1].plot(subset["date"], subset["FF"], label=f"Dép {dep}", linewidth=0.5, color=color)
        axs[2].plot(subset["date"], subset["P"], label=f"Dép {dep}", linewidth=0.5, color=color)

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
    ordre_deps = sorted(data["dep"].unique(), reverse=True)
    data["dep"] = pd.Categorical(data["dep"], categories=ordre_deps, ordered=True)

    palette = {
        dep: COULEURS_DEPS.get(dep, "gray")
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

    palette = {dep: COULEURS_DEPS.get(dep, "gray") for dep in ordre_deps}

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for i, variable in enumerate(variables):
        sns.boxplot(data=data, x="dep", y=variable, ax=axs[i], palette=palette)
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
        plt.hist(subset["T"], bins=50, alpha=0.5, label=f"Dép {dep}",
                 color=COULEURS_DEPS.get(dep, None))
    ticks = plt.xticks()[0]
    plt.xticks(ticks, [f"{tick:.0f}" for tick in ticks])
    plt.legend()
    plt.title("Histogramme des températures")
    plt.xlabel("Température (°C)")
    plt.ylabel("Fréquence")
    plt.show()

def hist_variable(data):
    variables = ["U", "FF", "P"]
    titres = ["Humidité (%)", "Vent moyen (m/s)", "Pression (hPa)"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for i, variable in enumerate(variables):
        ax = axs[i]
        for dep in data["dep"].unique():
            subset = data[data["dep"] == dep]
            ax.hist(subset[variable], bins=50, alpha=0.5, label=f"Dép {dep}",
                    color=COULEURS_DEPS.get(dep, None))
        ax.set_title(titres[i])
        ax.set_xlabel(variable)
        ax.set_ylabel("Fréquence")
        ax.legend()

    plt.tight_layout()
    plt.show()

def courbe_moyenne_par_mois(df: pd.DataFrame, colonne: str = "RR1", label=None, group_by_dep=True):
    if colonne not in df.columns or "date" not in df.columns:
        print("Colonnes manquantes : il faut 'date' et", colonne)
        return

    df = df.copy()
    df[colonne] = pd.to_numeric(df[colonne], errors="coerce")
    df = df.dropna(subset=[colonne, "date"])
    df["mois"] = df["date"].dt.month

    plt.figure(figsize=(10, 5))
    if group_by_dep and "dep" in df.columns:
        for dep in df["dep"].unique():
            subset = df[df["dep"] == dep]
            moyennes = subset.groupby("mois")[colonne].mean()
            mois_labels = [calendar.month_name[m] for m in moyennes.index]
            plt.plot(moyennes.index, moyennes.values, marker='o',
                     label=f"Dép {dep}", color=COULEURS_DEPS.get(dep, None))
    else:
        moyennes = df.groupby("mois")[colonne].mean()
        mois_labels = [calendar.month_name[m] for m in moyennes.index]
        plt.plot(moyennes.index, moyennes.values, marker='o', label=label)

    plt.xticks(moyennes.index, mois_labels, rotation=45)
    plt.xlabel("Mois")
    plt.ylabel(f"Moyenne de {colonne}")
    plt.title(f"Moyenne mensuelle de {colonne}")
    if label or group_by_dep:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
