# Boite à moustaches
from matplotlib import pyplot as plt


def boiteAMoustache(df):
    colonnes_ue = df.columns[2:]
  
    moyennes = {col: df[col].mean() for col in colonnes_ue}     # Calcul des moyennes

    colonnes_ue_triees = sorted(moyennes, key=moyennes.get)     # Tri des colonnes par moyenne croissante

    data = [df[col].dropna() for col in colonnes_ue_triees]     # Préparer les données dans l'ordre trié

    # Tracé du graphique
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=colonnes_ue_triees, patch_artist=True)
    plt.title("Boîtes à moustaches des notes par UE (triées par moyenne croissante)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()