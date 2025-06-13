def extraction_infos_series(serie):
    return {
        "Nom": serie.name,
        "Min": serie.min(),
        "Max": serie.max(),
        "Moyenne": serie.mean(),
        "Écart-type": serie.std(),
        "Médiane": serie.median(),
        "Nb valeurs": serie.count()
    }

def statistiques(df):
    return df.describe().loc[["mean", "50%", "std"]].rename(index={"50%": "median"})
