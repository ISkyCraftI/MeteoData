from importeur import COULEURS_DEPS

def regression_lineaire(df, explicatives, cible):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import numpy as np

    X = df[explicatives].values
    y = df[cible].values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)

    # Récupère le département (pour couleur et titre)
    dep = df["dep"].iloc[0] if "dep" in df.columns else "Inconnu"
    couleur = COULEURS_DEPS.get(dep, "gray")  # Couleur par défaut : gris
    titre = f"{cible} ~ {' + '.join(explicatives)}"

    print(f"[{dep}] {titre} | R² multiple = {score:.3f}")
    
    # Visualisation 2D par variable explicative
    for i, var in enumerate(explicatives):
        x_var = df[var].values.reshape(-1, 1)
        model_1d = LinearRegression().fit(x_var, y)
        y_pred_1d = model_1d.predict(x_var)
        r2_simple = r2_score(y, y_pred_1d)

        plt.figure(figsize=(8, 5))
        plt.scatter(x_var, y, alpha=0.5, edgecolor='k', color=couleur, label=f"Dép {dep}")
        plt.plot(x_var, y_pred_1d, color='black', linewidth=2, label="Régression simple")
        plt.xlabel(var)
        plt.ylabel(cible)
        plt.title(f"{cible} ~ {var} — Dép {dep}\nR² simple : {r2_simple:.2f} | R² multiple : {score:.2f}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
