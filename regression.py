

from sklearn.linear_model import *
from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt

def regression_lineaire(df, explicatives, cible):

    print(f"\n Régression : {cible} ~ {', '.join(explicatives)} ")

    X = df[explicatives].values  # (n, p)
    y = df[cible].values         # (n, )

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Coefficients et intercept
    coeffs = model.coef_
    intercept = model.intercept_

    # Affichage de la fonction f(x)
    terme_lineaire = " + ".join([f"{omega:.4f}·{var}" for omega, var in zip(coeffs, explicatives)])
    print(f"\nForme de la fonction : f(x) = {terme_lineaire} + {intercept:.4f}")

    # Calculs des erreurs
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    rss = np.sum((y - y_pred) ** 2)
    r2 = r2_score(y, y_pred)

    print(f"\nÉvaluation des erreurs :")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  RSS  : {rss:.4f}")
    print(f"  R²   : {r2:.4f}")

    # Visualisation si simple régression
    if len(explicatives) == 1:
        plt.figure(figsize=(6, 4))
        plt.scatter(X, y, alpha=0.3, label='Données réelles')
        plt.plot(X, y_pred, color='red', label='Régression')
        plt.xlabel(explicatives[0])
        plt.ylabel(cible)
        plt.title(f"Régression : {cible} ~ {explicatives[0]}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        # Affichage y réel vs prédit pour cas multiple
        plt.figure(figsize=(6, 4))
        plt.scatter(y, y_pred, alpha=0.3)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel("Valeurs réelles")
        plt.ylabel("Valeurs prédites")
        plt.title("Régression multiple : Réel vs Prédit")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        