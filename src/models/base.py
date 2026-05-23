"""
Model Protocol — Contractul oficial pentru orice model din pipeline-ul de ML.

De ce Protocol si nu ABC (Clasa Abstracta)?
-------------------------------------------
ABC (Abstract Base Class) impune MOSTENIRE EXPLICITA:
    class RidgeWrapper(BaseForecaster): ...  ← obligatoriu

Protocol impune COMPORTAMENT (Duck Typing / Structural Typing):
    Orice clasa care are .fit() si .predict() este automat compatibila.
    Ridge(), XGBRegressor(), LGBMRegressor() — toate sunt compatibile
    FARA sa mosteneasca nimic, FARA wrappere inutile.

Principiul: "Programming to an interface, not an implementation"
"""

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class SklearnModel(Protocol):
    """
    Contractul oficial (priza universala) pentru orice model sklearn-compatible.

    Orice obiect care implementeaza aceste doua metode va fi acceptat
    automat de Trainer si Runner, indiferent de libraria din care provine.

    Modele compatibile (fara niciun wrapper):
        - sklearn: Ridge, LinearRegression, Lasso, ElasticNet
        - xgboost: XGBRegressor
        - lightgbm: LGBMRegressor
        - catboost: CatBoostRegressor
        - orice model custom care respecta fit/predict cu numpy arrays
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnModel":
        """
        Antreneaza modelul pe datele de training.

        Args:
            X: Feature matrix [n_samples, n_features] 
            y: Target vector [n_samples] 

        Returns:
            self 
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genereaza predictii pentru datele de test.

        Args:
            X: Feature matrix [n_samples, n_features]

        Returns:
            y_pred: Array de predictii [n_samples]
        """
        ...
