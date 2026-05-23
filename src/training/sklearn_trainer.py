"""
Sklearn Trainer — Single responsibility: receives data (already scaled), returns predictions.
NO Evaluation, No Tracking, No Storage
"""

import numpy as np
from sklearn.base import clone
from src.models.base import SklearnModel


def train_and_predict(
    model: SklearnModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Train a sklearn model and return predictions on test data.

    Args:
        model: Any sklearn-compatible estimator that respect the protocol SklearnModel.
        X_train: Scaled training features.
        y_train: Training targets.
        X_test: Scaled test features.

    Returns:
        y_pred: Predictions on X_test.
    """
    model_clone = clone(model)
    model_clone.fit(X_train, y_train)
    return model_clone.predict(X_test)
