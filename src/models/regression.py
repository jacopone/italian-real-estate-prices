"""Linear regression models.

Implements OLS and regularized regression (Ridge, Lasso) for
baseline price/rent prediction.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from src.models.base import LinearModel, ModelResult


class OLSModel(LinearModel):
    """Ordinary Least Squares regression model.

    A simple baseline model that's interpretable and fast.
    Useful for understanding linear relationships and
    as a baseline to compare against more complex models.

    Example:
        >>> model = OLSModel()
        >>> model.fit(X_train, y_train)
        >>> print(model.get_coefficients())
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        name: str | None = None,
    ):
        """Initialize OLS model.

        Args:
            fit_intercept: Whether to fit intercept.
            name: Model name.
        """
        super().__init__(name or "OLS")
        self._fit_intercept = fit_intercept
        self._model: LinearRegression | None = None
        self._intercept: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OLSModel":
        """Fit OLS model.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self.
        """
        self._feature_names = list(X.columns)

        self._model = LinearRegression(fit_intercept=self._fit_intercept)
        self._model.fit(X, y)

        self._intercept = self._model.intercept_
        self._is_fitted = True

        logger.info(
            f"Fitted {self.name} with {len(self._feature_names)} features, "
            f"intercept={self._intercept:.4f}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predictions.
        """
        self._check_fitted()
        predictions = self._model.predict(X)
        return pd.Series(predictions, index=X.index, name="prediction")

    def get_coefficients(self) -> dict[str, float]:
        """Get model coefficients.

        Returns:
            Feature name to coefficient mapping.
        """
        self._check_fitted()
        return dict(zip(self._feature_names, self._model.coef_))

    def get_intercept(self) -> float:
        """Get model intercept."""
        self._check_fitted()
        return self._intercept


class RidgeModel(LinearModel):
    """Ridge regression (L2 regularization).

    Prevents overfitting by penalizing large coefficients.
    Useful when features are correlated.

    Example:
        >>> model = RidgeModel(alpha=1.0)
        >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        name: str | None = None,
    ):
        """Initialize Ridge model.

        Args:
            alpha: Regularization strength.
            fit_intercept: Whether to fit intercept.
            name: Model name.
        """
        super().__init__(name or f"Ridge(α={alpha})")
        self._alpha = alpha
        self._fit_intercept = fit_intercept
        self._model: Ridge | None = None
        self._intercept: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        """Fit Ridge model.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self.
        """
        self._feature_names = list(X.columns)

        self._model = Ridge(
            alpha=self._alpha,
            fit_intercept=self._fit_intercept,
        )
        self._model.fit(X, y)

        self._intercept = self._model.intercept_
        self._is_fitted = True

        logger.info(
            f"Fitted {self.name} with {len(self._feature_names)} features"
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        self._check_fitted()
        predictions = self._model.predict(X)
        return pd.Series(predictions, index=X.index, name="prediction")

    def get_coefficients(self) -> dict[str, float]:
        """Get model coefficients."""
        self._check_fitted()
        return dict(zip(self._feature_names, self._model.coef_))


class LassoModel(LinearModel):
    """Lasso regression (L1 regularization).

    Performs feature selection by driving some coefficients to zero.
    Useful for sparse feature sets.

    Example:
        >>> model = LassoModel(alpha=0.1)
        >>> model.fit(X_train, y_train)
        >>> # Check which features were selected
        >>> print({k: v for k, v in model.get_coefficients().items() if v != 0})
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        name: str | None = None,
    ):
        """Initialize Lasso model.

        Args:
            alpha: Regularization strength.
            fit_intercept: Whether to fit intercept.
            max_iter: Maximum iterations.
            name: Model name.
        """
        super().__init__(name or f"Lasso(α={alpha})")
        self._alpha = alpha
        self._fit_intercept = fit_intercept
        self._max_iter = max_iter
        self._model: Lasso | None = None
        self._intercept: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoModel":
        """Fit Lasso model."""
        self._feature_names = list(X.columns)

        self._model = Lasso(
            alpha=self._alpha,
            fit_intercept=self._fit_intercept,
            max_iter=self._max_iter,
        )
        self._model.fit(X, y)

        self._intercept = self._model.intercept_
        self._is_fitted = True

        # Count non-zero coefficients
        n_selected = np.sum(np.abs(self._model.coef_) > 1e-10)
        logger.info(
            f"Fitted {self.name}: {n_selected}/{len(self._feature_names)} features selected"
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        self._check_fitted()
        predictions = self._model.predict(X)
        return pd.Series(predictions, index=X.index, name="prediction")

    def get_coefficients(self) -> dict[str, float]:
        """Get model coefficients."""
        self._check_fitted()
        return dict(zip(self._feature_names, self._model.coef_))

    def get_selected_features(self) -> list[str]:
        """Get features with non-zero coefficients.

        Returns:
            List of selected feature names.
        """
        self._check_fitted()
        coeffs = self.get_coefficients()
        return [k for k, v in coeffs.items() if abs(v) > 1e-10]


def create_regression_model(
    model_type: str = "ols",
    **kwargs: Any,
) -> LinearModel:
    """Factory function to create regression models.

    Args:
        model_type: One of 'ols', 'ridge', 'lasso'.
        **kwargs: Model-specific parameters.

    Returns:
        Configured regression model.

    Example:
        >>> model = create_regression_model('ridge', alpha=0.5)
    """
    models = {
        "ols": OLSModel,
        "ridge": RidgeModel,
        "lasso": LassoModel,
    }

    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type.lower()](**kwargs)
