"""Ensemble models for price/rent prediction.

Implements Gradient Boosting and Random Forest models that typically
achieve higher accuracy than linear models at the cost of interpretability.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.config import ModelConfig
from src.models.base import EnsembleModel, ModelResult


class GradientBoostingModel(EnsembleModel):
    """Gradient Boosting regression model.

    This is the primary model for price prediction, achieving R² > 0.80.
    Gradient Boosting builds trees sequentially, each correcting
    errors of the previous ones.

    Key hyperparameters (tuned values):
    - n_estimators: 500 (number of boosting stages)
    - learning_rate: 0.05 (shrinkage, lower = more regularization)
    - max_depth: 6 (individual tree complexity)
    - min_samples_leaf: 10 (prevents overfitting to outliers)
    - subsample: 0.8 (stochastic gradient boosting)

    Example:
        >>> model = GradientBoostingModel()
        >>> model.fit(X_train, y_train)
        >>> result = model.evaluate(X_test, y_test)
        >>> print(f"R² = {result.r_squared:.4f}")
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        subsample: float = 0.8,
        max_features: str = "sqrt",
        random_state: int = 42,
        name: str | None = None,
    ):
        """Initialize Gradient Boosting model.

        Args:
            n_estimators: Number of boosting stages.
            learning_rate: Learning rate shrinks contribution of each tree.
            max_depth: Maximum depth of individual trees.
            min_samples_split: Minimum samples to split internal node.
            min_samples_leaf: Minimum samples at leaf node.
            subsample: Fraction of samples used for each tree.
            max_features: Features to consider for best split.
            random_state: Random seed.
            name: Model name.
        """
        super().__init__(name or "GradientBoosting")

        self._params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "subsample": subsample,
            "max_features": max_features,
            "random_state": random_state,
        }
        self._model: GradientBoostingRegressor | None = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> "GradientBoostingModel":
        """Create model from configuration.

        Args:
            config: Model configuration.

        Returns:
            Configured GradientBoostingModel.
        """
        return cls(**config.gb_params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        """Fit Gradient Boosting model.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self.
        """
        self._feature_names = list(X.columns)

        self._model = GradientBoostingRegressor(**self._params)
        self._model.fit(X, y)
        self._is_fitted = True

        # Log training info
        train_score = self._model.score(X, y)
        logger.info(
            f"Fitted {self.name} with {len(self._feature_names)} features, "
            f"train R²={train_score:.4f}"
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

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns impurity-based importance (higher = more important).

        Returns:
            Feature name to importance mapping.
        """
        self._check_fitted()
        importance = self._model.feature_importances_
        return dict(zip(self._feature_names, importance))

    def get_tree_depths(self) -> list[int]:
        """Get depths of all trees in ensemble.

        Returns:
            List of tree depths.
        """
        self._check_fitted()
        return [
            tree[0].get_depth()
            for tree in self._model.estimators_
        ]

    def get_staged_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get predictions at each boosting stage.

        Useful for analyzing learning curves.

        Args:
            X: Feature matrix.

        Returns:
            DataFrame with predictions at each stage.
        """
        self._check_fitted()

        staged = list(self._model.staged_predict(X))
        return pd.DataFrame(
            staged,
            columns=X.index,
            index=range(1, len(staged) + 1),
        ).T

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features.

        Args:
            n: Number of features to return.

        Returns:
            List of (feature_name, importance) tuples.
        """
        importance = self.get_feature_importance()
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:n]


class RandomForestModel(EnsembleModel):
    """Random Forest regression model.

    Builds multiple decision trees with random subsets of features
    and samples, then averages predictions. More robust to overfitting
    than single trees, and provides uncertainty estimates.

    Example:
        >>> model = RandomForestModel(n_estimators=200)
        >>> model.fit(X_train, y_train)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
        name: str | None = None,
    ):
        """Initialize Random Forest model.

        Args:
            n_estimators: Number of trees.
            max_depth: Maximum tree depth (None = unlimited).
            min_samples_split: Minimum samples to split.
            min_samples_leaf: Minimum samples at leaf.
            max_features: Features to consider for split.
            n_jobs: Parallel jobs (-1 = all cores).
            random_state: Random seed.
            name: Model name.
        """
        super().__init__(name or "RandomForest")

        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        self._model: RandomForestRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """Fit Random Forest model."""
        self._feature_names = list(X.columns)

        self._model = RandomForestRegressor(**self._params)
        self._model.fit(X, y)
        self._is_fitted = True

        train_score = self._model.score(X, y)
        logger.info(
            f"Fitted {self.name} with {len(self._feature_names)} features, "
            f"train R²={train_score:.4f}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions."""
        self._check_fitted()
        predictions = self._model.predict(X)
        return pd.Series(predictions, index=X.index, name="prediction")

    def predict_with_std(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Make predictions with uncertainty estimates.

        Returns mean and std of predictions across trees.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (mean_predictions, std_predictions).
        """
        self._check_fitted()

        # Get predictions from each tree
        all_preds = np.array([
            tree.predict(X) for tree in self._model.estimators_
        ])

        mean_preds = pd.Series(all_preds.mean(axis=0), index=X.index)
        std_preds = pd.Series(all_preds.std(axis=0), index=X.index)

        return mean_preds, std_preds

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        self._check_fitted()
        importance = self._model.feature_importances_
        return dict(zip(self._feature_names, importance))

    def get_tree_depths(self) -> list[int]:
        """Get depths of all trees."""
        self._check_fitted()
        return [tree.get_depth() for tree in self._model.estimators_]

    def get_oob_score(self) -> float | None:
        """Get out-of-bag score if available.

        Returns:
            OOB R² score or None if not available.
        """
        self._check_fitted()
        if hasattr(self._model, "oob_score_"):
            return self._model.oob_score_
        return None


def create_ensemble_model(
    model_type: str = "gradient_boosting",
    **kwargs: Any,
) -> EnsembleModel:
    """Factory function to create ensemble models.

    Args:
        model_type: One of 'gradient_boosting', 'random_forest'.
        **kwargs: Model-specific parameters.

    Returns:
        Configured ensemble model.

    Example:
        >>> model = create_ensemble_model('gradient_boosting', n_estimators=300)
    """
    models = {
        "gradient_boosting": GradientBoostingModel,
        "gb": GradientBoostingModel,
        "random_forest": RandomForestModel,
        "rf": RandomForestModel,
    }

    if model_type.lower() not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type.lower()](**kwargs)
