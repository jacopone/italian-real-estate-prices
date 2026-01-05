"""Base classes for models.

Defines the interface that all models must implement, providing
consistency across different model types (OLS, Gradient Boosting, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class ModelResult:
    """Container for model evaluation results.

    Attributes:
        model_name: Name/identifier for the model.
        r_squared: R² score (coefficient of determination).
        rmse: Root mean squared error.
        mae: Mean absolute error.
        feature_importance: Feature importance scores.
        predictions: Model predictions on evaluation set.
        residuals: Prediction residuals.
        metadata: Additional model-specific metrics.
    """

    model_name: str
    r_squared: float
    rmse: float
    mae: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    predictions: pd.Series | None = None
    residuals: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "mae": self.mae,
            "feature_importance": self.feature_importance,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"ModelResult({self.model_name}: "
            f"R²={self.r_squared:.4f}, RMSE={self.rmse:.4f}, MAE={self.mae:.4f})"
        )


class BaseModel(ABC):
    """Abstract base class for all models.

    Provides consistent interface for training, prediction, and evaluation.
    All models must implement fit(), predict(), and get_feature_importance().

    Example:
        >>> model = GradientBoostingModel(params)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> result = model.evaluate(X_test, y_test)
    """

    def __init__(self, name: str | None = None):
        """Initialize model.

        Args:
            name: Model name/identifier.
        """
        self._name = name or self.__class__.__name__
        self._is_fitted = False
        self._feature_names: list[str] = []

    @property
    def name(self) -> str:
        """Get model name."""
        return self._name

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted

    @property
    def feature_names(self) -> list[str]:
        """Get feature names used in training."""
        return self._feature_names.copy()

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "BaseModel":
        """Train the model.

        Args:
            X: Feature matrix.
            y: Target values.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.

        Raises:
            ValueError: If model has not been fitted.
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        pass

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> ModelResult:
        """Evaluate model on test data.

        Args:
            X: Feature matrix.
            y: True target values.

        Returns:
            ModelResult with metrics and diagnostics.
        """
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        self._check_fitted()

        predictions = self.predict(X)
        residuals = y - predictions

        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)

        return ModelResult(
            model_name=self.name,
            r_squared=r2,
            rmse=rmse,
            mae=mae,
            feature_importance=self.get_feature_importance(),
            predictions=predictions,
            residuals=residuals,
        )

    def _check_fitted(self) -> None:
        """Raise error if model not fitted."""
        if not self._is_fitted:
            raise ValueError(
                f"{self.name} has not been fitted. Call fit() first."
            )

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model.
        """
        import joblib

        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved {self.name} to {path}")

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from disk.

        Args:
            path: Path to saved model.

        Returns:
            Loaded model instance.
        """
        import joblib

        model = joblib.load(path)
        logger.info(f"Loaded {model.name} from {path}")
        return model


class EnsembleModel(BaseModel):
    """Base class for ensemble models (bagging, boosting).

    Adds ensemble-specific functionality like:
    - Tree-based feature importance
    - Out-of-bag predictions
    - Staged predictions for boosting
    """

    @abstractmethod
    def get_tree_depths(self) -> list[int]:
        """Get depths of individual trees."""
        pass

    def get_feature_importance_gain(self) -> dict[str, float]:
        """Get feature importance based on gain (for tree-based models).

        Override in subclass if model supports gain-based importance.
        """
        return self.get_feature_importance()


class LinearModel(BaseModel):
    """Base class for linear models (OLS, Ridge, Lasso).

    Adds linear model-specific functionality like:
    - Coefficient access
    - Statistical significance
    - Standardized coefficients
    """

    @abstractmethod
    def get_coefficients(self) -> dict[str, float]:
        """Get model coefficients.

        Returns:
            Dictionary mapping feature names to coefficients.
        """
        pass

    def get_feature_importance(self) -> dict[str, float]:
        """Feature importance as absolute coefficient values.

        For linear models, importance is the absolute value of coefficients.
        """
        coeffs = self.get_coefficients()
        return {k: abs(v) for k, v in coeffs.items()}


@dataclass
class CrossValidationResult:
    """Results from cross-validation.

    Attributes:
        model_name: Name of the model.
        cv_scores: R² scores for each fold.
        mean_score: Mean R² across folds.
        std_score: Standard deviation of R² across folds.
        fold_results: Detailed results per fold.
    """

    model_name: str
    cv_scores: list[float]
    mean_score: float
    std_score: float
    fold_results: list[ModelResult] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"CrossValidationResult({self.model_name}: "
            f"R²={self.mean_score:.4f} ± {self.std_score:.4f})"
        )


def cross_validate(
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> CrossValidationResult:
    """Perform cross-validation on a model.

    Args:
        model: Model to evaluate.
        X: Feature matrix.
        y: Target values.
        cv: Number of folds.
        random_state: Random seed for reproducibility.

    Returns:
        CrossValidationResult with scores and metrics.
    """
    import numpy as np
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Clone model for this fold
        model_copy = model.__class__.__name__
        # Note: In practice, you'd clone the model properly
        model.fit(X_train, y_train)
        result = model.evaluate(X_val, y_val)

        scores.append(result.r_squared)
        fold_results.append(result)

        logger.debug(f"Fold {fold_idx + 1}: R²={result.r_squared:.4f}")

    return CrossValidationResult(
        model_name=model.name,
        cv_scores=scores,
        mean_score=np.mean(scores),
        std_score=np.std(scores),
        fold_results=fold_results,
    )
