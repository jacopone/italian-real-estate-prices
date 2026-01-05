"""Model implementations for price and rent prediction.

Provides both linear (OLS, Ridge) and ensemble (Gradient Boosting, Random Forest)
models with a consistent interface for training and evaluation.
"""

from src.models.base import (
    BaseModel,
    CrossValidationResult,
    EnsembleModel,
    LinearModel,
    ModelResult,
    cross_validate,
)
from src.models.ensemble import (
    GradientBoostingModel,
    RandomForestModel,
    create_ensemble_model,
)
from src.models.regression import (
    LassoModel,
    OLSModel,
    RidgeModel,
    create_regression_model,
)
from src.models.training import (
    ModelTrainer,
    PriceRentTrainer,
    TrainingResult,
)

__all__ = [
    # Base
    "BaseModel",
    "LinearModel",
    "EnsembleModel",
    "ModelResult",
    "CrossValidationResult",
    "cross_validate",
    # Regression
    "OLSModel",
    "RidgeModel",
    "LassoModel",
    "create_regression_model",
    # Ensemble
    "GradientBoostingModel",
    "RandomForestModel",
    "create_ensemble_model",
    # Training
    "ModelTrainer",
    "PriceRentTrainer",
    "TrainingResult",
]
