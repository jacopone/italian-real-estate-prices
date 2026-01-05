"""Model training orchestration.

Provides high-level training workflows that combine data preparation,
model fitting, evaluation, and result tracking.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.config import Config, ModelConfig
from src.models.base import BaseModel, ModelResult, cross_validate
from src.models.ensemble import GradientBoostingModel
from src.models.regression import OLSModel, RidgeModel


@dataclass
class TrainingResult:
    """Complete results from a training run.

    Attributes:
        model: Trained model.
        train_result: Evaluation on training set.
        test_result: Evaluation on test set.
        cv_result: Cross-validation results (if performed).
        feature_names: Features used.
        data_info: Information about the training data.
    """

    model: BaseModel
    train_result: ModelResult
    test_result: ModelResult
    cv_result: Any | None = None
    feature_names: list[str] = field(default_factory=list)
    data_info: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Get training summary."""
        lines = [
            f"Model: {self.model.name}",
            f"Features: {len(self.feature_names)}",
            f"Train R²: {self.train_result.r_squared:.4f}",
            f"Test R²: {self.test_result.r_squared:.4f}",
            f"Test RMSE: {self.test_result.rmse:.4f}",
        ]
        if self.cv_result:
            lines.append(
                f"CV R²: {self.cv_result.mean_score:.4f} ± {self.cv_result.std_score:.4f}"
            )
        return "\n".join(lines)


class ModelTrainer:
    """Orchestrates model training workflow.

    Handles:
    - Train/test splitting
    - Feature selection
    - Model training
    - Evaluation
    - Result tracking

    Example:
        >>> trainer = ModelTrainer(config.model)
        >>> result = trainer.train(
        ...     features_df,
        ...     target='log_price_mid',
        ...     model_type='gradient_boosting',
        ... )
        >>> print(result.summary())
    """

    def __init__(self, config: ModelConfig | None = None):
        """Initialize trainer.

        Args:
            config: Model configuration.
        """
        self.config = config or ModelConfig()
        self._trained_models: dict[str, TrainingResult] = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        target: str,
        feature_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training.

        Args:
            df: Full DataFrame with features and target.
            target: Target column name.
            feature_cols: Specific feature columns (uses all numeric if None).
            exclude_cols: Columns to exclude from features.

        Returns:
            Tuple of (X, y).
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")

        # Determine feature columns
        if feature_cols is None:
            # Use all numeric columns except target and identifiers
            exclude = {target, "istat_code", "anno"}
            if exclude_cols:
                exclude.update(exclude_cols)

            feature_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude
            ]

        # Drop rows with missing values in features or target
        subset = df[feature_cols + [target]].dropna()

        X = subset[feature_cols]
        y = subset[target]

        logger.info(
            f"Prepared data: {len(X):,} samples, {len(feature_cols)} features, "
            f"target='{target}'"
        )

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float | None = None,
        random_state: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.

        Args:
            X: Feature matrix.
            y: Target values.
            test_size: Fraction for test set (uses config if None).
            random_state: Random seed (uses config if None).

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        test_size = test_size or self.config.test_size
        random_state = random_state or self.config.random_state

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )

        logger.info(
            f"Split data: {len(X_train):,} train, {len(X_test):,} test "
            f"({test_size*100:.0f}% test)"
        )

        return X_train, X_test, y_train, y_test

    def create_model(
        self,
        model_type: str = "gradient_boosting",
        **kwargs: Any,
    ) -> BaseModel:
        """Create a model instance.

        Args:
            model_type: Type of model to create.
            **kwargs: Model-specific parameters.

        Returns:
            Model instance.
        """
        if model_type == "gradient_boosting":
            # Use config params if not overridden
            params = self.config.gb_params.copy()
            params.update(kwargs)
            return GradientBoostingModel(**params)

        elif model_type == "ols":
            return OLSModel(**kwargs)

        elif model_type == "ridge":
            alpha = kwargs.pop("alpha", self.config.ols_alpha)
            return RidgeModel(alpha=alpha, **kwargs)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(
        self,
        df: pd.DataFrame,
        target: str,
        model_type: str = "gradient_boosting",
        feature_cols: list[str] | None = None,
        perform_cv: bool = True,
        **model_kwargs: Any,
    ) -> TrainingResult:
        """Full training workflow.

        Args:
            df: DataFrame with features and target.
            target: Target column name.
            model_type: Type of model.
            feature_cols: Specific features to use.
            perform_cv: Whether to perform cross-validation.
            **model_kwargs: Model-specific parameters.

        Returns:
            TrainingResult with trained model and metrics.
        """
        # Prepare data
        X, y = self.prepare_data(df, target, feature_cols)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Create and train model
        model = self.create_model(model_type, **model_kwargs)
        model.fit(X_train, y_train)

        # Evaluate
        train_result = model.evaluate(X_train, y_train)
        test_result = model.evaluate(X_test, y_test)

        logger.info(
            f"Training complete: Train R²={train_result.r_squared:.4f}, "
            f"Test R²={test_result.r_squared:.4f}"
        )

        # Cross-validation if requested
        cv_result = None
        if perform_cv:
            # Refit on full training data for CV
            model_for_cv = self.create_model(model_type, **model_kwargs)
            cv_result = cross_validate(
                model_for_cv,
                X_train,
                y_train,
                cv=self.config.cv_folds,
                random_state=self.config.random_state,
            )
            logger.info(
                f"Cross-validation: R²={cv_result.mean_score:.4f} ± "
                f"{cv_result.std_score:.4f}"
            )

        result = TrainingResult(
            model=model,
            train_result=train_result,
            test_result=test_result,
            cv_result=cv_result,
            feature_names=list(X.columns),
            data_info={
                "n_samples": len(X),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "target": target,
            },
        )

        # Store result
        self._trained_models[model.name] = result

        return result

    def compare_models(
        self,
        df: pd.DataFrame,
        target: str,
        model_types: list[str] | None = None,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compare multiple model types.

        Args:
            df: DataFrame with features and target.
            target: Target column name.
            model_types: Models to compare (default: OLS, Ridge, GB).
            feature_cols: Features to use.

        Returns:
            DataFrame comparing model performance.
        """
        if model_types is None:
            model_types = ["ols", "ridge", "gradient_boosting"]

        results = []
        for model_type in model_types:
            logger.info(f"Training {model_type}...")
            result = self.train(
                df, target,
                model_type=model_type,
                feature_cols=feature_cols,
                perform_cv=True,
            )
            results.append({
                "model": model_type,
                "train_r2": result.train_result.r_squared,
                "test_r2": result.test_result.r_squared,
                "test_rmse": result.test_result.rmse,
                "cv_mean": result.cv_result.mean_score if result.cv_result else None,
                "cv_std": result.cv_result.std_score if result.cv_result else None,
            })

        return pd.DataFrame(results)

    def get_trained_model(self, name: str) -> TrainingResult | None:
        """Get a previously trained model by name.

        Args:
            name: Model name.

        Returns:
            TrainingResult or None if not found.
        """
        return self._trained_models.get(name)


class PriceRentTrainer:
    """Specialized trainer for price and rent models.

    Handles the specific workflow for training both price and rent
    prediction models with optional STR features.

    Example:
        >>> trainer = PriceRentTrainer(config)
        >>> price_result = trainer.train_price_model(features_df)
        >>> rent_result = trainer.train_rent_model(features_df)
    """

    def __init__(self, config: Config | None = None):
        """Initialize trainer.

        Args:
            config: Full configuration.
        """
        self.config = config or Config()
        self._trainer = ModelTrainer(self.config.model)
        self._price_result: TrainingResult | None = None
        self._rent_result: TrainingResult | None = None

    def train_price_model(
        self,
        df: pd.DataFrame,
        include_str: bool = True,
    ) -> TrainingResult:
        """Train price prediction model.

        Args:
            df: Features DataFrame.
            include_str: Whether to include STR features.

        Returns:
            TrainingResult for price model.
        """
        target = self.config.model.price_target

        # Build feature list
        feature_cols = None  # Use all available
        if not include_str:
            # Exclude STR features
            exclude = {"str_density", "log_str_density", "str_proxy", "log_str_proxy"}
            feature_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude and col != target
            ]

        self._price_result = self._trainer.train(
            df,
            target=target,
            model_type="gradient_boosting",
            feature_cols=feature_cols,
            name=f"PriceModel_{'with' if include_str else 'without'}_STR",
        )

        return self._price_result

    def train_rent_model(
        self,
        df: pd.DataFrame,
        include_str: bool = True,
    ) -> TrainingResult:
        """Train rent prediction model.

        Args:
            df: Features DataFrame.
            include_str: Whether to include STR features.

        Returns:
            TrainingResult for rent model.
        """
        target = self.config.model.rent_target

        # Check if rent data is available
        if target not in df.columns:
            logger.warning(f"Rent target '{target}' not found, skipping")
            return None

        self._rent_result = self._trainer.train(
            df,
            target=target,
            model_type="gradient_boosting",
            name=f"RentModel_{'with' if include_str else 'without'}_STR",
        )

        return self._rent_result

    def train_both(
        self,
        df: pd.DataFrame,
        include_str: bool = True,
    ) -> tuple[TrainingResult, TrainingResult | None]:
        """Train both price and rent models.

        Args:
            df: Features DataFrame.
            include_str: Whether to include STR features.

        Returns:
            Tuple of (price_result, rent_result).
        """
        price_result = self.train_price_model(df, include_str)
        rent_result = self.train_rent_model(df, include_str)
        return price_result, rent_result

    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of price vs rent model performance.

        Returns:
            DataFrame with model comparison.
        """
        results = []

        if self._price_result:
            results.append({
                "model": "Price",
                "r2_train": self._price_result.train_result.r_squared,
                "r2_test": self._price_result.test_result.r_squared,
                "rmse_test": self._price_result.test_result.rmse,
            })

        if self._rent_result:
            results.append({
                "model": "Rent",
                "r2_train": self._rent_result.train_result.r_squared,
                "r2_test": self._rent_result.test_result.r_squared,
                "rmse_test": self._rent_result.test_result.rmse,
            })

        return pd.DataFrame(results)
