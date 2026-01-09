"""Train price prediction models with vacancy features.

This script compares model performance with and without vacancy features,
demonstrating the predictive value of vacancy classification.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split

from src.config import ModelConfig
from src.features.vacancy_features import create_vacancy_model_features
from src.models import (
    GradientBoostingModel,
    OLSModel,
    RandomForestModel,
    RidgeModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VacancyModelTrainer:
    """Trainer for price models with vacancy features."""

    def __init__(
        self,
        data_path: Path = Path("data/processed/municipal_features.csv"),
        config: ModelConfig | None = None,
    ):
        self.data_path = data_path
        self.config = config or ModelConfig()
        self.results: dict[str, dict[str, Any]] = {}

    def load_data(self) -> pd.DataFrame:
        """Load and prepare municipal features data."""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Basic data cleaning
        df = df.dropna(subset=["price_avg"])

        # Remove outliers (prices below €100 or above €10,000 per sqm)
        df = df[(df["price_avg"] >= 100) & (df["price_avg"] <= 10000)]

        logger.info(f"Loaded {len(df)} municipalities after cleaning")
        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        include_vacancy: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training.

        Args:
            df: Raw DataFrame
            include_vacancy: Whether to include vacancy features

        Returns:
            Tuple of (X, y)
        """
        # Create vacancy features
        if include_vacancy:
            df = create_vacancy_model_features(df, include_interactions=True)

        # Define feature sets
        base_features = ["zone_count"]

        yield_features = ["gross_yield"]

        vacancy_features = [
            "vacancy_rate",
            "log_vacancy_rate",
            "is_tourist_vacancy",
            "is_decline_vacancy",
            "is_high_vacancy",
            "vacancy_risk_score",
        ]

        demographic_features = [
            "pop_10yr_change",
            "pop_change_category",
            "is_depopulating_severe",
        ]

        tourism_features = [
            "log_tourism_intensity",
            "tourism_category",
            "is_high_tourism",
        ]

        interaction_features = [
            "vacancy_x_tourism",
            "vacancy_x_pop_change",
        ]

        # Build feature list
        all_features = base_features + yield_features

        if include_vacancy:
            all_features.extend(vacancy_features)
            all_features.extend(demographic_features)
            all_features.extend(tourism_features)
            all_features.extend(interaction_features)
        else:
            # Minimal features for baseline
            all_features.append("vacancy_rate")

        # Filter to available columns
        available_features = [f for f in all_features if f in df.columns]

        # Create target (log price for better distribution)
        df["log_price_avg"] = np.log1p(df["price_avg"])

        # Drop missing values
        subset = df[available_features + ["log_price_avg"]].dropna()

        X = subset[available_features]
        y = subset["log_price_avg"]

        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Samples: {len(X)}")

        return X, y

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = "GradientBoosting",
    ) -> dict[str, Any]:
        """Train a model and return evaluation metrics.

        Args:
            X: Feature matrix
            y: Target values
            model_name: Type of model to train

        Returns:
            Dictionary with metrics and model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        # Create model
        if model_name == "GradientBoosting":
            model = GradientBoostingModel(**self.config.gb_params)
        elif model_name == "RandomForest":
            model = RandomForestModel(n_estimators=200, random_state=42)
        elif model_name == "Ridge":
            model = RidgeModel(alpha=1.0)
        elif model_name == "OLS":
            model = OLSModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        train_result = model.evaluate(X_train, y_train)
        test_result = model.evaluate(X_test, y_test)

        # Cross-validation
        cv_scores = cross_val_score(
            model._model, X_train, y_train,
            cv=self.config.cv_folds,
            scoring="r2",
        )

        # Convert predictions back to price scale for RMSE in EUR
        test_predictions_price = np.expm1(test_result.predictions)
        test_actual_price = np.expm1(y_test)
        price_rmse = np.sqrt(np.mean((test_predictions_price - test_actual_price) ** 2))
        price_mae = np.mean(np.abs(test_predictions_price - test_actual_price))

        return {
            "model": model,
            "train_r2": train_result.r_squared,
            "test_r2": test_result.r_squared,
            "test_rmse_log": test_result.rmse,
            "test_mae_log": test_result.mae,
            "test_rmse_eur": price_rmse,
            "test_mae_eur": price_mae,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "feature_importance": model.get_feature_importance(),
            "n_features": len(X.columns),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def compare_models(self) -> pd.DataFrame:
        """Compare models with and without vacancy features.

        Returns:
            DataFrame comparing model performance
        """
        df = self.load_data()

        results = []

        # Test different model types
        model_types = ["GradientBoosting", "RandomForest", "Ridge"]

        for model_type in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type}")
            logger.info("="*60)

            # Without vacancy features
            logger.info("\n--- Without Vacancy Features ---")
            X_base, y_base = self.prepare_features(df.copy(), include_vacancy=False)
            result_base = self.train_and_evaluate(X_base, y_base, model_type)

            results.append({
                "model": model_type,
                "features": "baseline",
                "n_features": result_base["n_features"],
                "train_r2": result_base["train_r2"],
                "test_r2": result_base["test_r2"],
                "cv_r2": result_base["cv_mean"],
                "cv_std": result_base["cv_std"],
                "rmse_eur": result_base["test_rmse_eur"],
                "mae_eur": result_base["test_mae_eur"],
            })

            # With vacancy features
            logger.info("\n--- With Vacancy Features ---")
            X_vacancy, y_vacancy = self.prepare_features(df.copy(), include_vacancy=True)
            result_vacancy = self.train_and_evaluate(X_vacancy, y_vacancy, model_type)

            results.append({
                "model": model_type,
                "features": "with_vacancy",
                "n_features": result_vacancy["n_features"],
                "train_r2": result_vacancy["train_r2"],
                "test_r2": result_vacancy["test_r2"],
                "cv_r2": result_vacancy["cv_mean"],
                "cv_std": result_vacancy["cv_std"],
                "rmse_eur": result_vacancy["test_rmse_eur"],
                "mae_eur": result_vacancy["test_mae_eur"],
            })

            # Store results
            self.results[f"{model_type}_baseline"] = result_base
            self.results[f"{model_type}_vacancy"] = result_vacancy

            # Calculate improvement
            r2_improvement = result_vacancy["test_r2"] - result_base["test_r2"]
            rmse_improvement = result_base["test_rmse_eur"] - result_vacancy["test_rmse_eur"]

            logger.info(f"\n{model_type} Improvement with Vacancy Features:")
            logger.info(f"  R² improvement: {r2_improvement:+.4f} ({r2_improvement/result_base['test_r2']*100:+.1f}%)")
            logger.info(f"  RMSE reduction: €{rmse_improvement:+.0f}")

        return pd.DataFrame(results)

    def get_feature_importance_report(self) -> pd.DataFrame:
        """Get feature importance from the best model.

        Returns:
            DataFrame with feature importance rankings
        """
        if "GradientBoosting_vacancy" not in self.results:
            raise ValueError("Run compare_models() first")

        importance = self.results["GradientBoosting_vacancy"]["feature_importance"]

        # Sort by importance
        sorted_importance = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Create DataFrame
        df = pd.DataFrame(sorted_importance, columns=["feature", "importance"])
        df["importance_pct"] = df["importance"] / df["importance"].sum() * 100

        # Flag vacancy features
        vacancy_features = {
            "vacancy_rate", "log_vacancy_rate", "is_tourist_vacancy",
            "is_decline_vacancy", "is_high_vacancy", "vacancy_risk_score",
            "vacancy_x_tourism", "vacancy_x_pop_change",
            "pop_10yr_change", "pop_change_category", "is_depopulating_severe",
            "log_tourism_intensity", "tourism_category", "is_high_tourism",
        }
        df["is_vacancy_feature"] = df["feature"].isin(vacancy_features)

        return df

    def save_results(self, output_dir: Path = Path("outputs")):
        """Save training results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comparison table
        comparison_df = self.compare_models()
        comparison_df.to_csv(output_dir / "vacancy_model_comparison.csv", index=False)

        # Save feature importance
        importance_df = self.get_feature_importance_report()
        importance_df.to_csv(output_dir / "vacancy_feature_importance.csv", index=False)

        # Save best model
        if "GradientBoosting_vacancy" in self.results:
            model = self.results["GradientBoosting_vacancy"]["model"]
            model.save(output_dir / "price_model_with_vacancy.joblib")

        logger.info(f"Results saved to {output_dir}")


def main():
    """Run model comparison with vacancy features."""
    trainer = VacancyModelTrainer()

    # Compare models
    comparison = trainer.compare_models()

    # Print summary
    print("\n" + "="*80)
    print("MODEL COMPARISON: With vs Without Vacancy Features")
    print("="*80)
    print(comparison.to_string(index=False))

    # Feature importance
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (GradientBoosting with Vacancy Features)")
    print("="*80)
    importance = trainer.get_feature_importance_report()
    print(importance.head(15).to_string(index=False))

    # Vacancy features contribution
    vacancy_importance = importance[importance["is_vacancy_feature"]]["importance_pct"].sum()
    print(f"\nVacancy features total importance: {vacancy_importance:.1f}%")

    # Calculate overall improvement
    gb_base = comparison[(comparison["model"] == "GradientBoosting") & (comparison["features"] == "baseline")].iloc[0]
    gb_vacancy = comparison[(comparison["model"] == "GradientBoosting") & (comparison["features"] == "with_vacancy")].iloc[0]

    print("\n" + "="*80)
    print("SUMMARY: GradientBoosting Model Improvement")
    print("="*80)
    print(f"Baseline R²:     {gb_base['test_r2']:.4f}")
    print(f"With Vacancy R²: {gb_vacancy['test_r2']:.4f}")
    print(f"R² Improvement:  {gb_vacancy['test_r2'] - gb_base['test_r2']:+.4f}")
    print(f"\nBaseline RMSE:     €{gb_base['rmse_eur']:,.0f}")
    print(f"With Vacancy RMSE: €{gb_vacancy['rmse_eur']:,.0f}")
    print(f"RMSE Reduction:    €{gb_base['rmse_eur'] - gb_vacancy['rmse_eur']:,.0f}")

    # Save results
    trainer.save_results()

    return trainer


if __name__ == "__main__":
    main()
