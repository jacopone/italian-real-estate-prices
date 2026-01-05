"""Evaluation metrics and diagnostics.

Provides comprehensive model evaluation including standard metrics,
residual analysis, and model comparison utilities.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class RegressionMetrics:
    """Container for regression evaluation metrics.

    Attributes:
        r_squared: Coefficient of determination (R²).
        rmse: Root mean squared error.
        mae: Mean absolute error.
        mape: Mean absolute percentage error.
        median_ae: Median absolute error.
        n_samples: Number of samples evaluated.
    """

    r_squared: float
    rmse: float
    mae: float
    mape: float | None = None
    median_ae: float | None = None
    n_samples: int = 0
    additional: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        result = {
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "mae": self.mae,
            "n_samples": self.n_samples,
        }
        if self.mape is not None:
            result["mape"] = self.mape
        if self.median_ae is not None:
            result["median_ae"] = self.median_ae
        result.update(self.additional)
        return result

    def __repr__(self) -> str:
        return f"R²={self.r_squared:.4f}, RMSE={self.rmse:.4f}, MAE={self.mae:.4f}"


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    include_mape: bool = True,
) -> RegressionMetrics:
    """Compute comprehensive regression metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        include_mape: Whether to compute MAPE (fails if y has zeros).

    Returns:
        RegressionMetrics with all computed metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Filter out NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    median_ae = np.median(np.abs(y_true - y_pred))

    mape = None
    if include_mape:
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        except Exception:
            pass  # Skip if division by zero

    return RegressionMetrics(
        r_squared=r2,
        rmse=rmse,
        mae=mae,
        mape=mape,
        median_ae=median_ae,
        n_samples=len(y_true),
    )


@dataclass
class ResidualDiagnostics:
    """Diagnostics from residual analysis.

    Attributes:
        mean_residual: Mean of residuals (should be ~0).
        std_residual: Standard deviation of residuals.
        skewness: Residual skewness.
        kurtosis: Residual kurtosis (excess).
        normality_test_pvalue: P-value from normality test.
        heteroscedasticity: Whether heteroscedasticity detected.
    """

    mean_residual: float
    std_residual: float
    skewness: float
    kurtosis: float
    normality_test_pvalue: float | None = None
    heteroscedasticity: bool | None = None

    def is_well_behaved(self) -> bool:
        """Check if residuals are well-behaved.

        Returns True if:
        - Mean close to 0
        - Skewness close to 0
        - Kurtosis not extreme
        """
        return (
            abs(self.mean_residual) < 0.1 * self.std_residual
            and abs(self.skewness) < 1.0
            and abs(self.kurtosis) < 3.0
        )


def analyze_residuals(
    residuals: pd.Series | np.ndarray,
    predictions: pd.Series | np.ndarray | None = None,
) -> ResidualDiagnostics:
    """Analyze model residuals for diagnostic purposes.

    Args:
        residuals: Model residuals (y_true - y_pred).
        predictions: Model predictions (for heteroscedasticity check).

    Returns:
        ResidualDiagnostics with analysis results.
    """
    from scipy import stats

    residuals = np.array(residuals)
    residuals = residuals[~np.isnan(residuals)]

    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)

    # Normality test (Shapiro-Wilk for small samples, else use D'Agostino)
    normality_p = None
    if len(residuals) >= 8:
        try:
            if len(residuals) < 5000:
                _, normality_p = stats.shapiro(residuals)
            else:
                _, normality_p = stats.normaltest(residuals)
        except Exception:
            pass

    # Heteroscedasticity check (simple: correlation of abs(residuals) with predictions)
    heteroscedasticity = None
    if predictions is not None:
        predictions = np.array(predictions)
        predictions = predictions[~np.isnan(predictions)]
        if len(predictions) == len(residuals):
            corr = np.corrcoef(np.abs(residuals), predictions)[0, 1]
            heteroscedasticity = abs(corr) > 0.2

    return ResidualDiagnostics(
        mean_residual=mean_res,
        std_residual=std_res,
        skewness=skewness,
        kurtosis=kurtosis,
        normality_test_pvalue=normality_p,
        heteroscedasticity=heteroscedasticity,
    )


def compute_adjusted_r2(
    r_squared: float,
    n_samples: int,
    n_features: int,
) -> float:
    """Compute adjusted R² that penalizes extra features.

    Args:
        r_squared: Regular R².
        n_samples: Number of samples.
        n_features: Number of features.

    Returns:
        Adjusted R².
    """
    if n_samples <= n_features + 1:
        return r_squared

    adjusted = 1 - (1 - r_squared) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted


def compare_models(
    results: list[dict[str, Any]],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Create comparison table of model results.

    Args:
        results: List of result dictionaries with 'model' and metric keys.
        metrics: Metrics to include (default: R², RMSE, MAE).

    Returns:
        DataFrame comparing models.
    """
    if metrics is None:
        metrics = ["r_squared", "rmse", "mae"]

    df = pd.DataFrame(results)

    # Ensure model column is first
    cols = ["model"] + [c for c in metrics if c in df.columns]
    df = df[cols]

    # Sort by R² descending
    if "r_squared" in df.columns:
        df = df.sort_values("r_squared", ascending=False)

    return df


def compute_feature_importance_ranking(
    importance_dict: dict[str, float],
    normalize: bool = True,
) -> pd.DataFrame:
    """Convert feature importance dict to ranked DataFrame.

    Args:
        importance_dict: Feature name to importance mapping.
        normalize: Whether to normalize to sum to 1.

    Returns:
        DataFrame with features ranked by importance.
    """
    df = pd.DataFrame(
        list(importance_dict.items()),
        columns=["feature", "importance"],
    )

    if normalize and df["importance"].sum() > 0:
        df["importance"] = df["importance"] / df["importance"].sum()

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["cumulative"] = df["importance"].cumsum()

    return df[["rank", "feature", "importance", "cumulative"]]
