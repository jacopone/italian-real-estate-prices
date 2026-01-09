"""Vacancy feature engineering for price prediction models.

Creates derived features from vacancy classification that improve
price prediction accuracy by capturing market dynamics.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_vacancy_model_features(
    df: pd.DataFrame,
    include_interactions: bool = True,
) -> pd.DataFrame:
    """Create model-ready vacancy features from municipal data.

    Features created:
    - vacancy_rate: Raw vacancy rate (proportion)
    - vacancy_type_code: Numeric encoding (0=low, 1=tourist, 2=decline, 3=mixed)
    - is_tourist_vacancy: Binary flag for tourist vacancy areas
    - is_decline_vacancy: Binary flag for declining areas
    - is_high_vacancy: Binary flag for vacancy > 30%
    - pop_change_category: Binned population change
    - tourism_category: Binned tourism intensity
    - vacancy_risk_score: Composite risk score
    - Interaction features (if include_interactions=True)

    Args:
        df: DataFrame with vacancy_type, vacancy_rate, pop_10yr_change, tourism_intensity
        include_interactions: Whether to create interaction features

    Returns:
        DataFrame with additional vacancy features for modeling
    """
    result = df.copy()

    # Ensure required columns exist
    required_cols = ["vacancy_rate"]
    missing = [c for c in required_cols if c not in result.columns]
    if missing:
        logger.warning(f"Missing columns for vacancy features: {missing}")
        return result

    # Binary flags for vacancy types
    if "vacancy_type" in result.columns:
        result["is_tourist_vacancy"] = (result["vacancy_type"] == "tourist_vacancy").astype(int)
        result["is_decline_vacancy"] = (result["vacancy_type"] == "decline_vacancy").astype(int)
        result["is_mixed_vacancy"] = (result["vacancy_type"] == "mixed_vacancy").astype(int)
        result["is_low_vacancy"] = (result["vacancy_type"] == "low_vacancy").astype(int)
    else:
        # Default to 0 if no vacancy type
        result["is_tourist_vacancy"] = 0
        result["is_decline_vacancy"] = 0
        result["is_mixed_vacancy"] = 0
        result["is_low_vacancy"] = 1

    # High vacancy flag (> 30%)
    result["is_high_vacancy"] = (result["vacancy_rate"] > 0.30).astype(int)

    # Log-transformed vacancy rate (more normal distribution)
    result["log_vacancy_rate"] = np.log1p(result["vacancy_rate"])

    # Population change category
    if "pop_10yr_change" in result.columns:
        result["pop_change_category"] = pd.cut(
            result["pop_10yr_change"],
            bins=[-np.inf, -0.15, -0.05, 0.05, np.inf],
            labels=[0, 1, 2, 3],  # severe_decline, decline, stable, growth
        ).astype(float).fillna(2)  # Default to stable

        # Severe decline flag
        result["is_depopulating_severe"] = (result["pop_10yr_change"] < -0.15).astype(int)
    else:
        result["pop_change_category"] = 2  # Stable
        result["is_depopulating_severe"] = 0

    # Tourism intensity category
    if "tourism_intensity" in result.columns:
        result["tourism_category"] = pd.cut(
            result["tourism_intensity"],
            bins=[-np.inf, 2, 10, 30, np.inf],
            labels=[0, 1, 2, 3],  # very_low, low, medium, high
        ).astype(float).fillna(1)  # Default to low

        # Log-transformed tourism
        result["log_tourism_intensity"] = np.log1p(result["tourism_intensity"].fillna(0))

        # High tourism flag
        result["is_high_tourism"] = (result["tourism_intensity"] > 20).astype(int)
    else:
        result["tourism_category"] = 1  # Low
        result["log_tourism_intensity"] = 0
        result["is_high_tourism"] = 0

    # Vacancy risk score (composite)
    # Higher = riskier for investment
    result["vacancy_risk_score"] = _calculate_vacancy_risk_score(result)

    # Interaction features
    if include_interactions:
        result = _add_interaction_features(result)

    # Log features created
    new_cols = [c for c in result.columns if c not in df.columns]
    logger.info(f"Created {len(new_cols)} vacancy features: {new_cols}")

    return result


def _calculate_vacancy_risk_score(df: pd.DataFrame) -> pd.Series:
    """Calculate composite vacancy risk score.

    Risk factors:
    - High vacancy rate (30%+ = max risk)
    - Declining population (15%+ decline = high risk)
    - Low tourism (no tourist support)
    - Decline vacancy type (structural abandonment)

    Returns:
        Series with risk scores from 0 (low risk) to 1 (high risk)
    """
    # Base score from vacancy rate (0-0.4 contribution)
    vacancy_contribution = np.clip(df["vacancy_rate"] / 0.5, 0, 0.4)

    # Population decline contribution (0-0.3)
    if "pop_10yr_change" in df.columns:
        pop_contribution = np.clip(-df["pop_10yr_change"] / 0.20, 0, 0.3)
    else:
        pop_contribution = 0.1  # Default moderate contribution

    # Vacancy type contribution (0-0.3)
    if "vacancy_type" in df.columns:
        type_scores = {
            "low_vacancy": 0.0,
            "tourist_vacancy": 0.1,  # Some regulatory risk
            "mixed_vacancy": 0.2,
            "decline_vacancy": 0.3,
        }
        type_contribution = df["vacancy_type"].map(type_scores).fillna(0.15)
    else:
        type_contribution = 0.15

    # Combine contributions
    risk_score = vacancy_contribution + pop_contribution + type_contribution

    return np.clip(risk_score, 0, 1)


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between vacancy and other variables.

    Interactions capture non-linear relationships like:
    - High vacancy + tourist area = different dynamics than decline area
    - Price sensitivity varies by vacancy type
    """
    result = df.copy()

    # Vacancy × tourism interaction
    if "tourism_intensity" in result.columns:
        result["vacancy_x_tourism"] = (
            result["vacancy_rate"] * result["tourism_intensity"].fillna(0) / 10
        )

    # Vacancy × population change interaction
    if "pop_10yr_change" in result.columns:
        result["vacancy_x_pop_change"] = (
            result["vacancy_rate"] * (1 + result["pop_10yr_change"].fillna(0))
        )

    # Price × vacancy interaction (if price available)
    if "price_avg" in result.columns:
        result["price_x_vacancy"] = result["price_avg"] * result["vacancy_rate"]

        # Log price for modeling
        result["log_price_avg"] = np.log1p(result["price_avg"])

    # Yield adjustment by vacancy type
    if "gross_yield" in result.columns and "vacancy_type" in result.columns:
        # Tourist areas: yield is sustainable
        # Decline areas: yield may be misleading (hard to rent)
        yield_multiplier = result["vacancy_type"].map({
            "low_vacancy": 1.0,
            "tourist_vacancy": 0.95,  # Slight discount for seasonality
            "mixed_vacancy": 0.85,
            "decline_vacancy": 0.70,  # Significant discount
        }).fillna(0.90)

        result["adjusted_yield"] = result["gross_yield"] * yield_multiplier

    return result


def prepare_model_features(
    df: pd.DataFrame,
    target: str = "log_price_avg",
    include_vacancy: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare features and target for model training.

    Args:
        df: Municipal features DataFrame
        target: Target column name
        include_vacancy: Whether to include vacancy features

    Returns:
        Tuple of (X features, y target, feature_names)
    """
    # Create vacancy features if requested
    if include_vacancy:
        df = create_vacancy_model_features(df, include_interactions=True)

    # Define vacancy feature group
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
        "tourism_intensity",
        "log_tourism_intensity",
        "tourism_category",
        "is_high_tourism",
    ]

    interaction_features = [
        "vacancy_x_tourism",
        "vacancy_x_pop_change",
    ]

    # Build feature list based on availability
    all_features = []
    if include_vacancy:
        all_features.extend(vacancy_features)
        all_features.extend(demographic_features)
        all_features.extend(tourism_features)
        all_features.extend(interaction_features)
    else:
        # Basic features only
        all_features.extend(["vacancy_rate"])

    # Filter to available columns
    available_features = [f for f in all_features if f in df.columns]

    # Ensure target exists
    if target not in df.columns:
        if "price_avg" in df.columns and target == "log_price_avg":
            df["log_price_avg"] = np.log1p(df["price_avg"])
        else:
            raise ValueError(f"Target '{target}' not found in DataFrame")

    # Drop rows with missing values
    subset = df[available_features + [target]].dropna()

    X = subset[available_features]
    y = subset[target]

    logger.info(f"Prepared {len(X)} samples with {len(available_features)} features")

    return X, y, available_features


def get_vacancy_feature_importance(
    feature_importance: dict[str, float],
    vacancy_features: list[str] | None = None,
) -> dict[str, float]:
    """Extract vacancy-related feature importance from model results.

    Args:
        feature_importance: Full feature importance dict from model
        vacancy_features: List of vacancy feature names (uses default if None)

    Returns:
        Dict of vacancy feature importances
    """
    if vacancy_features is None:
        vacancy_features = [
            "vacancy_rate", "log_vacancy_rate",
            "is_tourist_vacancy", "is_decline_vacancy", "is_high_vacancy",
            "vacancy_risk_score", "vacancy_x_tourism", "vacancy_x_pop_change",
            "pop_10yr_change", "pop_change_category", "is_depopulating_severe",
            "tourism_intensity", "log_tourism_intensity", "tourism_category",
            "is_high_tourism", "vacancy_type_code",
        ]

    return {
        k: v for k, v in feature_importance.items()
        if k in vacancy_features
    }
