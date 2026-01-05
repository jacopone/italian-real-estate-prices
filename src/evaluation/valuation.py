"""Valuation analysis and undervaluation detection.

Implements the core logic for identifying undervalued properties
based on model residuals and computing investment metrics.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from src.config import EvaluationConfig


@dataclass
class ValuationResult:
    """Result of valuation analysis for a municipality.

    Attributes:
        istat_code: Municipality ISTAT code.
        name: Municipality name.
        actual_price: Actual price EUR/sqm.
        predicted_price: Model predicted price.
        residual: actual - predicted (negative = undervalued).
        price_gap_pct: Residual as percentage of predicted.
        valuation_category: Categorical valuation label.
        gross_yield: Gross rental yield percentage.
        valuation_score: Combined score for ranking.
    """

    istat_code: str
    name: str
    actual_price: float
    predicted_price: float
    residual: float
    price_gap_pct: float
    valuation_category: str
    gross_yield: float | None = None
    valuation_score: float | None = None
    metadata: dict = field(default_factory=dict)


class ValuationAnalyzer:
    """Analyze property valuations and identify investment opportunities.

    Uses model residuals to detect undervalued properties:
    - Negative residual = actual < predicted = undervalued
    - Combines with yield analysis for smart picks

    Example:
        >>> analyzer = ValuationAnalyzer(config.evaluation)
        >>> valuations = analyzer.compute_valuations(df, predictions)
        >>> smart_picks = analyzer.get_smart_picks(valuations)
    """

    def __init__(self, config: EvaluationConfig | None = None):
        """Initialize analyzer.

        Args:
            config: Evaluation configuration.
        """
        self.config = config or EvaluationConfig()

        # Valuation category thresholds
        self._categories = {
            "severely_undervalued": (-1.0, -0.30),
            "undervalued": (-0.30, -0.15),
            "fair_value": (-0.15, 0.15),
            "overvalued": (0.15, 0.30),
            "severely_overvalued": (0.30, 1.0),
        }

    def compute_valuations(
        self,
        df: pd.DataFrame,
        predictions: pd.Series,
        price_col: str = "prezzo_medio",
        rent_col: str | None = "affitto_medio",
        name_col: str = "nome",
    ) -> pd.DataFrame:
        """Compute valuations for all municipalities.

        Args:
            df: DataFrame with actual prices and metadata.
            predictions: Model predictions (same index as df).
            price_col: Column with actual prices.
            rent_col: Column with rents (for yield calculation).
            name_col: Column with municipality names.

        Returns:
            DataFrame with valuation metrics.
        """
        result = pd.DataFrame(index=df.index)

        # Identifiers
        if "istat_code" in df.columns:
            result["istat_code"] = df["istat_code"]
        if name_col in df.columns:
            result["name"] = df[name_col]

        # Actual values
        result["actual_price"] = df[price_col]
        result["predicted_price"] = predictions

        # Residual and percentage gap
        result["residual"] = result["actual_price"] - result["predicted_price"]
        result["price_gap_pct"] = (
            result["residual"] / result["predicted_price"] * 100
        )

        # Categorize
        result["valuation_category"] = result["price_gap_pct"].apply(
            self._categorize_valuation
        )

        # Gross yield if rent available
        if rent_col and rent_col in df.columns:
            annual_rent = df[rent_col] * 12 * 70  # Assume 70 sqm
            apartment_price = result["actual_price"] * 70
            result["gross_yield_pct"] = (annual_rent / apartment_price * 100)
            result["gross_yield_pct"] = result["gross_yield_pct"].replace(
                [np.inf, -np.inf], np.nan
            )

        # Valuation score (lower = more undervalued)
        result["valuation_score"] = result["price_gap_pct"]

        logger.info(
            f"Computed valuations: "
            f"{(result['valuation_category'] == 'undervalued').sum()} undervalued, "
            f"{(result['valuation_category'] == 'severely_undervalued').sum()} severely undervalued"
        )

        return result

    def _categorize_valuation(self, gap_pct: float) -> str:
        """Categorize valuation based on price gap percentage.

        Args:
            gap_pct: Price gap as percentage.

        Returns:
            Valuation category string.
        """
        gap_frac = gap_pct / 100

        for category, (low, high) in self._categories.items():
            if low <= gap_frac < high:
                return category

        return "fair_value"

    def get_undervalued(
        self,
        valuations: pd.DataFrame,
        threshold_pct: float | None = None,
    ) -> pd.DataFrame:
        """Get undervalued municipalities.

        Args:
            valuations: DataFrame from compute_valuations.
            threshold_pct: Undervaluation threshold (default from config).

        Returns:
            Filtered DataFrame of undervalued properties.
        """
        threshold = threshold_pct or (self.config.undervaluation_threshold * 100)

        undervalued = valuations[
            valuations["price_gap_pct"] <= threshold
        ].copy()

        # Sort by most undervalued
        undervalued = undervalued.sort_values("price_gap_pct")

        logger.info(
            f"Found {len(undervalued)} undervalued municipalities "
            f"(threshold: {threshold:.0f}%)"
        )

        return undervalued

    def get_smart_picks(
        self,
        valuations: pd.DataFrame,
        min_yield_pct: float | None = None,
        max_price_gap_pct: float | None = None,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Get smart investment picks combining undervaluation and yield.

        Smart picks are municipalities that are:
        1. Undervalued (price_gap_pct <= threshold)
        2. High yielding (gross_yield_pct >= threshold)

        Args:
            valuations: DataFrame from compute_valuations.
            min_yield_pct: Minimum yield threshold.
            max_price_gap_pct: Maximum price gap (most negative = most undervalued).
            top_n: Limit to top N picks.

        Returns:
            DataFrame of smart picks sorted by combined score.
        """
        min_yield = min_yield_pct or self.config.min_yield_pct
        max_gap = max_price_gap_pct or self.config.max_price_gap_pct

        # Must have yield data
        if "gross_yield_pct" not in valuations.columns:
            logger.warning("No yield data available for smart picks")
            return pd.DataFrame()

        picks = valuations[
            (valuations["price_gap_pct"] <= max_gap)
            & (valuations["gross_yield_pct"] >= min_yield)
            & valuations["gross_yield_pct"].notna()
        ].copy()

        # Combined score: more negative gap + higher yield = better
        # Normalize both to 0-1 range and combine
        if len(picks) > 0:
            gap_norm = -picks["price_gap_pct"] / 100  # Flip sign
            yield_norm = picks["gross_yield_pct"] / 20  # Normalize to ~1
            picks["combined_score"] = gap_norm + yield_norm

            picks = picks.sort_values("combined_score", ascending=False)

        if top_n:
            picks = picks.head(top_n)

        logger.info(
            f"Found {len(picks)} smart picks "
            f"(gap <= {max_gap}%, yield >= {min_yield}%)"
        )

        return picks

    def compute_regional_valuations(
        self,
        valuations: pd.DataFrame,
        region_col: str = "regione",
    ) -> pd.DataFrame:
        """Aggregate valuations by region.

        Args:
            valuations: Municipality-level valuations.
            region_col: Column with region names.

        Returns:
            Regional summary DataFrame.
        """
        if region_col not in valuations.columns:
            logger.warning(f"Region column '{region_col}' not found")
            return pd.DataFrame()

        regional = (
            valuations.groupby(region_col)
            .agg({
                "price_gap_pct": ["mean", "median", "std"],
                "gross_yield_pct": ["mean", "median"],
                "istat_code": "count",
            })
        )

        regional.columns = [
            "avg_gap_pct", "median_gap_pct", "std_gap_pct",
            "avg_yield_pct", "median_yield_pct", "n_municipalities",
        ]

        regional = regional.sort_values("avg_gap_pct")

        return regional.reset_index()


def identify_undervalued_municipalities(
    df: pd.DataFrame,
    predictions: pd.Series,
    config: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Convenience function to identify undervalued municipalities.

    Args:
        df: Data with actual prices.
        predictions: Model predictions.
        config: Optional evaluation configuration.

    Returns:
        DataFrame of undervalued municipalities.
    """
    analyzer = ValuationAnalyzer(config)
    valuations = analyzer.compute_valuations(df, predictions)
    return analyzer.get_undervalued(valuations)


def compute_smart_picks(
    df: pd.DataFrame,
    price_predictions: pd.Series,
    config: EvaluationConfig | None = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """Convenience function to compute smart investment picks.

    Args:
        df: Data with prices and rents.
        price_predictions: Price model predictions.
        config: Optional evaluation configuration.
        top_n: Number of picks to return.

    Returns:
        DataFrame of top smart picks.
    """
    analyzer = ValuationAnalyzer(config)
    valuations = analyzer.compute_valuations(df, price_predictions)
    return analyzer.get_smart_picks(valuations, top_n=top_n)
