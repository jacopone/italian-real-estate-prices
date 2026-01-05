"""Demographic feature transformers.

Computes features related to population dynamics, age structure,
and demographic trends that influence real estate demand.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.features.base import FeatureTransformer


class DemographicFeatures(FeatureTransformer):
    """Compute demographic features from population data.

    Features computed:
    - log_population: Log-transformed population
    - pop_change_pct: Population change percentage
    - pop_declining: Binary flag for declining population
    - pop_growing_fast: Binary flag for fast growth (>5%)
    - population_density: Population per sqkm (if area available)

    Example:
        >>> transformer = DemographicFeatures()
        >>> features = transformer.fit_transform(demographics_df)
    """

    def __init__(
        self,
        population_col: str = "popolazione",
        change_col: str = "pop_change_pct",
        include_log: bool = True,
        name: str | None = None,
    ):
        """Initialize demographic feature transformer.

        Args:
            population_col: Column name for population.
            change_col: Column name for population change.
            include_log: Whether to include log-transformed population.
            name: Optional transformer name.
        """
        super().__init__(name or "DemographicFeatures")
        self._population_col = population_col
        self._change_col = change_col
        self._include_log = include_log
        self._output_features: list[str] = []

    def fit(self, df: pd.DataFrame) -> "DemographicFeatures":
        """Determine which features can be computed.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        self._output_features = []

        # Check available columns
        if self._population_col in df.columns:
            if self._include_log:
                self._output_features.append("log_population")
            self._output_features.append("population_normalized")

        if self._change_col in df.columns:
            self._output_features.extend([
                "pop_change_pct",
                "pop_declining",
                "pop_growing_fast",
            ])

        # Check for decline column
        if "pop_declining" in df.columns:
            if "pop_declining" not in self._output_features:
                self._output_features.append("pop_declining")

        if "pop_growing_fast" in df.columns:
            if "pop_growing_fast" not in self._output_features:
                self._output_features.append("pop_growing_fast")

        self._metadata.output_features = self._output_features
        self._metadata.is_fitted = True

        logger.info(f"DemographicFeatures will produce: {self._output_features}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute demographic features.

        Args:
            df: DataFrame with population data.

        Returns:
            DataFrame with demographic features.
        """
        self._check_fitted()
        result = pd.DataFrame(index=df.index)

        # Log population
        if self._population_col in df.columns:
            pop = df[self._population_col].fillna(0)
            if self._include_log:
                result["log_population"] = np.log1p(pop)

            # Normalized population (percentile rank)
            result["population_normalized"] = pop.rank(pct=True)

        # Population change metrics
        if self._change_col in df.columns:
            change = df[self._change_col]
            result["pop_change_pct"] = change

            # Binary flags
            result["pop_declining"] = (change < -2).astype(int)
            result["pop_growing_fast"] = (change > 5).astype(int)

        # Pass through existing flags if available
        for col in ["pop_declining", "pop_growing_fast"]:
            if col in df.columns and col not in result.columns:
                result[col] = df[col]

        return result[[c for c in self._output_features if c in result.columns]]

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
        return self._output_features.copy()


class AgingFeatures(FeatureTransformer):
    """Compute features related to age structure.

    Features computed:
    - aging_index: Ratio of 65+ to 0-14 population
    - dependency_ratio: Ratio of dependents to working age
    - youth_share: Share of population under 15

    These features capture demographic pressure on real estate:
    - Aging areas may see reduced demand
    - Areas with young families may have more housing demand
    """

    def __init__(
        self,
        pop_0_14_col: str = "pop_0_14",
        pop_15_64_col: str = "pop_15_64",
        pop_65_plus_col: str = "pop_65_plus",
        name: str | None = None,
    ):
        """Initialize aging feature transformer.

        Args:
            pop_0_14_col: Column for 0-14 age group.
            pop_15_64_col: Column for 15-64 age group.
            pop_65_plus_col: Column for 65+ age group.
            name: Optional transformer name.
        """
        super().__init__(name or "AgingFeatures")
        self._pop_0_14 = pop_0_14_col
        self._pop_15_64 = pop_15_64_col
        self._pop_65_plus = pop_65_plus_col
        self._has_age_data = False

    def fit(self, df: pd.DataFrame) -> "AgingFeatures":
        """Check for age structure columns.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        required = [self._pop_0_14, self._pop_15_64, self._pop_65_plus]
        self._has_age_data = all(col in df.columns for col in required)

        if self._has_age_data:
            self._metadata.output_features = [
                "aging_index",
                "dependency_ratio",
                "youth_share",
                "elderly_share",
            ]
        else:
            self._metadata.output_features = []
            logger.warning(
                f"Age structure columns not found ({required}), "
                "AgingFeatures will produce no output"
            )

        self._metadata.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute aging features.

        Args:
            df: DataFrame with age structure data.

        Returns:
            DataFrame with aging features.
        """
        self._check_fitted()

        if not self._has_age_data:
            return pd.DataFrame(index=df.index)

        result = pd.DataFrame(index=df.index)

        pop_0_14 = df[self._pop_0_14].fillna(0)
        pop_15_64 = df[self._pop_15_64].fillna(0)
        pop_65_plus = df[self._pop_65_plus].fillna(0)
        total_pop = pop_0_14 + pop_15_64 + pop_65_plus

        # Aging index: elderly / youth
        result["aging_index"] = np.where(
            pop_0_14 > 0,
            pop_65_plus / pop_0_14,
            np.nan,
        )

        # Dependency ratio: (youth + elderly) / working age
        result["dependency_ratio"] = np.where(
            pop_15_64 > 0,
            (pop_0_14 + pop_65_plus) / pop_15_64,
            np.nan,
        )

        # Age shares
        result["youth_share"] = np.where(
            total_pop > 0,
            pop_0_14 / total_pop,
            np.nan,
        )
        result["elderly_share"] = np.where(
            total_pop > 0,
            pop_65_plus / total_pop,
            np.nan,
        )

        return result

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
        return self._metadata.output_features.copy()
