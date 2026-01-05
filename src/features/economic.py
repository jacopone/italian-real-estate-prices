"""Economic feature transformers.

Computes features related to income, employment, and economic indicators
that influence real estate prices and affordability.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.features.base import FeatureTransformer


class IncomeFeatures(FeatureTransformer):
    """Compute income-related features.

    Features computed:
    - log_income: Log-transformed average income
    - income_normalized: Percentile rank of income
    - income_change_pct: Income change percentage
    - income_ratio: Ratio to regional/national average
    - affordability_index: Price / Income ratio

    Example:
        >>> transformer = IncomeFeatures()
        >>> features = transformer.fit_transform(income_df)
    """

    def __init__(
        self,
        income_col: str = "avg_income",
        change_col: str = "income_change_pct",
        price_col: str | None = "prezzo_medio",
        include_log: bool = True,
        name: str | None = None,
    ):
        """Initialize income feature transformer.

        Args:
            income_col: Column name for average income.
            change_col: Column name for income change.
            price_col: Column name for price (for affordability).
            include_log: Whether to include log-transformed income.
            name: Optional transformer name.
        """
        super().__init__(name or "IncomeFeatures")
        self._income_col = income_col
        self._change_col = change_col
        self._price_col = price_col
        self._include_log = include_log
        self._national_mean: float = 0.0
        self._output_features: list[str] = []

    def fit(self, df: pd.DataFrame) -> "IncomeFeatures":
        """Learn income statistics from training data.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        self._output_features = []

        if self._income_col in df.columns:
            self._national_mean = df[self._income_col].mean()
            self._metadata.parameters["national_mean"] = self._national_mean

            if self._include_log:
                self._output_features.append("log_income")
            self._output_features.extend([
                "income_normalized",
                "income_ratio",
            ])

        if self._change_col in df.columns:
            self._output_features.append("income_change_pct")

        if self._price_col and self._price_col in df.columns and self._income_col in df.columns:
            self._output_features.append("affordability_index")

        self._metadata.output_features = self._output_features
        self._metadata.is_fitted = True

        logger.info(
            f"IncomeFeatures fitted: national_mean={self._national_mean:,.0f}, "
            f"features={self._output_features}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute income features.

        Args:
            df: DataFrame with income data.

        Returns:
            DataFrame with income features.
        """
        self._check_fitted()
        result = pd.DataFrame(index=df.index)

        if self._income_col in df.columns:
            income = df[self._income_col].fillna(0)

            if self._include_log:
                result["log_income"] = np.log1p(income)

            # Normalized income (percentile rank)
            result["income_normalized"] = income.rank(pct=True)

            # Ratio to national average
            if self._national_mean > 0:
                result["income_ratio"] = income / self._national_mean

        if self._change_col in df.columns:
            result["income_change_pct"] = df[self._change_col]

        # Affordability: price / (annual income / 12)
        if (
            self._price_col
            and self._price_col in df.columns
            and self._income_col in df.columns
        ):
            monthly_income = df[self._income_col] / 12
            # Assume 70 sqm apartment
            apartment_price = df[self._price_col] * 70
            result["affordability_index"] = apartment_price / df[self._income_col]
            result["affordability_index"] = result["affordability_index"].replace(
                [np.inf, -np.inf], np.nan
            )

        return result[[c for c in self._output_features if c in result.columns]]

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
        return self._output_features.copy()


class EconomicIndicators(FeatureTransformer):
    """Compute broader economic indicators.

    Features computed:
    - tax_compliance_proxy: Income per taxpayer vs regional average
    - economic_vitality: Composite of income growth and population
    - income_inequality: Gini-like measure if data available

    These capture economic health of an area beyond just income levels.
    """

    def __init__(
        self,
        income_col: str = "avg_income",
        taxpayers_col: str = "n_contribuenti",
        population_col: str = "popolazione",
        region_col: str = "cod_regione",
        name: str | None = None,
    ):
        """Initialize economic indicators transformer.

        Args:
            income_col: Column for average income.
            taxpayers_col: Column for number of taxpayers.
            population_col: Column for population.
            region_col: Column for region code.
            name: Optional transformer name.
        """
        super().__init__(name or "EconomicIndicators")
        self._income_col = income_col
        self._taxpayers_col = taxpayers_col
        self._population_col = population_col
        self._region_col = region_col
        self._regional_means: dict[int, float] = {}

    def fit(self, df: pd.DataFrame) -> "EconomicIndicators":
        """Learn regional income averages.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        output_features = []

        # Learn regional means if region column available
        if self._region_col in df.columns and self._income_col in df.columns:
            self._regional_means = (
                df.groupby(self._region_col)[self._income_col]
                .mean()
                .to_dict()
            )
            output_features.append("income_vs_region")

        # Tax compliance proxy
        if self._taxpayers_col in df.columns and self._population_col in df.columns:
            output_features.append("taxpayer_ratio")

        self._metadata.output_features = output_features
        self._metadata.parameters["regional_means"] = self._regional_means
        self._metadata.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute economic indicators.

        Args:
            df: DataFrame with economic data.

        Returns:
            DataFrame with economic indicators.
        """
        self._check_fitted()
        result = pd.DataFrame(index=df.index)

        # Income vs regional average
        if self._region_col in df.columns and self._income_col in df.columns:
            regional_avg = df[self._region_col].map(self._regional_means)
            result["income_vs_region"] = df[self._income_col] / regional_avg
            result["income_vs_region"] = result["income_vs_region"].replace(
                [np.inf, -np.inf], np.nan
            )

        # Taxpayer ratio (proxy for formal economy participation)
        if self._taxpayers_col in df.columns and self._population_col in df.columns:
            result["taxpayer_ratio"] = (
                df[self._taxpayers_col] / df[self._population_col]
            )
            result["taxpayer_ratio"] = result["taxpayer_ratio"].replace(
                [np.inf, -np.inf], np.nan
            )

        return result

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
        return self._metadata.output_features.copy()
