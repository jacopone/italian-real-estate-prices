"""Tourism and short-term rental (STR) feature transformers.

Computes features related to tourism intensity and Airbnb/STR activity.
These are critical for understanding price dynamics in tourist areas.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.features.base import FeatureTransformer


class TourismFeatures(FeatureTransformer):
    """Compute tourism-related features.

    Features computed:
    - tourism_intensity: Tourist arrivals per 1000 residents
    - log_tourism: Log-transformed tourism intensity
    - tourism_category: Categorical level (low/medium/high/very_high)
    - tourism_above_median: Binary flag

    Tourism intensity affects prices through:
    - Increased short-term rental demand
    - Amenity value of tourist infrastructure
    - Competition for housing

    Example:
        >>> transformer = TourismFeatures()
        >>> features = transformer.fit_transform(tourism_df)
    """

    def __init__(
        self,
        intensity_col: str = "tourism_intensity",
        include_log: bool = True,
        include_categories: bool = True,
        name: str | None = None,
    ):
        """Initialize tourism feature transformer.

        Args:
            intensity_col: Column for tourism intensity.
            include_log: Whether to include log transform.
            include_categories: Whether to include categorical features.
            name: Optional transformer name.
        """
        super().__init__(name or "TourismFeatures")
        self._intensity_col = intensity_col
        self._include_log = include_log
        self._include_categories = include_categories
        self._median_intensity = 0.0

        # Thresholds for categories
        self._thresholds = {
            "low": 100,
            "medium": 500,
            "high": 2000,
        }

    def fit(self, df: pd.DataFrame) -> "TourismFeatures":
        """Learn tourism statistics.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        output_features = []

        if self._intensity_col in df.columns:
            self._median_intensity = df[self._intensity_col].median()
            self._metadata.parameters["median_intensity"] = self._median_intensity

            output_features.append("tourism_intensity")
            if self._include_log:
                output_features.append("log_tourism")
            if self._include_categories:
                output_features.extend([
                    "tourism_category",
                    "tourism_above_median",
                    "high_tourism",
                ])
        else:
            logger.warning(
                f"Tourism intensity column '{self._intensity_col}' not found"
            )

        self._metadata.output_features = output_features
        self._metadata.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute tourism features.

        Args:
            df: DataFrame with tourism data.

        Returns:
            DataFrame with tourism features.
        """
        self._check_fitted()
        result = pd.DataFrame(index=df.index)

        if self._intensity_col not in df.columns:
            return result

        intensity = df[self._intensity_col].fillna(0)
        result["tourism_intensity"] = intensity

        if self._include_log:
            result["log_tourism"] = np.log1p(intensity)

        if self._include_categories:
            # Categorical level
            result["tourism_category"] = pd.cut(
                intensity,
                bins=[0, 100, 500, 2000, np.inf],
                labels=["low", "medium", "high", "very_high"],
                include_lowest=True,
            )

            # Binary flags
            result["tourism_above_median"] = (
                intensity > self._median_intensity
            ).astype(int)
            result["high_tourism"] = (intensity > 500).astype(int)

        return result[[c for c in self._metadata.output_features if c in result.columns]]

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
        return self._metadata.output_features.copy()


class STRFeatures(FeatureTransformer):
    """Compute short-term rental (Airbnb) features.

    Features computed:
    - str_density: STR listings per 1000 residents
    - log_str_density: Log-transformed STR density
    - str_price: Median/mean Airbnb price
    - str_premium: Airbnb monthly revenue vs long-term rent
    - str_revenue_potential: Estimated annual STR revenue

    STR density is the single most important feature for price prediction
    in tourist areas, capturing the "Airbnb effect" on housing markets.

    Example:
        >>> transformer = STRFeatures()
        >>> features = transformer.fit_transform(airbnb_df)
    """

    def __init__(
        self,
        density_col: str = "str_density",
        price_col: str = "airbnb_price_median",
        premium_col: str = "str_premium",
        include_log: bool = True,
        name: str | None = None,
    ):
        """Initialize STR feature transformer.

        Args:
            density_col: Column for STR density.
            price_col: Column for STR price.
            premium_col: Column for STR premium.
            include_log: Whether to include log transforms.
            name: Optional transformer name.
        """
        super().__init__(name or "STRFeatures")
        self._density_col = density_col
        self._price_col = price_col
        self._premium_col = premium_col
        self._include_log = include_log

    def fit(self, df: pd.DataFrame) -> "STRFeatures":
        """Determine which features can be computed.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        output_features = []

        if self._density_col in df.columns:
            output_features.append("str_density")
            if self._include_log:
                output_features.append("log_str_density")
            output_features.append("has_str_data")

        if self._price_col in df.columns:
            output_features.append("str_price")
            if self._include_log:
                output_features.append("log_str_price")

        if self._premium_col in df.columns:
            output_features.append("str_premium")

        self._metadata.output_features = output_features
        self._metadata.is_fitted = True

        logger.info(f"STRFeatures will produce: {output_features}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute STR features.

        Args:
            df: DataFrame with STR data.

        Returns:
            DataFrame with STR features.
        """
        self._check_fitted()
        result = pd.DataFrame(index=df.index)

        if self._density_col in df.columns:
            density = df[self._density_col].fillna(0)
            result["str_density"] = density

            if self._include_log:
                result["log_str_density"] = np.log1p(density)

            # Flag for having STR data
            result["has_str_data"] = (density > 0).astype(int)

        if self._price_col in df.columns:
            price = df[self._price_col]
            result["str_price"] = price

            if self._include_log:
                result["log_str_price"] = np.log1p(price.fillna(0))

        if self._premium_col in df.columns:
            result["str_premium"] = df[self._premium_col]

        return result[[c for c in self._metadata.output_features if c in result.columns]]

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
        return self._metadata.output_features.copy()


class STRProxyFeatures(FeatureTransformer):
    """Create STR proxy features for areas without direct Airbnb data.

    Uses tourism intensity as a proxy for STR activity, calibrated
    against areas where both tourism and Airbnb data are available.

    This allows the model to estimate STR effects even in provinces
    not covered by InsideAirbnb.
    """

    def __init__(
        self,
        tourism_col: str = "tourism_intensity",
        str_density_col: str = "str_density",
        name: str | None = None,
    ):
        """Initialize STR proxy transformer.

        Args:
            tourism_col: Tourism intensity column.
            str_density_col: Actual STR density column (for calibration).
            name: Optional transformer name.
        """
        super().__init__(name or "STRProxyFeatures")
        self._tourism_col = tourism_col
        self._str_density_col = str_density_col
        self._calibration_ratio = 1.0

    def fit(self, df: pd.DataFrame) -> "STRProxyFeatures":
        """Learn calibration ratio from overlapping data.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        if self._tourism_col not in df.columns:
            logger.warning(f"Tourism column '{self._tourism_col}' not found")
            self._metadata.is_fitted = True
            return self

        # If we have both tourism and STR data, learn the relationship
        if self._str_density_col in df.columns:
            has_both = (
                df[self._tourism_col].notna() &
                (df[self._tourism_col] > 0) &
                df[self._str_density_col].notna() &
                (df[self._str_density_col] > 0)
            )

            if has_both.sum() > 0:
                avg_str = df.loc[has_both, self._str_density_col].mean()
                avg_tourism = df.loc[has_both, self._tourism_col].mean()
                self._calibration_ratio = avg_str / avg_tourism

                logger.info(
                    f"STR proxy calibration: ratio={self._calibration_ratio:.4f} "
                    f"(from {has_both.sum()} observations)"
                )

        self._metadata.parameters["calibration_ratio"] = self._calibration_ratio
        self._metadata.output_features = ["str_proxy", "log_str_proxy"]
        self._metadata.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute STR proxy features.

        Args:
            df: DataFrame with tourism data.

        Returns:
            DataFrame with STR proxy features.
        """
        self._check_fitted()
        result = pd.DataFrame(index=df.index)

        if self._tourism_col in df.columns:
            tourism = df[self._tourism_col].fillna(0)
            result["str_proxy"] = tourism * self._calibration_ratio
            result["log_str_proxy"] = np.log1p(result["str_proxy"])

        return result

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
        return self._metadata.output_features.copy()
