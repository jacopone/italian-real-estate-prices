"""Geographic feature transformers.

Computes features related to location, distance, and spatial characteristics
that influence real estate prices.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.features.base import FeatureTransformer
from src.utils.constants import MAJOR_CITIES, haversine_distance


class DistanceFeatures(FeatureTransformer):
    """Compute distance-based features.

    Features computed:
    - dist_major_city: Distance to nearest major city (km)
    - dist_milan: Distance to Milan specifically
    - dist_coast: Distance to nearest coast point
    - log_dist_city: Log-transformed distance to city

    Distance to economic centers is a key price predictor:
    closer = higher prices, with nonlinear decay.

    Example:
        >>> transformer = DistanceFeatures()
        >>> features = transformer.fit_transform(geo_df)
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        include_log: bool = True,
        name: str | None = None,
    ):
        """Initialize distance feature transformer.

        Args:
            lat_col: Column name for latitude.
            lon_col: Column name for longitude.
            include_log: Whether to include log-transformed distances.
            name: Optional transformer name.
        """
        super().__init__(name or "DistanceFeatures")
        self._lat_col = lat_col
        self._lon_col = lon_col
        self._include_log = include_log

        # Coastal reference points (approximate Italian coastline)
        self._coast_points = [
            (43.7, 10.3),   # Livorno (Tyrrhenian)
            (41.5, 12.5),   # Lazio coast
            (38.2, 15.6),   # Sicily (Messina)
            (40.8, 14.3),   # Naples
            (41.1, 16.9),   # Bari (Adriatic)
            (44.4, 12.2),   # Rimini
            (45.4, 12.3),   # Venice
            (44.3, 8.5),    # Genoa (Ligurian)
        ]

    def fit(self, df: pd.DataFrame) -> "DistanceFeatures":
        """Verify coordinate columns exist.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        self._check_columns(df, [self._lat_col, self._lon_col])

        output_features = [
            "dist_major_city",
            "dist_milan",
            "dist_coast",
            "nearest_city",
        ]
        if self._include_log:
            output_features.extend([
                "log_dist_city",
                "log_dist_coast",
            ])

        self._metadata.output_features = output_features
        self._metadata.is_fitted = True
        return self

    def _compute_dist_to_city(
        self,
        lat: float,
        lon: float,
    ) -> tuple[float, str]:
        """Compute distance to nearest major city.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Tuple of (distance_km, city_name).
        """
        min_dist = float("inf")
        nearest = ""

        for city in MAJOR_CITIES.values():
            dist = haversine_distance(lat, lon, city.lat, city.lon)
            if dist < min_dist:
                min_dist = dist
                nearest = city.name

        return min_dist, nearest

    def _compute_dist_to_coast(self, lat: float, lon: float) -> float:
        """Compute distance to nearest coast point.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Distance in km.
        """
        distances = [
            haversine_distance(lat, lon, coast_lat, coast_lon)
            for coast_lat, coast_lon in self._coast_points
        ]
        return min(distances)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute distance features.

        Args:
            df: DataFrame with coordinates.

        Returns:
            DataFrame with distance features.
        """
        self._check_fitted()
        self._check_columns(df, [self._lat_col, self._lon_col])

        result = pd.DataFrame(index=df.index)

        # Compute distances (vectorized where possible)
        distances = df.apply(
            lambda row: self._compute_dist_to_city(
                row[self._lat_col], row[self._lon_col]
            ),
            axis=1,
        )
        result["dist_major_city"] = [d[0] for d in distances]
        result["nearest_city"] = [d[1] for d in distances]

        # Distance to Milan specifically
        milan = MAJOR_CITIES["Milano"]
        result["dist_milan"] = df.apply(
            lambda row: haversine_distance(
                row[self._lat_col], row[self._lon_col],
                milan.lat, milan.lon,
            ),
            axis=1,
        )

        # Distance to coast
        result["dist_coast"] = df.apply(
            lambda row: self._compute_dist_to_coast(
                row[self._lat_col], row[self._lon_col]
            ),
            axis=1,
        )

        # Log transforms
        if self._include_log:
            result["log_dist_city"] = np.log1p(result["dist_major_city"])
            result["log_dist_coast"] = np.log1p(result["dist_coast"])

        return result

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
        return self._metadata.output_features.copy()


class LocationFlags(FeatureTransformer):
    """Compute binary location classification flags.

    Features computed:
    - northern: Is in Northern Italy (regions 1-8)
    - coastal: Is within 20km of coast
    - alpine_zone: Is in Alpine region
    - urban: Is urban area (population > threshold)
    - large_city: Population > 500k

    Binary flags capture categorical location effects.
    """

    def __init__(
        self,
        lat_col: str = "lat",
        region_col: str = "cod_regione",
        dist_coast_col: str = "dist_coast",
        population_col: str = "popolazione",
        coastal_threshold_km: float = 20.0,
        urban_threshold: int = 50000,
        name: str | None = None,
    ):
        """Initialize location flags transformer.

        Args:
            lat_col: Latitude column.
            region_col: Region code column.
            dist_coast_col: Distance to coast column.
            population_col: Population column.
            coastal_threshold_km: Km threshold for coastal flag.
            urban_threshold: Population threshold for urban flag.
            name: Optional transformer name.
        """
        super().__init__(name or "LocationFlags")
        self._lat_col = lat_col
        self._region_col = region_col
        self._dist_coast_col = dist_coast_col
        self._population_col = population_col
        self._coastal_threshold = coastal_threshold_km
        self._urban_threshold = urban_threshold

        # Northern regions (ISTAT codes)
        self._northern_regions = {1, 2, 3, 4, 5, 6, 7, 8}
        # Alpine regions
        self._alpine_regions = {1, 2, 4}  # Piemonte, Valle d'Aosta, Trentino

    def fit(self, df: pd.DataFrame) -> "LocationFlags":
        """Determine which flags can be computed.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        output_features = []

        if self._region_col in df.columns:
            output_features.extend(["northern", "alpine_zone"])
        elif self._lat_col in df.columns:
            output_features.append("northern")  # Use latitude as proxy

        if self._dist_coast_col in df.columns:
            output_features.append("coastal")

        if self._population_col in df.columns:
            output_features.extend(["urban", "large_city"])

        self._metadata.output_features = output_features
        self._metadata.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute location flags.

        Args:
            df: DataFrame with location data.

        Returns:
            DataFrame with binary flags.
        """
        self._check_fitted()
        result = pd.DataFrame(index=df.index)

        # Northern flag
        if self._region_col in df.columns:
            result["northern"] = df[self._region_col].isin(
                self._northern_regions
            ).astype(int)
            result["alpine_zone"] = df[self._region_col].isin(
                self._alpine_regions
            ).astype(int)
        elif self._lat_col in df.columns:
            # Use latitude 43.5 as approximate north/south divide
            result["northern"] = (df[self._lat_col] > 43.5).astype(int)

        # Coastal flag
        if self._dist_coast_col in df.columns:
            result["coastal"] = (
                df[self._dist_coast_col] < self._coastal_threshold
            ).astype(int)

        # Urban flags
        if self._population_col in df.columns:
            result["urban"] = (
                df[self._population_col] > self._urban_threshold
            ).astype(int)
            result["large_city"] = (
                df[self._population_col] > 500000
            ).astype(int)

        return result[[c for c in self._metadata.output_features if c in result.columns]]

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
        return self._metadata.output_features.copy()


class CoordinateFeatures(FeatureTransformer):
    """Pass through and optionally transform coordinate features.

    Coordinates can be used directly or standardized.
    Latitude captures north-south economic gradient.
    Longitude captures east-west variation.
    """

    def __init__(
        self,
        lat_col: str = "lat",
        lon_col: str = "lon",
        standardize: bool = False,
        name: str | None = None,
    ):
        """Initialize coordinate transformer.

        Args:
            lat_col: Latitude column.
            lon_col: Longitude column.
            standardize: Whether to standardize coordinates.
            name: Optional transformer name.
        """
        super().__init__(name or "CoordinateFeatures")
        self._lat_col = lat_col
        self._lon_col = lon_col
        self._standardize = standardize
        self._lat_mean = 0.0
        self._lat_std = 1.0
        self._lon_mean = 0.0
        self._lon_std = 1.0

    def fit(self, df: pd.DataFrame) -> "CoordinateFeatures":
        """Learn coordinate statistics if standardizing.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        self._check_columns(df, [self._lat_col, self._lon_col])

        if self._standardize:
            self._lat_mean = df[self._lat_col].mean()
            self._lat_std = df[self._lat_col].std()
            self._lon_mean = df[self._lon_col].mean()
            self._lon_std = df[self._lon_col].std()

            self._metadata.parameters = {
                "lat_mean": self._lat_mean,
                "lat_std": self._lat_std,
                "lon_mean": self._lon_mean,
                "lon_std": self._lon_std,
            }

        self._metadata.output_features = ["lat", "lon"]
        self._metadata.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform coordinates.

        Args:
            df: DataFrame with coordinates.

        Returns:
            DataFrame with coordinate features.
        """
        self._check_fitted()
        result = pd.DataFrame(index=df.index)

        if self._standardize:
            result["lat"] = (df[self._lat_col] - self._lat_mean) / self._lat_std
            result["lon"] = (df[self._lon_col] - self._lon_mean) / self._lon_std
        else:
            result["lat"] = df[self._lat_col]
            result["lon"] = df[self._lon_col]

        return result

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
        return ["lat", "lon"]
