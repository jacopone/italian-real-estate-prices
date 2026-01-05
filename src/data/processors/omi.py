"""OMI (Osservatorio Mercato Immobiliare) data processor.

Processes real estate quotation data from Agenzia delle Entrate.
OMI provides semi-annual price ranges (min/max) for different property types
and zones within each Italian municipality.

Data source: https://github.com/ondata/quotazioni-immobiliari-agenzia-entrate
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from src.config import OMIConfig
from src.utils.constants import OMI_ZONE_HIERARCHY, RESIDENTIAL_TYPES


class OMIProcessor:
    """Process OMI real estate quotation data.

    This processor handles:
    - Loading raw OMI data (valori.csv)
    - Filtering by property type and zone
    - Aggregating zone-level data to municipality level
    - Computing price metrics (midpoint, range, volatility)
    - Extracting rental data

    Example:
        >>> processor = OMIProcessor(config.omi)
        >>> prices = processor.load_and_process(data_dir)
        >>> prices.columns
        ['istat_code', 'anno', 'prezzo_medio', 'affitto_medio', ...]
    """

    def __init__(self, config: OMIConfig | None = None):
        """Initialize processor with configuration.

        Args:
            config: OMI configuration. Uses defaults if None.
        """
        self.config = config or OMIConfig()

    def load_raw(self, data_dir: Path) -> pd.DataFrame:
        """Load raw OMI valori.csv data.

        Args:
            data_dir: Root data directory.

        Returns:
            Raw DataFrame with all OMI records.

        Raises:
            FileNotFoundError: If valori.csv doesn't exist.
        """
        path = data_dir / "raw" / "omi" / "valori.csv"
        if not path.exists():
            raise FileNotFoundError(f"OMI valori.csv not found at {path}")

        logger.info(f"Loading OMI data from {path}")

        df = pd.read_csv(
            path,
            sep=";",
            encoding="utf-8",
            dtype={
                "Comune_ISTAT": str,
                "Prov": str,
                "Fascia": str,
                "Zona": str,
            },
            low_memory=False,
        )

        logger.info(f"Loaded {len(df):,} raw OMI records")
        return df

    def _parse_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse Italian-format numeric columns.

        Italian CSV uses comma as decimal separator (1.234,56).

        Args:
            df: DataFrame with string numeric columns.

        Returns:
            DataFrame with parsed float columns.
        """
        numeric_cols = ["Compr_min", "Compr_max", "Loc_min", "Loc_max"]

        for col in numeric_cols:
            if col in df.columns:
                # Handle Italian decimal format (comma -> dot)
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(".", "", regex=False)  # Remove thousand separator
                    .str.replace(",", ".", regex=False)  # Decimal comma to dot
                    .str.replace("[^0-9.]", "", regex=True)  # Remove non-numeric
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _extract_year_semester(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract year and semester from the file column.

        The 'file' column contains strings like 'QI_294577_1_20182_VALORI_utf8.csv'
        where '20182' means 2018 semester 2.

        Args:
            df: DataFrame with 'file' column.

        Returns:
            DataFrame with 'anno' and 'semestre' columns.
        """
        if "file" not in df.columns:
            logger.warning("No 'file' column found, cannot extract year")
            return df

        # Extract year-semester pattern (e.g., '20182' from filename)
        df["year_sem"] = df["file"].str.extract(r"_(\d{5})_")[0]
        df["anno"] = df["year_sem"].str[:4].astype(float).astype("Int64")
        df["semestre"] = df["year_sem"].str[4:].astype(float).astype("Int64")

        return df.drop(columns=["year_sem"], errors="ignore")

    def _standardize_istat_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize ISTAT codes to 6-digit format.

        OMI uses 7-digit codes (province prefix + municipality).
        We extract the last 6 digits for consistency with other sources.

        Args:
            df: DataFrame with Comune_ISTAT column.

        Returns:
            DataFrame with standardized istat_code column.
        """
        # Extract last 6 digits (remove leading region digit)
        df["istat_code"] = (
            df["Comune_ISTAT"]
            .astype(str)
            .str.zfill(7)
            .str[-6:]
        )
        return df

    def filter_residential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to residential property types.

        Args:
            df: Full OMI DataFrame.

        Returns:
            DataFrame with only residential properties.
        """
        # Use configured property types or default residential types
        if self.config.property_types:
            mask = df["Descr_Tipologia"].isin(self.config.property_types)
        else:
            mask = df["Cod_Tip"].isin(RESIDENTIAL_TYPES)

        filtered = df[mask].copy()
        logger.info(f"Filtered to {len(filtered):,} residential records")
        return filtered

    def filter_year_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to configured year range.

        Args:
            df: DataFrame with 'anno' column.

        Returns:
            Filtered DataFrame.
        """
        mask = (df["anno"] >= self.config.min_year) & (df["anno"] <= self.config.max_year)
        filtered = df[mask].copy()
        logger.info(
            f"Filtered to years {self.config.min_year}-{self.config.max_year}: "
            f"{len(filtered):,} records"
        )
        return filtered

    def aggregate_to_municipality(
        self,
        df: pd.DataFrame,
        aggregation: Literal["mean", "best_zone", "weighted"] = "mean",
    ) -> pd.DataFrame:
        """Aggregate zone-level data to municipality-year level.

        OMI provides prices for multiple zones within each municipality.
        This aggregates to a single price per municipality-year.

        Args:
            df: Zone-level DataFrame.
            aggregation: Aggregation method:
                - "mean": Simple average across zones
                - "best_zone": Use most central zone (B > C > D > E > R)
                - "weighted": Weight by zone centrality

        Returns:
            Municipality-year level DataFrame.
        """
        # Compute price midpoints first
        df = df.copy()
        df["price_mid"] = (df["Compr_min"] + df["Compr_max"]) / 2
        df["rent_mid"] = (df["Loc_min"] + df["Loc_max"]) / 2

        # Filter out invalid prices
        df = df[
            (df["price_mid"] >= self.config.min_price_sqm)
            & (df["price_mid"] <= self.config.max_price_sqm)
        ]

        if aggregation == "best_zone":
            # Select most central zone per municipality-year
            df["zone_rank"] = df["Fascia"].apply(
                lambda x: OMI_ZONE_HIERARCHY.index(x)
                if x in OMI_ZONE_HIERARCHY
                else len(OMI_ZONE_HIERARCHY)
            )
            idx = df.groupby(["istat_code", "anno"])["zone_rank"].idxmin()
            result = df.loc[idx]

        elif aggregation == "weighted":
            # Weight by inverse zone rank (central zones weighted more)
            df["zone_weight"] = df["Fascia"].apply(
                lambda x: 1.0 / (OMI_ZONE_HIERARCHY.index(x) + 1)
                if x in OMI_ZONE_HIERARCHY
                else 0.1
            )
            grouped = df.groupby(["istat_code", "anno"])
            result = grouped.apply(
                lambda g: pd.Series({
                    "prezzo_medio": np.average(
                        g["price_mid"], weights=g["zone_weight"]
                    ),
                    "affitto_medio": np.average(
                        g["rent_mid"].dropna(),
                        weights=g.loc[g["rent_mid"].notna(), "zone_weight"],
                    )
                    if g["rent_mid"].notna().any()
                    else np.nan,
                    "n_zones": len(g),
                })
            ).reset_index()

        else:  # mean
            result = (
                df.groupby(["istat_code", "anno"])
                .agg(
                    prezzo_medio=("price_mid", "mean"),
                    affitto_medio=("rent_mid", "mean"),
                    prezzo_min=("Compr_min", "min"),
                    prezzo_max=("Compr_max", "max"),
                    n_zones=("price_mid", "count"),
                )
                .reset_index()
            )

        logger.info(
            f"Aggregated to {len(result):,} municipality-year observations "
            f"using '{aggregation}' method"
        )
        return result

    def load_and_process(
        self,
        data_dir: Path,
        aggregation: Literal["mean", "best_zone", "weighted"] = "mean",
    ) -> pd.DataFrame:
        """Full pipeline: load, filter, and aggregate OMI data.

        Args:
            data_dir: Root data directory.
            aggregation: Aggregation method for zones.

        Returns:
            Municipality-year level price DataFrame with columns:
            - istat_code: 6-digit ISTAT code
            - anno: Year
            - prezzo_medio: Average price EUR/sqm
            - affitto_medio: Average rent EUR/sqm/month
            - n_zones: Number of zones aggregated
        """
        # Load
        df = self.load_raw(data_dir)

        # Parse numerics
        df = self._parse_numeric_columns(df)

        # Extract year
        df = self._extract_year_semester(df)

        # Standardize ISTAT code
        df = self._standardize_istat_code(df)

        # Filter
        df = self.filter_residential(df)
        df = self.filter_year_range(df)

        # Aggregate
        result = self.aggregate_to_municipality(df, aggregation=aggregation)

        logger.info(
            f"Processed OMI data: {result['istat_code'].nunique():,} municipalities, "
            f"{result['anno'].nunique()} years"
        )

        return result

    def get_price_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate year-over-year price changes.

        Args:
            df: Municipality-year DataFrame with prezzo_medio.

        Returns:
            DataFrame with price_change_pct column added.
        """
        df = df.sort_values(["istat_code", "anno"])
        df["price_change_pct"] = df.groupby("istat_code")["prezzo_medio"].pct_change() * 100
        return df


def load_omi_prices(data_dir: Path, config: OMIConfig | None = None) -> pd.DataFrame:
    """Convenience function to load processed OMI price data.

    Args:
        data_dir: Root data directory.
        config: Optional OMI configuration.

    Returns:
        Processed municipality-year price DataFrame.
    """
    processor = OMIProcessor(config)
    return processor.load_and_process(data_dir)
