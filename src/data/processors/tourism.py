"""Tourism statistics data processor.

Processes tourism data including arrivals, stays, and tourism intensity.
Tourism intensity (arrivals per 1000 residents) is a key predictor of
real estate prices, especially in tourist-heavy areas.

Data sources:
- ISTAT tourism statistics
- Regional tourism observatories
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class TourismProcessor:
    """Process tourism statistics data.

    This processor handles:
    - Loading tourism arrivals/stays data
    - Computing tourism intensity metrics
    - Merging with population data for per-capita measures

    Example:
        >>> processor = TourismProcessor()
        >>> tourism = processor.load_and_process(data_dir)
        >>> tourism.columns
        ['prov_code', 'anno', 'arrivals', 'nights', 'tourism_intensity', ...]
    """

    def load_raw(self, data_dir: Path) -> pd.DataFrame:
        """Load raw tourism data.

        Args:
            data_dir: Root data directory.

        Returns:
            Raw tourism DataFrame.
        """
        # Check multiple possible locations
        possible_paths = [
            data_dir / "raw" / "tourism" / "tourism_by_province.csv",
            data_dir / "processed" / "tourism_intensity_by_province.csv",
            data_dir / "raw" / "istat" / "tourism_arrivals.csv",
        ]

        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading tourism data from {path}")
                df = pd.read_csv(path, encoding="utf-8")
                logger.info(f"Loaded {len(df):,} tourism records")
                return df

        logger.warning("No tourism data files found")
        return pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names.

        Args:
            df: Raw tourism DataFrame.

        Returns:
            DataFrame with standardized column names.
        """
        column_map = {
            # Province code variants
            "ITTER107": "prov_code",
            "prov_code": "prov_code",
            "COD_PROV": "prov_code",
            "cod_provincia": "prov_code",
            # Year variants
            "TIME_PERIOD": "anno",
            "anno": "anno",
            "year": "anno",
            # Arrivals variants
            "OBS_VALUE": "arrivals",
            "arrivals": "arrivals",
            "arrivi": "arrivals",
            "ARRIVI": "arrivals",
            # Nights/stays variants
            "presenze": "nights",
            "PRESENZE": "nights",
            "nights": "nights",
            # Population
            "population": "population",
            "popolazione": "population",
            # Intensity
            "tourism_intensity": "tourism_intensity",
        }

        df = df.rename(columns=column_map)
        return df

    def _standardize_province_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize province codes to 3-digit format.

        Args:
            df: DataFrame with prov_code column.

        Returns:
            DataFrame with standardized prov_code.
        """
        if "prov_code" not in df.columns:
            logger.warning("No prov_code column found")
            return df

        df["prov_code"] = (
            df["prov_code"]
            .astype(str)
            .str.replace(r"IT", "", regex=True)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(3)
        )
        return df

    def compute_tourism_intensity(
        self,
        df: pd.DataFrame,
        population_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute tourism intensity (arrivals per 1000 residents).

        Args:
            df: Tourism DataFrame with arrivals.
            population_df: Optional population data to merge.

        Returns:
            DataFrame with tourism_intensity column.
        """
        df = df.copy()

        # If population is already in the data
        if "population" in df.columns and "arrivals" in df.columns:
            df["tourism_intensity"] = (
                df["arrivals"] / df["population"] * 1000
            )
            df["tourism_intensity"] = df["tourism_intensity"].replace(
                [np.inf, -np.inf], np.nan
            )
            return df

        # If external population provided
        if population_df is not None and "arrivals" in df.columns:
            merged = df.merge(
                population_df[["prov_code", "anno", "population"]],
                on=["prov_code", "anno"],
                how="left",
            )
            merged["tourism_intensity"] = (
                merged["arrivals"] / merged["population"] * 1000
            )
            return merged

        return df

    def get_latest_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get data for the most recent year.

        Args:
            df: Tourism DataFrame with anno column.

        Returns:
            DataFrame filtered to latest year.
        """
        if "anno" not in df.columns or df.empty:
            return df

        latest_year = df["anno"].max()
        latest = df[df["anno"] == latest_year].copy()
        logger.info(f"Filtered to latest year: {latest_year} ({len(latest)} records)")
        return latest

    def aggregate_by_province(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate tourism data by province.

        Args:
            df: Tourism DataFrame (may be at municipality level).

        Returns:
            Province-level aggregated DataFrame.
        """
        if "prov_code" not in df.columns:
            return df

        agg_cols = {}
        if "arrivals" in df.columns:
            agg_cols["arrivals"] = "sum"
        if "nights" in df.columns:
            agg_cols["nights"] = "sum"
        if "population" in df.columns:
            agg_cols["population"] = "sum"

        if not agg_cols:
            return df

        group_cols = ["prov_code"]
        if "anno" in df.columns:
            group_cols.append("anno")

        aggregated = df.groupby(group_cols).agg(agg_cols).reset_index()
        logger.info(f"Aggregated to {len(aggregated):,} province-year records")
        return aggregated

    def classify_tourism_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify provinces by tourism intensity level.

        Categories:
        - low: < 100 arrivals per 1000 residents
        - medium: 100-500
        - high: 500-2000
        - very_high: > 2000

        Args:
            df: DataFrame with tourism_intensity column.

        Returns:
            DataFrame with tourism_level column.
        """
        if "tourism_intensity" not in df.columns:
            return df

        df = df.copy()
        df["tourism_level"] = pd.cut(
            df["tourism_intensity"],
            bins=[0, 100, 500, 2000, np.inf],
            labels=["low", "medium", "high", "very_high"],
            include_lowest=True,
        )
        return df

    def load_and_process(
        self,
        data_dir: Path,
        year: int | None = None,
    ) -> pd.DataFrame:
        """Full pipeline: load and process tourism data.

        Args:
            data_dir: Root data directory.
            year: Specific year. If None, uses latest available.

        Returns:
            Province-level DataFrame with columns:
            - prov_code: Province code
            - anno: Year
            - arrivals: Tourist arrivals
            - nights: Tourist nights (if available)
            - tourism_intensity: Arrivals per 1000 residents
            - tourism_level: Categorical level
        """
        df = self.load_raw(data_dir)
        if df.empty:
            return df

        df = self._standardize_columns(df)
        df = self._standardize_province_code(df)

        # Compute intensity if not already present
        if "tourism_intensity" not in df.columns:
            df = self.compute_tourism_intensity(df)

        # Filter to year if specified
        if year and "anno" in df.columns:
            df = df[df["anno"] == year].copy()
        elif "anno" in df.columns:
            df = self.get_latest_year(df)

        # Classify tourism level
        df = self.classify_tourism_level(df)

        # Select output columns
        output_cols = [
            "prov_code", "anno", "arrivals", "nights",
            "population", "tourism_intensity", "tourism_level",
        ]
        df = df[[c for c in output_cols if c in df.columns]]

        if "tourism_intensity" in df.columns:
            logger.info(
                f"Processed tourism data: {len(df):,} provinces, "
                f"intensity range: {df['tourism_intensity'].min():.0f}-"
                f"{df['tourism_intensity'].max():.0f}"
            )

        return df


def load_tourism_data(data_dir: Path, year: int | None = None) -> pd.DataFrame:
    """Convenience function to load processed tourism data.

    Args:
        data_dir: Root data directory.
        year: Specific year to load.

    Returns:
        Processed province-level tourism DataFrame.
    """
    processor = TourismProcessor()
    return processor.load_and_process(data_dir, year=year)
