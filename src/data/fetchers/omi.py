"""OMI (Osservatorio Mercato Immobiliare) data fetcher.

Data source: https://github.com/ondata/quotazioni-immobiliari-agenzia-entrate
License: CC-BY (cite: Agenzia Entrate - OMI)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# GitHub raw content URLs
OMI_GITHUB_BASE = "https://raw.githubusercontent.com/ondata/quotazioni-immobiliari-agenzia-entrate/main"


class OMIFetcher:
    """Fetcher for OMI real estate quotation data."""

    def __init__(self, cache_dir: Path = Path("data/raw/omi")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_valori(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch OMI property values (quotazioni).

        Returns:
            DataFrame with min/max prices by zone and property type
        """
        cache_file = self.cache_dir / "valori.csv"

        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached valori from {cache_file}")
            return pd.read_csv(cache_file)

        logger.info("Fetching OMI valori...")
        url = f"{OMI_GITHUB_BASE}/data/valori.csv"

        try:
            df = pd.read_csv(url)
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(df)} valori records to {cache_file}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch valori: {e}")
            return pd.DataFrame()

    def fetch_zone(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch OMI zone definitions.

        Returns:
            DataFrame with zone geographic info
        """
        cache_file = self.cache_dir / "zone.csv"

        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached zone from {cache_file}")
            return pd.read_csv(cache_file)

        logger.info("Fetching OMI zone...")
        url = f"{OMI_GITHUB_BASE}/data/zone.csv"

        try:
            df = pd.read_csv(url)
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(df)} zone records to {cache_file}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch zone: {e}")
            return pd.DataFrame()

    def fetch_comuni(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch OMI municipality codes.

        Returns:
            DataFrame with municipality info and ISTAT codes
        """
        cache_file = self.cache_dir / "comuni.csv"

        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached comuni from {cache_file}")
            return pd.read_csv(cache_file)

        logger.info("Fetching OMI comuni...")
        url = f"{OMI_GITHUB_BASE}/data/comuni.csv"

        try:
            df = pd.read_csv(url)
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(df)} comuni records to {cache_file}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch comuni: {e}")
            return pd.DataFrame()

    def fetch_all(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch all OMI datasets.

        Returns:
            Dictionary with valori, zone, and comuni DataFrames
        """
        return {
            "valori": self.fetch_valori(force_refresh),
            "zone": self.fetch_zone(force_refresh),
            "comuni": self.fetch_comuni(force_refresh),
        }


def calculate_price_metrics(valori_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived price metrics from OMI valori.

    Args:
        valori_df: Raw valori DataFrame

    Returns:
        DataFrame with additional price metrics
    """
    df = valori_df.copy()

    # Average price (midpoint of range)
    if "Compr_min" in df.columns and "Compr_max" in df.columns:
        df["price_avg"] = (df["Compr_min"] + df["Compr_max"]) / 2
        df["price_range"] = df["Compr_max"] - df["Compr_min"]
        df["price_volatility"] = df["price_range"] / df["price_avg"]

    # Rental metrics
    if "Loc_min" in df.columns and "Loc_max" in df.columns:
        df["rent_avg"] = (df["Loc_min"] + df["Loc_max"]) / 2

        # Gross rental yield (annual rent / price)
        if "price_avg" in df.columns:
            df["gross_yield"] = (df["rent_avg"] * 12) / df["price_avg"]

    return df


def aggregate_by_municipality(
    valori_df: pd.DataFrame,
    zone_df: pd.DataFrame,
    property_type: str = "Abitazioni civili"
) -> pd.DataFrame:
    """Aggregate OMI prices by municipality.

    Args:
        valori_df: Valori with prices
        zone_df: Zone definitions
        property_type: Type of property to filter (default: residential)

    Returns:
        Municipal-level price aggregations
    """
    # Filter for property type
    filtered = valori_df[valori_df["Descr_Tipologia"] == property_type].copy()

    if filtered.empty:
        logger.warning(f"No data for property type: {property_type}")
        return pd.DataFrame()

    # valori_df already contains Comune_ISTAT, Comune_descrizione, Regione, Prov
    # No need to merge with zone_df
    merged = filtered

    # Aggregate by municipality
    agg = merged.groupby(["Comune_ISTAT", "Comune_descrizione", "Regione", "Prov"]).agg(
        price_min=("Compr_min", "min"),
        price_max=("Compr_max", "max"),
        price_avg=("Compr_min", "mean"),
        compr_max_avg=("Compr_max", "mean"),
        rent_min_avg=("Loc_min", "mean"),
        rent_max_avg=("Loc_max", "mean"),
        zone_count=("LinkZona", "nunique"),
    ).reset_index()

    # Calculate averages properly
    agg["price_avg"] = (agg["price_avg"] + agg["compr_max_avg"]) / 2
    agg["rent_avg"] = (agg["rent_min_avg"] + agg["rent_max_avg"]) / 2
    agg = agg.drop(columns=["compr_max_avg", "rent_min_avg", "rent_max_avg"])

    # Calculate yield
    agg["gross_yield"] = (agg["rent_avg"] * 12) / agg["price_avg"]

    return agg
