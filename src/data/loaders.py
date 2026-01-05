"""Data loading utilities for OMI, ISTAT, and geographic data."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger


def load_omi_valori(data_dir: Path) -> pd.DataFrame:
    """Load OMI property valuations data.

    The valori.csv contains real estate price quotations:
    - Price ranges (min/max) in EUR/sqm for purchase
    - Rental ranges in EUR/sqm/month
    - By municipality, OMI zone, property type, and semester

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with OMI valuations
    """
    path = data_dir / "raw" / "omi" / "valori.csv"
    if not path.exists():
        raise FileNotFoundError(f"OMI valori.csv not found at {path}")

    logger.info(f"Loading OMI valuations from {path}")

    # OMI CSV uses semicolon separator and Italian decimal format
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        dtype={
            "cod_regione": str,
            "cod_provincia": str,
            "cod_comune": str,
            "cod_zona": str,
        },
    )

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Parse numeric columns (Italian format uses comma as decimal separator)
    numeric_cols = ["val_min", "val_max", "sup_loc_min", "sup_loc_max"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".")
                .str.replace("[^0-9.]", "", regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Loaded {len(df):,} OMI valuation records")
    return df


def load_omi_zone(data_dir: Path) -> pd.DataFrame:
    """Load OMI zone definitions.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with zone metadata
    """
    path = data_dir / "raw" / "omi" / "zone.csv"
    if not path.exists():
        raise FileNotFoundError(f"OMI zone.csv not found at {path}")

    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df.columns = df.columns.str.lower().str.strip()
    logger.info(f"Loaded {len(df):,} OMI zone records")
    return df


def load_omi_comuni(data_dir: Path) -> pd.DataFrame:
    """Load OMI municipality mappings.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with municipality codes and names
    """
    path = data_dir / "raw" / "omi" / "comuni.csv"
    if not path.exists():
        raise FileNotFoundError(f"OMI comuni.csv not found at {path}")

    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df.columns = df.columns.str.lower().str.strip()
    logger.info(f"Loaded {len(df):,} OMI municipality records")
    return df


def load_geo_boundaries(
    data_dir: Path, level: str = "comuni"
) -> gpd.GeoDataFrame:
    """Load geographic boundaries as GeoDataFrame.

    Args:
        data_dir: Path to data directory
        level: Administrative level ('regioni', 'province', 'comuni')

    Returns:
        GeoDataFrame with administrative boundaries
    """
    valid_levels = ["regioni", "province", "comuni"]
    if level not in valid_levels:
        raise ValueError(f"level must be one of {valid_levels}")

    path = data_dir / "raw" / "geo" / f"{level}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found at {path}")

    logger.info(f"Loading {level} boundaries from {path}")
    gdf = gpd.read_file(path)
    logger.info(f"Loaded {len(gdf):,} {level} boundaries")
    return gdf


def load_istat_population(data_dir: Path) -> pd.DataFrame:
    """Load ISTAT population data.

    Expected columns:
    - cod_comune: ISTAT municipality code
    - anno: Year
    - popolazione: Total population
    - eta_0_14, eta_15_64, eta_65_plus: Age groups

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with population data
    """
    path = data_dir / "raw" / "istat" / "population_by_municipality.csv"
    if not path.exists():
        logger.warning(f"ISTAT population data not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.lower().str.strip()
    logger.info(f"Loaded {len(df):,} ISTAT population records")
    return df


def load_municipality_metadata(data_dir: Path) -> pd.DataFrame:
    """Load municipality metadata including coordinates, province, and region.

    Uses the main.csv file from opendatasicilia/comuni-italiani which contains:
    - comune: Municipality name
    - pro_com_t: ISTAT code (6 digits)
    - lat, long: Coordinates
    - den_prov, sigla: Province name and abbreviation
    - den_reg, cod_reg: Region name and code
    - cap: Postal code
    - And more (PEC, mail, sito_web, wikipedia, stemma)

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with municipality metadata
    """
    path = data_dir / "raw" / "istat" / "main.csv"
    if not path.exists():
        # Fallback to comuni.csv
        path = data_dir / "raw" / "istat" / "comuni.csv"
        if not path.exists():
            logger.warning("Municipality metadata not found")
            return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")
    # Ensure ISTAT code is string with leading zeros
    df["pro_com_t"] = df["pro_com_t"].astype(str).str.zfill(6)
    logger.info(f"Loaded {len(df):,} municipality records")
    return df


def load_population_trends(data_dir: Path) -> pd.DataFrame:
    """Load population trend data (2018-2021) by municipality.

    Returns DataFrame with columns:
    - comune: Municipality name
    - pro_com_t: ISTAT code
    - pop_res_18 through pop_res_21: Population by year

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with population trends
    """
    path = data_dir / "raw" / "istat" / "population_trends_2018_2021.csv"
    if not path.exists():
        logger.warning(f"Population trends not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")
    df["pro_com_t"] = df["pro_com_t"].astype(str).str.zfill(6)
    logger.info(f"Loaded population trends for {len(df):,} municipalities")
    return df


def load_population_2021(data_dir: Path) -> pd.DataFrame:
    """Load 2021 population data with basic age breakdown.

    Returns DataFrame with columns:
    - pro_com_t: ISTAT code
    - pop_under_12: Population under 12
    - pop_12_plus: Population 12 and over
    - pop_total: Total population

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with 2021 population
    """
    path = data_dir / "raw" / "istat" / "istat_population_2021.csv"
    if not path.exists():
        logger.warning(f"Population 2021 not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={
        "<12": "pop_under_12",
        ">=12": "pop_12_plus",
        "totale": "pop_total",
    })
    df["pro_com_t"] = df["pro_com_t"].astype(str).str.zfill(6)
    logger.info(f"Loaded 2021 population for {len(df):,} municipalities")
    return df


def load_demographic_balance(data_dir: Path) -> pd.DataFrame:
    """Load demographic balance data (births, deaths, migration).

    Currently available at regional/provincial level (NUTS3).
    DATA_TYPE codes:
    - FLBIRTH: Live births
    - FDEATH: Deaths
    - FNATGR: Natural growth (births - deaths)
    - FINTNMIG: Internal migration
    - FINTRNMIGR: International migration
    - ACQCITIZ: Citizenship acquisitions

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with demographic balance indicators
    """
    path = data_dir / "raw" / "istat" / "demographic_balance_regional.csv"
    if not path.exists():
        logger.warning(f"Demographic balance not found at {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="utf-8")
    # Pivot to have indicators as columns
    pivot_df = df.pivot_table(
        index=["REF_AREA", "TIME_PERIOD", "SEX"],
        columns="DATA_TYPE",
        values="OBS_VALUE",
        aggfunc="first",
    ).reset_index()
    pivot_df.columns.name = None
    logger.info(f"Loaded demographic balance: {len(pivot_df):,} records")
    return pivot_df


def load_all_istat_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all available ISTAT data into a dictionary.

    Returns:
        Dictionary with keys: 'metadata', 'population_trends',
        'population_2021', 'demographic_balance'
    """
    return {
        "metadata": load_municipality_metadata(data_dir),
        "population_trends": load_population_trends(data_dir),
        "population_2021": load_population_2021(data_dir),
        "demographic_balance": load_demographic_balance(data_dir),
    }


def calculate_price_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate price metrics from OMI data.

    Adds:
    - price_mid: midpoint of min/max range
    - price_range: max - min
    - price_volatility: range / mid

    Args:
        df: OMI valuations DataFrame

    Returns:
        DataFrame with added price metrics
    """
    df = df.copy()

    if "val_min" in df.columns and "val_max" in df.columns:
        df["price_mid"] = (df["val_min"] + df["val_max"]) / 2
        df["price_range"] = df["val_max"] - df["val_min"]
        df["price_volatility"] = df["price_range"] / df["price_mid"]

    return df
