"""Prepare supplementary data from alternative sources.

This module processes data from Eurostat, OpenDataSicilia, and other sources
when ISTAT APIs are unavailable.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# NUTS2 to Italian Region mapping
NUTS2_TO_REGION = {
    "ITC1": "PIEMONTE",
    "ITC2": "VALLE D'AOSTA/VALLEE D'AOSTE",
    "ITC3": "LIGURIA",
    "ITC4": "LOMBARDIA",
    "ITH1": "TRENTINO-ALTO ADIGE",  # Bolzano
    "ITH2": "TRENTINO-ALTO ADIGE",  # Trento
    "ITH3": "VENETO",
    "ITH4": "FRIULI-VENEZIA GIULIA",
    "ITH5": "EMILIA-ROMAGNA",
    "ITI1": "TOSCANA",
    "ITI2": "UMBRIA",
    "ITI3": "MARCHE",
    "ITI4": "LAZIO",
    "ITF1": "ABRUZZO",
    "ITF2": "MOLISE",
    "ITF3": "CAMPANIA",
    "ITF4": "PUGLIA",
    "ITF5": "BASILICATA",
    "ITF6": "CALABRIA",
    "ITG1": "SICILIA",
    "ITG2": "SARDEGNA",
}

# Region code to region name (from opendatasicilia)
REGION_CODE_TO_NAME = {
    1: "PIEMONTE",
    2: "VALLE D'AOSTA/VALLEE D'AOSTE",
    3: "LOMBARDIA",
    4: "TRENTINO-ALTO ADIGE",
    5: "VENETO",
    6: "FRIULI-VENEZIA GIULIA",
    7: "LIGURIA",
    8: "EMILIA-ROMAGNA",
    9: "TOSCANA",
    10: "UMBRIA",
    11: "MARCHE",
    12: "LAZIO",
    13: "ABRUZZO",
    14: "MOLISE",
    15: "CAMPANIA",
    16: "PUGLIA",
    17: "BASILICATA",
    18: "CALABRIA",
    19: "SICILIA",
    20: "SARDEGNA",
}


def prepare_population_data(data_dir: Path) -> pd.DataFrame:
    """Prepare population data with 10-year change calculation.

    Uses 2011 census (from comuni-json) and 2021 data (from opendatasicilia).

    Returns:
        DataFrame with columns: Comune_ISTAT, pop_2011, pop_2021, pop_10yr_change
    """
    logger.info("Preparing population data...")

    # Load 2021 population
    pop_2021_file = data_dir / "raw" / "istat" / "popolazione_2021.csv"
    if not pop_2021_file.exists():
        logger.error(f"Missing {pop_2021_file}")
        return pd.DataFrame()

    pop_2021 = pd.read_csv(pop_2021_file)
    pop_2021 = pop_2021.rename(columns={"pro_com_t": "Comune_ISTAT", "pop_res_21": "pop_2021"})

    # Convert to 6-digit string format (e.g., 1001 -> 001001)
    pop_2021["Comune_ISTAT"] = pop_2021["Comune_ISTAT"].astype(str).str.zfill(6)

    # Load 2011 census population
    pop_2011_file = data_dir / "raw" / "istat" / "comuni_2011.json"
    if not pop_2011_file.exists():
        logger.warning(f"Missing {pop_2011_file}, cannot calculate population change")
        pop_2021["pop_2011"] = None
        pop_2021["pop_10yr_change"] = 0
        return pop_2021

    with open(pop_2011_file) as f:
        comuni_2011 = json.load(f)

    # Extract population from 2011 census
    pop_2011_data = []
    for comune in comuni_2011:
        pop_2011_data.append({
            "Comune_ISTAT": comune["codice"],
            "pop_2011": comune.get("popolazione", 0),
            "comune_nome": comune["nome"],
            "regione": comune.get("regione", {}).get("nome", ""),
        })

    pop_2011 = pd.DataFrame(pop_2011_data)

    # Merge 2011 and 2021 data
    result = pop_2021.merge(pop_2011, on="Comune_ISTAT", how="left")

    # Calculate 10-year change
    result["pop_10yr_change"] = (
        (result["pop_2021"] - result["pop_2011"]) / result["pop_2011"]
    ).fillna(0)

    # Flag depopulating areas (> 10% decline)
    result["is_depopulating"] = result["pop_10yr_change"] < -0.10

    logger.info(f"Prepared population data for {len(result)} municipalities")
    logger.info(f"  Average 10yr change: {result['pop_10yr_change'].mean():.1%}")
    logger.info(f"  Depopulating areas: {result['is_depopulating'].sum()} ({result['is_depopulating'].mean():.1%})")

    return result


def prepare_regional_tourism(data_dir: Path) -> pd.DataFrame:
    """Prepare regional tourism intensity from Eurostat data.

    Returns:
        DataFrame with columns: Regione, presenze_2023, tourism_intensity
    """
    logger.info("Preparing regional tourism data...")

    tourism_file = data_dir / "raw" / "tourism" / "eurostat_regional.csv"
    if not tourism_file.exists():
        logger.error(f"Missing {tourism_file}")
        return pd.DataFrame()

    df = pd.read_csv(tourism_file)

    # Filter for Italian NUTS2 regions, total (DOM+FOR), 2023, hotels (I551)
    # TOUR_OCC_NIN2 = Nights spent at tourist accommodation establishments
    italian_df = df[
        (df["geo"].str.startswith("IT")) &
        (df["geo"].str.len() == 4) &  # NUTS2 level (e.g., ITC1)
        (df["TIME_PERIOD"] == 2023) &
        (df["c_resid"] == "TOTAL") &
        (df["nace_r2"] == "I551-I553")  # All accommodation types
    ].copy()

    if italian_df.empty:
        # Try without c_resid filter
        italian_df = df[
            (df["geo"].str.startswith("IT")) &
            (df["geo"].str.len() == 4) &
            (df["TIME_PERIOD"] == 2023) &
            (df["nace_r2"] == "I551-I553")
        ].copy()

    if italian_df.empty:
        logger.warning("No Italian tourism data found in Eurostat file")
        return pd.DataFrame()

    # Map NUTS2 to region names
    italian_df["Regione"] = italian_df["geo"].map(NUTS2_TO_REGION)

    # Aggregate by region (some regions have multiple NUTS2, like Trentino)
    regional = italian_df.groupby("Regione").agg(
        presenze_2023=("OBS_VALUE", "sum")
    ).reset_index()

    # Load regional population for intensity calculation
    # Use 2021 regional population estimates
    regional_pop = {
        "PIEMONTE": 4256350,
        "VALLE D'AOSTA/VALLEE D'AOSTE": 123360,
        "LOMBARDIA": 10027602,
        "TRENTINO-ALTO ADIGE": 1078069,
        "VENETO": 4847745,
        "FRIULI-VENEZIA GIULIA": 1194647,
        "LIGURIA": 1509805,
        "EMILIA-ROMAGNA": 4438937,
        "TOSCANA": 3668333,
        "UMBRIA": 865013,
        "MARCHE": 1487150,
        "LAZIO": 5714882,
        "ABRUZZO": 1275950,
        "MOLISE": 294294,
        "CAMPANIA": 5624260,
        "PUGLIA": 3900852,
        "BASILICATA": 545130,
        "CALABRIA": 1860601,
        "SICILIA": 4833705,
        "SARDEGNA": 1587413,
    }

    regional["popolazione"] = regional["Regione"].map(regional_pop)
    regional["tourism_intensity"] = regional["presenze_2023"] / regional["popolazione"]

    # Classify tourism intensity
    # High: > 20 nights per capita (tourist regions)
    # Medium: 5-20 nights per capita
    # Low: < 5 nights per capita
    regional["tourism_level"] = pd.cut(
        regional["tourism_intensity"],
        bins=[0, 5, 20, float("inf")],
        labels=["low", "medium", "high"]
    )

    logger.info(f"Prepared tourism data for {len(regional)} regions")
    logger.info(f"  High tourism regions: {(regional['tourism_level'] == 'high').sum()}")
    logger.info(f"  Medium tourism regions: {(regional['tourism_level'] == 'medium').sum()}")

    return regional


def prepare_municipal_supplementary_data(data_dir: Path) -> pd.DataFrame:
    """Prepare complete supplementary dataset at municipal level.

    Combines:
    - Population change (2011-2021)
    - Regional tourism intensity
    - Regional vacancy rates

    Returns:
        DataFrame ready to merge with OMI price data
    """
    logger.info("Preparing complete municipal supplementary data...")

    # Load comuni anagrafica for region mapping
    anagrafica_file = data_dir / "raw" / "istat" / "comuni_anagrafica.csv"
    if not anagrafica_file.exists():
        logger.error(f"Missing {anagrafica_file}")
        return pd.DataFrame()

    anagrafica = pd.read_csv(anagrafica_file)
    anagrafica["Comune_ISTAT"] = anagrafica["pro_com_t"].astype(str).str.zfill(6)
    anagrafica["Regione"] = anagrafica["den_reg"].str.upper()

    # Fix Valle d'Aosta naming
    anagrafica["Regione"] = anagrafica["Regione"].replace(
        "VALLE D'AOSTA", "VALLE D'AOSTA/VALLEE D'AOSTE"
    )

    # Get population data
    pop_data = prepare_population_data(data_dir)

    # Get tourism data
    tourism_data = prepare_regional_tourism(data_dir)

    # Get vacancy data
    vacancy_file = data_dir / "raw" / "istat" / "regional_vacancy_2021.csv"
    vacancy_data = pd.DataFrame()
    if vacancy_file.exists():
        vacancy_data = pd.read_csv(vacancy_file)

    # Start with anagrafica
    result = anagrafica[["Comune_ISTAT", "comune", "den_prov", "sigla", "Regione"]].copy()
    result = result.rename(columns={
        "comune": "Comune_descrizione",
        "den_prov": "Provincia",
        "sigla": "Prov"
    })

    # Merge population data
    if not pop_data.empty:
        pop_cols = ["Comune_ISTAT", "pop_2011", "pop_2021", "pop_10yr_change", "is_depopulating"]
        pop_subset = pop_data[[c for c in pop_cols if c in pop_data.columns]]
        result = result.merge(pop_subset, on="Comune_ISTAT", how="left")

    # Merge regional tourism
    if not tourism_data.empty:
        tourism_cols = ["Regione", "presenze_2023", "tourism_intensity", "tourism_level"]
        tourism_subset = tourism_data[[c for c in tourism_cols if c in tourism_data.columns]]
        result = result.merge(tourism_subset, on="Regione", how="left")

    # Merge regional vacancy
    if not vacancy_data.empty:
        vacancy_cols = ["Regione", "vacancy_rate"]
        vacancy_subset = vacancy_data[[c for c in vacancy_cols if c in vacancy_data.columns]]
        result = result.merge(vacancy_subset, on="Regione", how="left")

    # Fill missing values
    result["pop_10yr_change"] = result.get("pop_10yr_change", pd.Series([0])).fillna(0)
    result["tourism_intensity"] = result.get("tourism_intensity", pd.Series([5])).fillna(5)
    result["vacancy_rate"] = result.get("vacancy_rate", pd.Series([0.272])).fillna(0.272)
    result["is_depopulating"] = result.get("is_depopulating", pd.Series([False])).fillna(False)

    # Save to processed
    output_file = data_dir / "processed" / "municipal_supplementary.csv"
    result.to_csv(output_file, index=False)
    logger.info(f"Saved supplementary data to {output_file}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    result = prepare_municipal_supplementary_data(data_dir)
    print(f"\nPrepared {len(result)} municipalities")
    print("\nSample data:")
    print(result.head(10))
