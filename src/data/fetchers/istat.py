"""ISTAT data fetcher for housing, demographics, and tourism data.

Uses the istatapi library for SDMX REST API access.
See: https://github.com/Attol8/istatapi
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ISTAT SDMX API endpoints
# Primary endpoint for SDMX REST API
ISTAT_BASE_URL = "https://sdmx.istat.it/SDMXWS/rest"
# Alternative endpoint (dati.istat.it)
ISTAT_ALT_URL = "https://esploradati.istat.it/SDMXWS/rest"
# Census-specific endpoint
CENSUS_BASE_URL = "https://esploradati.censimentopopolazione.istat.it"

# Dataflow identifiers (from ISTAT API documentation)
DATAFLOWS = {
    "population": "22_289",  # DCIS_POPRES1 - Popolazione residente
    "demographics": "22_293",  # DCIS_INDDEMOG1 - Indicatori demografici
    "tourism": "122_54",  # DCSC_TUR - Capacità esercizi ricettivi e movimento clienti
    "pop_balance": "22_315",  # DCIS_POPORESBIL1 - Bilancio demografico
}


class ISTATFetcher:
    """Fetcher for ISTAT statistical data."""

    def __init__(self, cache_dir: Path = Path("data/raw/istat")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_housing_census(self, year: int = 2021) -> pd.DataFrame:
        """Fetch housing occupancy data from ISTAT Census.

        Returns data on occupied vs non-occupied dwellings by region/province.

        Args:
            year: Census year (2021 is latest permanent census)

        Returns:
            DataFrame with columns:
            - territorio: Region/Province code
            - territorio_nome: Region/Province name
            - abitazioni_totali: Total dwellings
            - abitazioni_occupate: Occupied dwellings
            - abitazioni_non_occupate: Non-occupied dwellings
            - vacancy_rate: Non-occupied / Total
        """
        cache_file = self.cache_dir / f"housing_census_{year}.csv"

        if cache_file.exists():
            logger.info(f"Loading cached housing census data from {cache_file}")
            return pd.read_csv(cache_file)

        logger.info(f"Fetching housing census data for {year}...")

        # ISTAT Census dataset codes
        # DCSS_ABITAZIONI: Dwellings by occupancy status (regional/provincial)
        # The data structure uses SDMX format

        df = pd.DataFrame()
        try:
            # Try using istatapi if available
            from istatapi import discovery, retrieval

            ds = discovery.DataSet(dataflow_identifier="DCSS_ABITAZIONI")
            ds.set_filters(TIME_PERIOD=str(year))
            df = retrieval.get_data(ds)

        except ImportError:
            logger.warning("istatapi not installed. Using fallback method.")
            df = self._fetch_housing_fallback(year)
        except Exception as e:
            logger.warning(f"istatapi failed ({e}). Using fallback method.")
            df = self._fetch_housing_fallback(year)

        if df is not None and not df.empty:
            # Calculate vacancy rate
            if "abitazioni_totali" in df.columns and "abitazioni_non_occupate" in df.columns:
                df["vacancy_rate"] = df["abitazioni_non_occupate"] / df["abitazioni_totali"]

            df.to_csv(cache_file, index=False)
            logger.info(f"Cached housing census data to {cache_file}")

        return df

    def _fetch_housing_fallback(self, year: int) -> pd.DataFrame:
        """Fallback method - try multiple ISTAT data sources."""
        # Try different API endpoints
        endpoints = [
            f"{ISTAT_BASE_URL}/data/DCSS_ABITAZIONI/..?startPeriod={year}&endPeriod={year}",
            f"{ISTAT_ALT_URL}/data/DCSS_ABITAZIONI/..?startPeriod={year}&endPeriod={year}",
            # Try generic I.Stat format
            f"https://dati.istat.it/api/v2/sdmx/data/dataflow/IT1/DCSS_ABITAZIONI/1.0?startPeriod={year}&endPeriod={year}",
        ]

        for url in endpoints:
            try:
                logger.info(f"Trying housing endpoint: {url}")
                response = requests.get(
                    url,
                    headers={"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"},
                    timeout=60
                )
                if response.status_code == 200 and response.text.strip():
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    if not df.empty:
                        logger.info(f"Successfully fetched housing data from {url}")
                        return df
            except Exception as e:
                logger.debug(f"Endpoint {url} failed: {e}")
                continue

        logger.warning("All housing data endpoints failed, returning empty DataFrame")
        return pd.DataFrame()

    def fetch_demographics(
        self,
        start_year: int = 2011,
        end_year: int = 2024,
        granularity: str = "municipal"
    ) -> pd.DataFrame:
        """Fetch population dynamics data from ISTAT Demo portal.

        Args:
            start_year: Start year for population data
            end_year: End year for population data
            granularity: 'regional', 'provincial', or 'municipal'

        Returns:
            DataFrame with population and demographic indicators
        """
        cache_file = self.cache_dir / f"demographics_{start_year}_{end_year}_{granularity}.csv"

        if cache_file.exists():
            logger.info(f"Loading cached demographics data from {cache_file}")
            return pd.read_csv(cache_file)

        logger.info(f"Fetching demographics data {start_year}-{end_year}...")

        df = pd.DataFrame()
        try:
            from istatapi import discovery, retrieval

            # DCIS_POPRES: Resident population by municipality
            ds = discovery.DataSet(dataflow_identifier="DCIS_POPRES1")
            ds.set_filters(TIME_PERIOD=[str(y) for y in range(start_year, end_year + 1)])
            df = retrieval.get_data(ds)

        except ImportError:
            logger.warning("istatapi not installed. Using fallback method.")
            df = self._fetch_demographics_fallback(start_year, end_year)
        except Exception as e:
            logger.warning(f"istatapi failed ({e}). Using fallback method.")
            df = self._fetch_demographics_fallback(start_year, end_year)

        if df is not None and not df.empty:
            # Calculate population change
            df = self._calculate_population_change(df)
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached demographics data to {cache_file}")

        return df

    def _fetch_demographics_fallback(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fallback for demographics data - try multiple sources."""
        endpoints = [
            f"{ISTAT_BASE_URL}/data/{DATAFLOWS['population']}/..?startPeriod={start_year}&endPeriod={end_year}",
            f"{ISTAT_BASE_URL}/data/DCIS_POPRES1/..?startPeriod={start_year}&endPeriod={end_year}",
            f"{ISTAT_ALT_URL}/data/DCIS_POPRES1/..?startPeriod={start_year}&endPeriod={end_year}",
        ]

        for url in endpoints:
            try:
                logger.info(f"Trying demographics endpoint: {url}")
                response = requests.get(
                    url,
                    headers={"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"},
                    timeout=120
                )
                if response.status_code == 200 and response.text.strip():
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    if not df.empty:
                        logger.info(f"Successfully fetched demographics from {url}")
                        return df
            except Exception as e:
                logger.debug(f"Endpoint {url} failed: {e}")
                continue

        logger.warning("All demographics endpoints failed, returning empty DataFrame")
        return pd.DataFrame()

    def _calculate_population_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate population change metrics."""
        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            return df

        # Group by territory and calculate change
        df = df.sort_values(["ITTER107", "TIME_PERIOD"])

        # Calculate year-over-year change
        df["pop_change_abs"] = df.groupby("ITTER107")["OBS_VALUE"].diff()
        df["pop_change_pct"] = df.groupby("ITTER107")["OBS_VALUE"].pct_change()

        # Calculate 10-year change for depopulation indicator
        df["pop_10yr_change"] = df.groupby("ITTER107")["OBS_VALUE"].transform(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) > 1 else 0
        )

        return df

    def fetch_tourism(
        self,
        start_year: int = 2019,
        end_year: int = 2024
    ) -> pd.DataFrame:
        """Fetch tourism statistics (arrivals and presences).

        Args:
            start_year: Start year
            end_year: End year

        Returns:
            DataFrame with tourism metrics by region/province
        """
        cache_file = self.cache_dir / f"tourism_{start_year}_{end_year}.csv"

        if cache_file.exists():
            logger.info(f"Loading cached tourism data from {cache_file}")
            return pd.read_csv(cache_file)

        logger.info(f"Fetching tourism data {start_year}-{end_year}...")

        df = pd.DataFrame()
        try:
            from istatapi import discovery, retrieval

            # DCSC_MOVIMENTO: Tourist flows
            ds = discovery.DataSet(dataflow_identifier="DCSC_MOVIMENTO")
            ds.set_filters(TIME_PERIOD=[str(y) for y in range(start_year, end_year + 1)])
            df = retrieval.get_data(ds)

        except ImportError:
            logger.warning("istatapi not installed. Using fallback method.")
            df = self._fetch_tourism_fallback(start_year, end_year)
        except Exception as e:
            logger.warning(f"istatapi failed ({e}). Using fallback method.")
            df = self._fetch_tourism_fallback(start_year, end_year)

        if df is not None and not df.empty:
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached tourism data to {cache_file}")

        return df

    def _fetch_tourism_fallback(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Fallback for tourism data - try multiple sources."""
        endpoints = [
            f"{ISTAT_BASE_URL}/data/{DATAFLOWS['tourism']}/..?startPeriod={start_year}&endPeriod={end_year}",
            f"{ISTAT_BASE_URL}/data/DCSC_TUR/..?startPeriod={start_year}&endPeriod={end_year}",
            f"{ISTAT_BASE_URL}/data/DCSC_MOVIMENTO/..?startPeriod={start_year}&endPeriod={end_year}",
            f"{ISTAT_ALT_URL}/data/DCSC_TUR/..?startPeriod={start_year}&endPeriod={end_year}",
        ]

        for url in endpoints:
            try:
                logger.info(f"Trying tourism endpoint: {url}")
                response = requests.get(
                    url,
                    headers={"Accept": "application/vnd.sdmx.data+csv;version=1.0.0"},
                    timeout=120
                )
                if response.status_code == 200 and response.text.strip():
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text))
                    if not df.empty:
                        logger.info(f"Successfully fetched tourism from {url}")
                        return df
            except Exception as e:
                logger.debug(f"Endpoint {url} failed: {e}")
                continue

        logger.warning("All tourism endpoints failed, returning empty DataFrame")
        return pd.DataFrame()


def calculate_tourism_intensity(
    tourism_df: pd.DataFrame,
    population_df: pd.DataFrame,
    territory_col: str = "ITTER107"
) -> pd.DataFrame:
    """Calculate tourism intensity (presenze per inhabitant).

    Args:
        tourism_df: Tourism data with presenze
        population_df: Population data
        territory_col: Column name for territory code

    Returns:
        DataFrame with tourism_intensity column
    """
    # Merge tourism and population
    merged = tourism_df.merge(
        population_df[[territory_col, "OBS_VALUE"]].rename(columns={"OBS_VALUE": "population"}),
        on=territory_col,
        how="left"
    )

    # Calculate intensity
    if "presenze" in merged.columns:
        merged["tourism_intensity"] = merged["presenze"] / merged["population"]
    elif "OBS_VALUE" in merged.columns:
        merged["tourism_intensity"] = merged["OBS_VALUE"] / merged["population"]

    return merged
