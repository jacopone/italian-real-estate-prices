"""ISTAT API client for downloading demographic data.

Uses the ISTAT SDMX RESTful API via the istatapi library.

API Documentation: https://developers.italia.it/en/api/istat-sdmx-rest.html
Endpoint: https://esploradati.istat.it/SDMXWS/rest

IMPORTANT: Rate limit is 5 queries per minute. Exceeding this results in
a 1-2 day IP block!

Key datasets for demographic analysis:
- DCIS_POPRES: Resident population by municipality
- DCIS_BILPOP: Population balance (births, deaths, migration)
- DCIS_INDDEMOG1: Demographic indicators
"""

import time
from pathlib import Path

import pandas as pd
from loguru import logger

# Rate limiting: ISTAT allows 5 queries per minute
RATE_LIMIT_DELAY = 15  # seconds between queries (conservative)


def get_available_datasets() -> pd.DataFrame:
    """List all available ISTAT datasets.

    Returns:
        DataFrame with dataset IDs and descriptions
    """
    try:
        from istatapi import discovery

        datasets = discovery.all_available()
        return datasets
    except ImportError:
        logger.error("istatapi not installed. Run: uv add istatapi")
        raise


def search_datasets(keyword: str) -> pd.DataFrame:
    """Search for datasets by keyword.

    Args:
        keyword: Search term (e.g., 'popolazione', 'demografici')

    Returns:
        DataFrame with matching datasets
    """
    from istatapi import discovery

    results = discovery.search_dataset(keyword)
    return results


def get_dataset_dimensions(dataset_id: str) -> dict:
    """Get the dimensions (filter options) for a dataset.

    Args:
        dataset_id: ISTAT dataset ID (e.g., 'DCIS_POPRES')

    Returns:
        Dictionary of dimension names and their possible values
    """
    from istatapi import discovery

    dimensions = discovery.dataset_dimensions(dataset_id)
    return dimensions


def download_population_data(
    data_dir: Path,
    start_year: int = 2010,
    end_year: int = 2024,
) -> Path | None:
    """Download resident population data by municipality.

    Dataset: DCIS_POPRES - Popolazione residente al 1° gennaio

    Args:
        data_dir: Directory to save the data
        start_year: Start year for data
        end_year: End year for data

    Returns:
        Path to downloaded CSV file, or None if failed
    """
    from istatapi import retrieval

    output_dir = data_dir / "raw" / "istat"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "population_by_municipality.csv"

    if output_file.exists():
        logger.info(f"Population data already exists: {output_file}")
        return output_file

    logger.info(f"Downloading ISTAT population data ({start_year}-{end_year})...")
    logger.warning(
        "This may take several minutes due to rate limiting. "
        "ISTAT allows only 5 queries/minute."
    )

    try:
        # DCIS_POPRES: Resident population on January 1st
        # Filter for total population (all ages, all sexes)
        dataset = retrieval.get_data(
            dataflow_identifier="DCIS_POPRES",
            start_period=str(start_year),
            end_period=str(end_year),
        )

        # Convert to DataFrame
        df = dataset

        if df is not None and len(df) > 0:
            df.to_csv(output_file, index=False)
            logger.success(f"Downloaded {len(df):,} rows to {output_file}")
            return output_file
        else:
            logger.warning("No data returned from ISTAT API")
            return None

    except Exception as e:
        logger.error(f"Failed to download population data: {e}")
        return None


def download_demographic_balance(
    data_dir: Path,
    start_year: int = 2010,
    end_year: int = 2024,
) -> Path | None:
    """Download demographic balance data (births, deaths, migration).

    Dataset: DCIS_BILPOP - Bilancio demografico

    Args:
        data_dir: Directory to save the data
        start_year: Start year for data
        end_year: End year for data

    Returns:
        Path to downloaded CSV file, or None if failed
    """
    from istatapi import retrieval

    output_dir = data_dir / "raw" / "istat"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "demographic_balance.csv"

    if output_file.exists():
        logger.info(f"Demographic balance data already exists: {output_file}")
        return output_file

    logger.info(f"Downloading ISTAT demographic balance ({start_year}-{end_year})...")

    # Respect rate limit
    time.sleep(RATE_LIMIT_DELAY)

    try:
        dataset = retrieval.get_data(
            dataflow_identifier="DCIS_BILPOP",
            start_period=str(start_year),
            end_period=str(end_year),
        )

        df = dataset

        if df is not None and len(df) > 0:
            df.to_csv(output_file, index=False)
            logger.success(f"Downloaded {len(df):,} rows to {output_file}")
            return output_file
        else:
            logger.warning("No data returned from ISTAT API")
            return None

    except Exception as e:
        logger.error(f"Failed to download demographic balance: {e}")
        return None


def download_all_demographic_data(
    data_dir: Path,
    start_year: int = 2010,
    end_year: int = 2024,
) -> dict[str, Path | None]:
    """Download all required demographic datasets.

    Args:
        data_dir: Directory to save the data
        start_year: Start year
        end_year: End year

    Returns:
        Dictionary mapping dataset name to file path
    """
    results = {}

    # Population data
    results["population"] = download_population_data(data_dir, start_year, end_year)

    # Wait for rate limit
    time.sleep(RATE_LIMIT_DELAY)

    # Demographic balance (births, deaths, migration)
    results["demographic_balance"] = download_demographic_balance(
        data_dir, start_year, end_year
    )

    return results


# Alternative: Direct SDMX API access for more control
def download_via_sdmx(
    dataset_id: str,
    output_path: Path,
    start_period: str = "2010",
    end_period: str = "2024",
) -> bool:
    """Download ISTAT data directly via SDMX REST API.

    This provides more control than istatapi but requires understanding
    the SDMX query structure.

    Args:
        dataset_id: ISTAT dataset ID
        output_path: Path to save CSV
        start_period: Start year
        end_period: End year

    Returns:
        True if successful
    """
    import httpx

    # ISTAT SDMX endpoint
    base_url = "https://esploradati.istat.it/SDMXWS/rest/data"

    # Build query URL
    # Format: /flowRef/key?startPeriod=X&endPeriod=Y&format=csv
    url = f"{base_url}/IT1,{dataset_id},1.0?startPeriod={start_period}&endPeriod={end_period}&format=csv"

    logger.info(f"Fetching: {url}")

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.get(url)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)

            logger.success(f"Downloaded to {output_path}")
            return True

    except httpx.HTTPError as e:
        logger.error(f"SDMX request failed: {e}")
        return False


if __name__ == "__main__":
    # Test: list available datasets
    print("Searching for population datasets...")
    datasets = search_datasets("popolazione")
    print(datasets.head(20))
