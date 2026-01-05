"""Data download utilities for Italian real estate and demographic data.

Data Sources:
- OMI (Osservatorio del Mercato Immobiliare): Real estate quotations from Agenzia delle Entrate
- ISTAT: Demographic data from Italian National Institute of Statistics
- OpenPolis: Geographic boundaries (GeoJSON)
"""

import os
from pathlib import Path

import httpx
from loguru import logger
from tqdm import tqdm


# Data source URLs
# Note: onData repo uses 'master' branch, not 'main'
OMI_GITHUB_BASE = "https://raw.githubusercontent.com/ondata/quotazioni-immobiliari-agenzia-entrate/master"
OPENPOLIS_GEO_BASE = "https://raw.githubusercontent.com/openpolis/geojson-italy/master"

# ISTAT data URLs (demo.istat.it exports)
ISTAT_POPULATION_URL = "https://demo.istat.it"


def get_data_dir() -> Path:
    """Get the data directory from environment or default."""
    data_dir = os.environ.get("DATA_DIR", "./data")
    return Path(data_dir)


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """Download a file with progress bar.

    Args:
        url: Source URL
        dest_path: Destination file path
        description: Description for progress bar

    Returns:
        True if download successful, False otherwise
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return True

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(dest_path, "wb") as f:
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=description or dest_path.name,
                ) as pbar:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        logger.success(f"Downloaded: {dest_path}")
        return True

    except httpx.HTTPError as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_omi_data(data_dir: Path | None = None) -> dict[str, Path]:
    """Download OMI real estate quotation data from onData GitHub repository.

    The onData repository provides pre-processed CSV files from Agenzia delle Entrate's
    OMI database, containing semiannual real estate price quotations for all Italian
    municipalities since 2004.

    Data is stored as 7z archives. This function downloads and extracts them.

    Args:
        data_dir: Optional data directory override

    Returns:
        Dictionary mapping data type to file path
    """
    import subprocess

    data_dir = data_dir or get_data_dir()
    omi_dir = data_dir / "raw" / "omi"
    omi_dir.mkdir(parents=True, exist_ok=True)

    # The consolidated data files are stored as 7z archives
    archives = {
        "valori": "data/valori.7z",  # Main price quotations (~1M rows)
        "zone": "data/zone.7z",  # OMI zone definitions
    }

    downloaded = {}

    for name, path in archives.items():
        url = f"{OMI_GITHUB_BASE}/{path}"
        archive_path = omi_dir / f"{name}.7z"
        csv_path = omi_dir / f"{name}.csv"

        # Skip if CSV already exists
        if csv_path.exists():
            logger.info(f"CSV already exists: {csv_path}")
            downloaded[name] = csv_path
            continue

        # Download the archive
        if not archive_path.exists():
            if not download_file(url, archive_path, f"OMI {name}"):
                continue

        # Extract the archive using 7z
        logger.info(f"Extracting {archive_path}...")
        try:
            result = subprocess.run(
                ["7z", "e", "-y", f"-o{omi_dir}", str(archive_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.success(f"Extracted: {csv_path}")
                downloaded[name] = csv_path
            else:
                logger.error(f"Failed to extract {archive_path}: {result.stderr}")
        except FileNotFoundError:
            logger.error("7z not found. Please install p7zip (included in devenv).")

    return downloaded


def download_geo_data(data_dir: Path | None = None) -> dict[str, Path]:
    """Download geographic boundaries from OpenPolis.

    Downloads GeoJSON files for Italian administrative boundaries:
    - Regions (regioni)
    - Provinces (province)
    - Municipalities (comuni)

    Args:
        data_dir: Optional data directory override

    Returns:
        Dictionary mapping boundary type to file path
    """
    data_dir = data_dir or get_data_dir()
    geo_dir = data_dir / "raw" / "geo"
    geo_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "regioni": "geojson/limits_IT_regions.geojson",
        "province": "geojson/limits_IT_provinces.geojson",
        "comuni": "geojson/limits_IT_municipalities.geojson",
    }

    downloaded = {}
    for name, path in files.items():
        url = f"{OPENPOLIS_GEO_BASE}/{path}"
        dest = geo_dir / f"{name}.geojson"
        if download_file(url, dest, f"GeoJSON {name}"):
            downloaded[name] = dest

    return downloaded


def download_istat_data(data_dir: Path | None = None) -> dict[str, Path]:
    """Download ISTAT demographic data.

    NOTE: ISTAT data requires manual download or API access. This function
    provides instructions and checks for manually placed files.

    The key datasets needed are:
    - Population by municipality and age (DCIS_POPRES)
    - Demographic indicators (DCIS_INDDEMOG1)
    - Births and deaths by municipality

    For automated access, consider using the ISTAT JSON-stat API:
    http://dati.istat.it/

    Args:
        data_dir: Optional data directory override

    Returns:
        Dictionary mapping data type to file path (if files exist)
    """
    data_dir = data_dir or get_data_dir()
    istat_dir = data_dir / "raw" / "istat"
    istat_dir.mkdir(parents=True, exist_ok=True)

    # Expected files (need to be manually downloaded or fetched via API)
    expected_files = {
        "population": istat_dir / "population_by_municipality.csv",
        "demographics": istat_dir / "demographic_indicators.csv",
        "births_deaths": istat_dir / "births_deaths.csv",
    }

    found = {}
    missing = []

    for name, path in expected_files.items():
        if path.exists():
            found[name] = path
            logger.info(f"Found ISTAT data: {path}")
        else:
            missing.append(name)

    if missing:
        logger.warning(
            f"ISTAT data files not found: {missing}\n"
            "Please download manually from:\n"
            "  - http://dati.istat.it/ (I.Stat database)\n"
            "  - https://demo.istat.it/ (DEMO database)\n"
            f"And place files in: {istat_dir}"
        )

    return found


def download_all_data(data_dir: Path | None = None) -> dict[str, dict[str, Path]]:
    """Download all required data sources.

    Args:
        data_dir: Optional data directory override

    Returns:
        Nested dictionary of downloaded files by source
    """
    logger.info("Starting data download...")

    results = {
        "omi": download_omi_data(data_dir),
        "geo": download_geo_data(data_dir),
        "istat": download_istat_data(data_dir),
    }

    # Summary
    total = sum(len(files) for files in results.values())
    logger.info(f"Download complete. {total} files available.")

    return results


if __name__ == "__main__":
    # Run download when executed directly
    download_all_data()
