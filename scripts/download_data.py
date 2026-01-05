#!/usr/bin/env python3
"""CLI script to download all data sources."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from loguru import logger

from src.data.download import download_all_data, download_geo_data, download_omi_data
from src.data.istat_client import download_all_demographic_data, search_datasets

app = typer.Typer(help="Download data sources for Italian real estate analysis")


@app.command()
def all(
    data_dir: Path = typer.Option(
        Path("./data"),
        "--data-dir",
        "-d",
        help="Data directory",
    ),
) -> None:
    """Download all available data sources."""
    logger.info(f"Downloading all data to: {data_dir}")
    results = download_all_data(data_dir)

    typer.echo("\nDownload Summary:")
    for source, files in results.items():
        typer.echo(f"  {source}: {len(files)} files")


@app.command()
def omi(
    data_dir: Path = typer.Option(
        Path("./data"),
        "--data-dir",
        "-d",
        help="Data directory",
    ),
) -> None:
    """Download OMI real estate price data."""
    logger.info("Downloading OMI data...")
    files = download_omi_data(data_dir)
    typer.echo(f"Downloaded {len(files)} OMI files")


@app.command()
def geo(
    data_dir: Path = typer.Option(
        Path("./data"),
        "--data-dir",
        "-d",
        help="Data directory",
    ),
) -> None:
    """Download geographic boundaries (GeoJSON)."""
    logger.info("Downloading geographic data...")
    files = download_geo_data(data_dir)
    typer.echo(f"Downloaded {len(files)} GeoJSON files")


@app.command()
def istat(
    data_dir: Path = typer.Option(
        Path("./data"),
        "--data-dir",
        "-d",
        help="Data directory",
    ),
    start_year: int = typer.Option(
        2010,
        "--start",
        "-s",
        help="Start year for demographic data",
    ),
    end_year: int = typer.Option(
        2024,
        "--end",
        "-e",
        help="End year for demographic data",
    ),
) -> None:
    """Download ISTAT demographic data via API.

    WARNING: ISTAT has a rate limit of 5 queries/minute.
    This download may take several minutes.
    """
    logger.info(f"Downloading ISTAT demographic data ({start_year}-{end_year})...")
    logger.warning("Rate limit: 5 queries/minute. This may take a while.")

    files = download_all_demographic_data(data_dir, start_year, end_year)

    downloaded = sum(1 for f in files.values() if f is not None)
    typer.echo(f"Downloaded {downloaded}/{len(files)} ISTAT datasets")


@app.command()
def istat_search(
    keyword: str = typer.Argument(..., help="Keyword to search for"),
) -> None:
    """Search for ISTAT datasets by keyword."""
    logger.info(f"Searching ISTAT datasets for: {keyword}")
    results = search_datasets(keyword)
    typer.echo(results.to_string())


if __name__ == "__main__":
    app()
