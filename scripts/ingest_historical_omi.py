#!/usr/bin/env python3
"""Ingest historical OMI data (2004-2024) from Agenzia delle Entrate downloads.

This script:
1. Reads all CSV files from data/raw/omi/historical/
2. Standardizes column names across different file formats
3. Extracts year and semester from filenames
4. Consolidates into a single parquet file for analysis
"""

import re
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Column name mappings for different file format versions
COLUMN_MAPPINGS = {
    # Standard columns (2016+)
    "area_territoriale": "area_territoriale",
    "regione": "regione",
    "prov": "provincia",
    "provincia": "provincia",
    "comune_istat": "comune_istat",
    "comune_cat": "comune_catastale",
    "comune_amm": "comune_amministrativo",
    "comune_descrizione": "comune_nome",
    "fascia": "fascia",
    "zona": "zona_omi",
    "cod_tip": "cod_tipologia",
    "descr_tipologia": "tipologia",
    "stato": "stato_conservazione",
    "compr_min": "prezzo_min",
    "compr_max": "prezzo_max",
    "loc_min": "affitto_min",
    "loc_max": "affitto_max",
    # Variations in older files
    "cod_comune": "comune_istat",
    "denominazione": "comune_nome",
    "tipo_immobile": "tipologia",
    "valore_min": "prezzo_min",
    "valore_max": "prezzo_max",
}


def extract_period_from_filename(filename: str) -> tuple[int, int]:
    """Extract year and semester from filename.

    Handles various naming conventions:
    - quotazioni_2024S1.csv
    - omi_20241.csv
    - valori_2024_1.csv
    - QI_XXXXX_1_20241_VALORI.csv (OMI download format)
    """
    name = filename.lower()

    # Pattern 1: OMI download format - QI_XXXXX_1_YYYYS_TYPE.csv
    # Match 5-digit pattern where first 4 are year 20xx, last is semester
    match = re.search(r'(20\d{2})([12])', name)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Pattern 2: YYYY_S format (e.g., 2024_1)
    match = re.search(r'(20\d{2})[_-]([12])', name)
    if match:
        return int(match.group(1)), int(match.group(2))

    # Pattern 3: Any other 4-digit year with semester
    match = re.search(r'(\d{4})s?(\d)', name)
    if match:
        return int(match.group(1)), int(match.group(2))

    raise ValueError(f"Cannot extract period from filename: {filename}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across different file formats."""
    # Lowercase all column names
    df.columns = df.columns.str.lower().str.strip()

    # Apply mappings
    rename_map = {}
    for old_name in df.columns:
        if old_name in COLUMN_MAPPINGS:
            rename_map[old_name] = COLUMN_MAPPINGS[old_name]

    df = df.rename(columns=rename_map)
    return df


def normalize_istat_code(code) -> str | None:
    """Convert various ISTAT code formats to 6-digit string."""
    if pd.isna(code) or code == 0:
        return None

    code_str = str(int(float(code)))

    if len(code_str) == 7:
        return code_str[1:].zfill(6)  # Drop region prefix
    elif len(code_str) == 8:
        return code_str[2:].zfill(6)  # Drop 2-digit region prefix
    elif len(code_str) == 6:
        return code_str.zfill(6)
    elif len(code_str) < 6:
        return code_str.zfill(6)
    else:
        return None


def read_omi_file(filepath: Path) -> pd.DataFrame:
    """Read a single OMI CSV file with encoding detection.

    OMI files have a metadata header on line 1, actual headers on line 2.
    They use semicolon as delimiter and Italian decimal notation (comma).
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    separators = [';', ',', '\t']
    skip_rows_options = [1, 0]  # Try skipping metadata line first

    for skip_rows in skip_rows_options:
        for encoding in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(
                        filepath,
                        encoding=encoding,
                        sep=sep,
                        skiprows=skip_rows,
                        low_memory=False,
                        decimal=','  # Italian decimal notation
                    )
                    if len(df.columns) > 5:  # Valid parse
                        return df
                except Exception:
                    continue

    raise ValueError(f"Could not parse file: {filepath}")


def ingest_historical_omi(data_dir: Path) -> pd.DataFrame:
    """Ingest all historical OMI files into a single DataFrame."""
    historical_dir = data_dir / "raw" / "omi" / "historical"

    if not historical_dir.exists():
        logger.error(f"Historical OMI directory not found: {historical_dir}")
        logger.info("Please download data from Agenzia delle Entrate and place in:")
        logger.info(f"  {historical_dir}")
        return pd.DataFrame()

    # Find all CSV files
    csv_files = list(historical_dir.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {historical_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(csv_files)} CSV files to process")

    all_data = []

    for filepath in sorted(csv_files):
        try:
            # Extract period
            year, semester = extract_period_from_filename(filepath.name)
            logger.info(f"Processing {filepath.name} ({year} S{semester})")

            # Read file
            df = read_omi_file(filepath)

            # Standardize columns
            df = standardize_columns(df)

            # Add period columns
            df["anno"] = year
            df["semestre"] = semester
            df["periodo"] = f"{year}S{semester}"

            all_data.append(df)
            logger.info(f"  Loaded {len(df):,} records")

        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}")
            continue

    if not all_data:
        logger.error("No data successfully loaded")
        return pd.DataFrame()

    # Concatenate all data
    logger.info("Concatenating all periods...")
    combined = pd.concat(all_data, ignore_index=True)

    # Normalize ISTAT codes
    if "comune_istat" in combined.columns:
        logger.info("Normalizing ISTAT codes...")
        combined["istat_code_6"] = combined["comune_istat"].apply(normalize_istat_code)

    # Convert price columns to numeric
    price_cols = ["prezzo_min", "prezzo_max", "affitto_min", "affitto_max"]
    for col in price_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Calculate midpoint prices
    if "prezzo_min" in combined.columns and "prezzo_max" in combined.columns:
        combined["prezzo_medio"] = (combined["prezzo_min"] + combined["prezzo_max"]) / 2

    logger.info(f"Total records: {len(combined):,}")
    logger.info(f"Time period: {combined['anno'].min()} - {combined['anno'].max()}")
    logger.info(f"Unique municipalities: {combined['istat_code_6'].nunique():,}")

    return combined


def save_processed_data(df: pd.DataFrame, data_dir: Path) -> Path:
    """Save processed data to parquet format."""
    output_dir = data_dir / "processed"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "omi_historical.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # Also save a summary
    summary_path = output_dir / "omi_historical_summary.csv"
    summary = df.groupby(["anno", "semestre"]).agg({
        "istat_code_6": "nunique",
        "prezzo_medio": ["mean", "median"],
    }).round(2)
    summary.columns = ["n_municipalities", "mean_price", "median_price"]
    summary.to_csv(summary_path)
    logger.info(f"Summary saved to {summary_path}")

    return output_path


def main():
    """Main entry point."""
    data_dir = Path(__file__).parent.parent / "data"

    logger.info("="*60)
    logger.info("OMI Historical Data Ingestion")
    logger.info("="*60)

    # Ingest data
    df = ingest_historical_omi(data_dir)

    if df.empty:
        logger.error("No data to process. Exiting.")
        sys.exit(1)

    # Save processed data
    output_path = save_processed_data(df, data_dir)

    logger.info("="*60)
    logger.info("Ingestion complete!")
    logger.info(f"Output: {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
