"""IRPEF income data processor.

Processes tax declaration data from the Italian Ministry of Finance.
IRPEF (Imposta sul Reddito delle Persone Fisiche) is the personal income tax,
and aggregated declaration data provides income statistics by municipality.

Data source: https://www.finanze.gov.it/it/statistiche-fiscali/
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class IRPEFProcessor:
    """Process IRPEF income declaration data.

    This processor handles:
    - Loading raw IRPEF data (CSV from MEF)
    - Standardizing municipality codes
    - Computing income metrics (average, per capita)
    - Calculating income change over time

    Example:
        >>> processor = IRPEFProcessor()
        >>> income = processor.load_and_process(data_dir)
        >>> income.columns
        ['istat_code', 'anno', 'income_avg', 'income_total', 'taxpayers', ...]
    """

    def load_raw(self, data_dir: Path, year: int | None = None) -> pd.DataFrame:
        """Load raw IRPEF data.

        Args:
            data_dir: Root data directory.
            year: Specific year to load. If None, loads latest available.

        Returns:
            Raw IRPEF DataFrame.
        """
        irpef_dir = data_dir / "raw" / "irpef"

        if not irpef_dir.exists():
            logger.warning(f"IRPEF directory not found at {irpef_dir}")
            return pd.DataFrame()

        # Find available files
        files = list(irpef_dir.glob("*.csv"))
        if not files:
            logger.warning("No IRPEF CSV files found")
            return pd.DataFrame()

        # Select file by year or use most recent
        if year:
            matching = [f for f in files if str(year) in f.name]
            if matching:
                file_path = matching[0]
            else:
                logger.warning(f"No IRPEF file found for year {year}")
                return pd.DataFrame()
        else:
            file_path = sorted(files)[-1]  # Most recent by filename

        logger.info(f"Loading IRPEF data from {file_path}")

        # IRPEF files may use various encodings and separators
        try:
            df = pd.read_csv(file_path, encoding="utf-8", sep=";")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1", sep=";")

        # Try comma separator if semicolon didn't work
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, encoding="utf-8", sep=",")

        logger.info(f"Loaded {len(df):,} IRPEF records")
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different IRPEF file formats.

        Args:
            df: Raw IRPEF DataFrame.

        Returns:
            DataFrame with standardized column names.
        """
        # Common column name mappings (Italian variations)
        column_map = {
            # Municipality code variants
            "Codice comune": "cod_comune",
            "CODICE_COMUNE": "cod_comune",
            "COD_COMUNE": "cod_comune",
            "codice_comune": "cod_comune",
            # Municipality name variants
            "Denominazione": "nome_comune",
            "DENOMINAZIONE": "nome_comune",
            "denominazione": "nome_comune",
            "Comune": "nome_comune",
            # Total income variants
            "Reddito complessivo": "reddito_totale",
            "REDDITO_COMPLESSIVO": "reddito_totale",
            "reddito_complessivo": "reddito_totale",
            "Reddito_complessivo": "reddito_totale",
            # Taxpayer count variants
            "Numero contribuenti": "n_contribuenti",
            "NUMERO_CONTRIBUENTI": "n_contribuenti",
            "numero_contribuenti": "n_contribuenti",
            "N contribuenti": "n_contribuenti",
        }

        df = df.rename(columns=column_map)
        return df

    def _standardize_istat_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize ISTAT codes to 6-digit format.

        Args:
            df: DataFrame with cod_comune column.

        Returns:
            DataFrame with standardized istat_code column.
        """
        if "cod_comune" not in df.columns:
            logger.warning("No cod_comune column found")
            return df

        df["istat_code"] = (
            df["cod_comune"]
            .astype(str)
            .str.replace(r"\D", "", regex=True)  # Remove non-digits
            .str.zfill(6)
            .str[-6:]  # Keep last 6 digits
        )
        return df

    def _parse_income_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse income values handling Italian number format.

        Args:
            df: DataFrame with reddito_totale column.

        Returns:
            DataFrame with parsed numeric columns.
        """
        if "reddito_totale" in df.columns:
            df["reddito_totale"] = pd.to_numeric(
                df["reddito_totale"]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False),
                errors="coerce",
            )

        if "n_contribuenti" in df.columns:
            df["n_contribuenti"] = pd.to_numeric(
                df["n_contribuenti"]
                .astype(str)
                .str.replace(".", "", regex=False),
                errors="coerce",
            )

        return df

    def compute_income_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived income metrics.

        Adds:
        - income_avg: Average income per taxpayer
        - income_log: Log-transformed average income

        Args:
            df: DataFrame with reddito_totale and n_contribuenti.

        Returns:
            DataFrame with income metrics added.
        """
        df = df.copy()

        if "reddito_totale" in df.columns and "n_contribuenti" in df.columns:
            # Average income per taxpayer
            df["income_avg"] = df["reddito_totale"] / df["n_contribuenti"]

            # Handle division issues
            df["income_avg"] = df["income_avg"].replace([np.inf, -np.inf], np.nan)

            # Log transform (add 1 to handle zeros)
            df["income_log"] = np.log1p(df["income_avg"].fillna(0))

        return df

    def load_multiple_years(
        self,
        data_dir: Path,
        years: list[int] | None = None,
    ) -> pd.DataFrame:
        """Load IRPEF data for multiple years.

        Args:
            data_dir: Root data directory.
            years: Years to load. If None, loads all available.

        Returns:
            Combined DataFrame with 'anno' column.
        """
        irpef_dir = data_dir / "raw" / "irpef"
        if not irpef_dir.exists():
            return pd.DataFrame()

        all_data = []
        files = list(irpef_dir.glob("*.csv"))

        for file_path in files:
            # Extract year from filename
            year_match = pd.Series(file_path.stem).str.extract(r"(\d{4})")
            if year_match[0].notna().any():
                year = int(year_match[0].iloc[0])
                if years and year not in years:
                    continue

                try:
                    df = pd.read_csv(file_path, encoding="utf-8", sep=";")
                except Exception:
                    df = pd.read_csv(file_path, encoding="latin-1", sep=";")

                df["anno"] = year
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined):,} records across {len(all_data)} years")
        return combined

    def compute_income_change(
        self,
        df: pd.DataFrame,
        base_year: int = 2015,
        end_year: int = 2022,
    ) -> pd.DataFrame:
        """Compute income change percentage between years.

        Args:
            df: DataFrame with istat_code, anno, income_avg columns.
            base_year: Starting year.
            end_year: Ending year.

        Returns:
            DataFrame with income_change_pct column.
        """
        if "anno" not in df.columns or "income_avg" not in df.columns:
            return df

        # Get base and end incomes
        base = df[df["anno"] == base_year][["istat_code", "income_avg"]]
        base = base.rename(columns={"income_avg": "income_base"})

        end = df[df["anno"] == end_year][["istat_code", "income_avg"]]
        end = end.rename(columns={"income_avg": "income_end"})

        merged = base.merge(end, on="istat_code", how="outer")
        merged["income_change_pct"] = (
            (merged["income_end"] - merged["income_base"]) / merged["income_base"] * 100
        )

        logger.info(
            f"Computed income change {base_year}-{end_year}: "
            f"mean={merged['income_change_pct'].mean():.1f}%"
        )

        return merged[["istat_code", "income_change_pct"]]

    def load_and_process(
        self,
        data_dir: Path,
        year: int | None = None,
    ) -> pd.DataFrame:
        """Full pipeline: load and process IRPEF income data.

        Args:
            data_dir: Root data directory.
            year: Specific year to process. If None, uses latest.

        Returns:
            Municipality-level DataFrame with columns:
            - istat_code: 6-digit ISTAT code
            - nome_comune: Municipality name
            - income_avg: Average income per taxpayer
            - n_contribuenti: Number of taxpayers
            - income_log: Log-transformed income
        """
        df = self.load_raw(data_dir, year=year)
        if df.empty:
            return df

        df = self._standardize_columns(df)
        df = self._standardize_istat_code(df)
        df = self._parse_income_values(df)
        df = self.compute_income_metrics(df)

        # Select output columns
        output_cols = [
            "istat_code", "nome_comune", "income_avg",
            "n_contribuenti", "income_log", "reddito_totale",
        ]
        df = df[[c for c in output_cols if c in df.columns]]

        logger.info(
            f"Processed IRPEF data: {len(df):,} municipalities, "
            f"avg income={df['income_avg'].mean():,.0f} EUR"
        )

        return df


def load_irpef_income(data_dir: Path, year: int | None = None) -> pd.DataFrame:
    """Convenience function to load processed IRPEF income data.

    Args:
        data_dir: Root data directory.
        year: Specific year to load.

    Returns:
        Processed municipality-level income DataFrame.
    """
    processor = IRPEFProcessor()
    return processor.load_and_process(data_dir, year=year)
