"""ISTAT demographic data processor.

Processes demographic statistics from the Italian National Institute of Statistics.
Includes population, age structure, migration, and demographic indicators.

Data sources:
- Population by municipality: https://demo.istat.it
- Census data: Census 2011, 2021
- opendatasicilia/comuni-italiani for municipality metadata
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.constants import (
    NORTHERN_REGIONS,
    REGION_CODES,
    distance_to_milan,
    haversine_distance,
)


class ISTATProcessor:
    """Process ISTAT demographic data.

    This processor handles:
    - Municipality metadata (coordinates, administrative hierarchy)
    - Population data and trends
    - Age structure and demographic indicators
    - Geographic features (distances, classifications)

    Example:
        >>> processor = ISTATProcessor()
        >>> demographics = processor.load_and_process(data_dir)
        >>> demographics.columns
        ['istat_code', 'nome', 'lat', 'lon', 'population', 'pop_change_pct', ...]
    """

    def load_municipality_metadata(self, data_dir: Path) -> pd.DataFrame:
        """Load municipality metadata from opendatasicilia/comuni-italiani.

        Args:
            data_dir: Root data directory.

        Returns:
            DataFrame with coordinates and administrative info.
        """
        path = data_dir / "raw" / "istat" / "main.csv"
        if not path.exists():
            logger.warning(f"Municipality metadata not found at {path}")
            return pd.DataFrame()

        logger.info(f"Loading municipality metadata from {path}")
        df = pd.read_csv(path, encoding="utf-8")

        # Standardize ISTAT code to 6 digits
        df["istat_code"] = df["pro_com_t"].astype(str).str.zfill(6)

        # Rename for consistency
        df = df.rename(columns={
            "comune": "nome",
            "lat": "lat",
            "long": "lon",
            "den_prov": "provincia",
            "sigla": "sigla_provincia",
            "den_reg": "regione",
            "cod_reg": "cod_regione",
        })

        # Select relevant columns
        cols = [
            "istat_code", "nome", "lat", "lon",
            "provincia", "sigla_provincia", "regione", "cod_regione",
        ]
        df = df[[c for c in cols if c in df.columns]]

        logger.info(f"Loaded metadata for {len(df):,} municipalities")
        return df

    def load_population_data(self, data_dir: Path) -> pd.DataFrame:
        """Load population data from multiple sources.

        Combines:
        - Population trends (2018-2021)
        - Census population (2021)
        - Detailed age breakdown

        Args:
            data_dir: Root data directory.

        Returns:
            DataFrame with population by municipality.
        """
        # Try population trends first
        trends_path = data_dir / "raw" / "istat" / "population_trends_2018_2021.csv"
        pop_2021_path = data_dir / "raw" / "istat" / "istat_population_2021.csv"

        population_df = pd.DataFrame()

        if trends_path.exists():
            logger.info(f"Loading population trends from {trends_path}")
            trends = pd.read_csv(trends_path, encoding="utf-8")
            trends["istat_code"] = trends["pro_com_t"].astype(str).str.zfill(6)

            # Melt to long format
            pop_cols = [c for c in trends.columns if c.startswith("pop_res_")]
            if pop_cols:
                melted = trends.melt(
                    id_vars=["istat_code"],
                    value_vars=pop_cols,
                    var_name="year_col",
                    value_name="population",
                )
                melted["anno"] = melted["year_col"].str.extract(r"(\d+)")[0].astype(int) + 2000
                population_df = melted[["istat_code", "anno", "population"]]
                logger.info(f"Loaded {len(population_df):,} population records from trends")

        if pop_2021_path.exists():
            logger.info(f"Loading 2021 population from {pop_2021_path}")
            pop_2021 = pd.read_csv(pop_2021_path, encoding="utf-8")
            pop_2021["istat_code"] = pop_2021["pro_com_t"].astype(str).str.zfill(6)
            pop_2021["anno"] = 2021

            # Rename columns
            pop_2021 = pop_2021.rename(columns={
                "totale": "population",
                "<12": "pop_under_12",
                ">=12": "pop_12_plus",
            })

            if population_df.empty:
                population_df = pop_2021[["istat_code", "anno", "population"]]
            else:
                # Merge age breakdown into existing data
                pop_2021_subset = pop_2021[
                    ["istat_code", "pop_under_12", "pop_12_plus"]
                ].drop_duplicates()
                population_df = population_df.merge(
                    pop_2021_subset, on="istat_code", how="left"
                )

        return population_df

    def compute_population_change(
        self,
        df: pd.DataFrame,
        base_year: int = 2011,
        end_year: int = 2021,
    ) -> pd.DataFrame:
        """Compute population change percentage.

        Args:
            df: DataFrame with istat_code, anno, population columns.
            base_year: Starting year for change calculation.
            end_year: Ending year for change calculation.

        Returns:
            DataFrame with pop_change_pct column.
        """
        if df.empty:
            return df

        # Get base and end populations
        base = df[df["anno"] == base_year][["istat_code", "population"]]
        base = base.rename(columns={"population": "pop_base"})

        end = df[df["anno"] == end_year][["istat_code", "population"]]
        end = end.rename(columns={"population": "pop_end"})

        # Merge and compute change
        merged = base.merge(end, on="istat_code", how="outer")
        merged["pop_change_pct"] = (
            (merged["pop_end"] - merged["pop_base"]) / merged["pop_base"] * 100
        )

        # Classify growth patterns
        merged["pop_declining"] = (merged["pop_change_pct"] < -2).astype(int)
        merged["pop_growing_fast"] = (merged["pop_change_pct"] > 5).astype(int)

        logger.info(
            f"Computed population change {base_year}-{end_year}: "
            f"{(merged['pop_change_pct'] < 0).sum():,} declining, "
            f"{(merged['pop_change_pct'] > 0).sum():,} growing"
        )

        return merged[["istat_code", "pop_change_pct", "pop_declining", "pop_growing_fast"]]

    def add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed geographic features.

        Adds:
        - dist_major_city: Distance to nearest major city (km)
        - dist_coast: Approximate distance to coast (km)
        - northern: Binary flag for Northern Italy
        - coastal: Binary flag (dist_coast < 20km)

        Args:
            df: DataFrame with lat, lon columns.

        Returns:
            DataFrame with geographic features added.
        """
        if "lat" not in df.columns or "lon" not in df.columns:
            logger.warning("Missing lat/lon columns, skipping geographic features")
            return df

        df = df.copy()

        # Distance to Milan (primary economic center)
        df["dist_major_city"] = df.apply(
            lambda row: distance_to_milan(row["lat"], row["lon"]), axis=1
        )

        # Approximate coastal distance using nearest coast point
        # Italian coastline approximation points
        coast_points = [
            (43.7, 10.3),   # Livorno (Tyrrhenian)
            (41.5, 12.5),   # Lazio coast
            (38.2, 15.6),   # Sicily (Messina)
            (40.8, 14.3),   # Naples
            (41.1, 16.9),   # Bari (Adriatic)
            (44.4, 12.2),   # Rimini
            (45.4, 12.3),   # Venice
        ]

        def min_coast_distance(row):
            distances = [
                haversine_distance(row["lat"], row["lon"], lat, lon)
                for lat, lon in coast_points
            ]
            return min(distances)

        df["dist_coast"] = df.apply(min_coast_distance, axis=1)
        df["coastal"] = (df["dist_coast"] < 20).astype(int)

        # Northern Italy flag
        if "cod_regione" in df.columns:
            df["northern"] = df["cod_regione"].isin(NORTHERN_REGIONS).astype(int)
        else:
            # Use latitude as proxy (roughly north of Rome)
            df["northern"] = (df["lat"] > 42.5).astype(int)

        # Urban flag based on coordinates (placeholder - should use population)
        df["urban"] = 0  # Will be set based on population threshold

        logger.info("Added geographic features")
        return df

    def load_and_process(self, data_dir: Path) -> pd.DataFrame:
        """Full pipeline: load and process all ISTAT demographic data.

        Args:
            data_dir: Root data directory.

        Returns:
            Municipality-level DataFrame with columns:
            - istat_code: 6-digit ISTAT code
            - nome: Municipality name
            - lat, lon: Coordinates
            - popolazione: Population (latest year)
            - pop_change_pct: Population change percentage
            - dist_major_city, dist_coast: Geographic distances
            - northern, coastal, urban: Geographic flags
        """
        # Load metadata
        metadata = self.load_municipality_metadata(data_dir)
        if metadata.empty:
            logger.error("Could not load municipality metadata")
            return pd.DataFrame()

        # Load population
        population = self.load_population_data(data_dir)

        # Compute population change
        if not population.empty:
            pop_change = self.compute_population_change(population)
            metadata = metadata.merge(pop_change, on="istat_code", how="left")

            # Get latest population
            latest_pop = population.groupby("istat_code")["population"].last().reset_index()
            latest_pop = latest_pop.rename(columns={"population": "popolazione"})
            metadata = metadata.merge(latest_pop, on="istat_code", how="left")

        # Add geographic features
        metadata = self.add_geographic_features(metadata)

        # Set urban flag based on population
        if "popolazione" in metadata.columns:
            metadata["urban"] = (metadata["popolazione"] > 50000).astype(int)
            metadata["large_city"] = (metadata["popolazione"] > 500000).astype(int)

        logger.info(
            f"Processed ISTAT data: {len(metadata):,} municipalities, "
            f"{metadata.columns.tolist()}"
        )

        return metadata


def load_istat_demographics(data_dir: Path) -> pd.DataFrame:
    """Convenience function to load processed ISTAT demographics.

    Args:
        data_dir: Root data directory.

    Returns:
        Processed municipality-level demographic DataFrame.
    """
    processor = ISTATProcessor()
    return processor.load_and_process(data_dir)
