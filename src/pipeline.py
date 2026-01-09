"""Main data pipeline for Italian real estate price analysis.

This module orchestrates data fetching, processing, and feature creation.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.fetchers import InsideAirbnbFetcher, ISTATFetcher, OMIFetcher
from src.data.prepare_supplementary_data import prepare_municipal_supplementary_data
from src.features import create_vacancy_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RealEstatePipeline:
    """Main pipeline for data processing and feature engineering."""

    def __init__(
        self,
        config_path: Path = Path("configs/data_sources.yaml"),
        data_dir: Path = Path("data"),
    ):
        self.config_path = config_path
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize fetchers
        self.omi_fetcher = OMIFetcher(cache_dir=self.raw_dir / "omi")
        self.istat_fetcher = ISTATFetcher(cache_dir=self.raw_dir / "istat")
        self.airbnb_fetcher = InsideAirbnbFetcher(cache_dir=self.raw_dir / "airbnb")

    def fetch_all_data(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch all data sources.

        Args:
            force_refresh: If True, bypass cache and re-download

        Returns:
            Dictionary of DataFrames by source name
        """
        data = {}

        # OMI data
        logger.info("Fetching OMI data...")
        omi_data = self.omi_fetcher.fetch_all(force_refresh)
        data["omi_valori"] = omi_data["valori"]
        data["omi_zone"] = omi_data["zone"]
        data["omi_comuni"] = omi_data["comuni"]

        # ISTAT housing census
        logger.info("Fetching ISTAT housing census...")
        data["housing_census"] = self.istat_fetcher.fetch_housing_census(year=2021)

        # ISTAT demographics
        logger.info("Fetching ISTAT demographics...")
        data["demographics"] = self.istat_fetcher.fetch_demographics(
            start_year=2011,
            end_year=2024
        )

        # ISTAT tourism
        logger.info("Fetching ISTAT tourism...")
        data["tourism"] = self.istat_fetcher.fetch_tourism(
            start_year=2019,
            end_year=2024
        )

        # Airbnb data (major cities)
        logger.info("Fetching Airbnb data...")
        try:
            data["airbnb"] = self.airbnb_fetcher.fetch_all_italian_cities()
        except Exception as e:
            logger.warning(f"Failed to fetch Airbnb data: {e}")
            data["airbnb"] = pd.DataFrame()

        return data

    def create_municipal_dataset(
        self,
        data: dict[str, pd.DataFrame] | None = None
    ) -> pd.DataFrame:
        """Create a unified municipal-level dataset with all features.

        Args:
            data: Pre-fetched data dictionary (fetches if not provided)

        Returns:
            Municipal-level DataFrame with all features
        """
        if data is None:
            data = self.fetch_all_data()

        logger.info("Creating municipal dataset...")

        # Start with OMI municipal aggregation
        from src.data.fetchers.omi import aggregate_by_municipality, calculate_price_metrics

        valori = calculate_price_metrics(data["omi_valori"])
        municipal_prices = aggregate_by_municipality(
            valori,
            data["omi_zone"],
            property_type="Abitazioni civili"
        )

        if municipal_prices.empty:
            logger.error("Failed to create municipal price aggregation")
            return pd.DataFrame()

        # Check if ISTAT data is empty - use supplementary data as fallback
        istat_data_empty = (
            data["housing_census"].empty and
            data["demographics"].empty and
            data["tourism"].empty
        )

        if istat_data_empty:
            logger.info("ISTAT APIs unavailable - using supplementary data sources")
            supplementary = prepare_municipal_supplementary_data(self.data_dir)

            if not supplementary.empty:
                # Merge supplementary data with prices
                housing_df = self._merge_supplementary_with_prices(municipal_prices, supplementary)
            else:
                housing_df = self._prepare_housing_data(data["housing_census"], municipal_prices)

            # Use supplementary tourism/demographics data
            demographics_df = supplementary if not supplementary.empty else data["demographics"]
            tourism_df = supplementary if not supplementary.empty else data["tourism"]
        else:
            housing_df = self._prepare_housing_data(data["housing_census"], municipal_prices)
            demographics_df = data["demographics"]
            tourism_df = data["tourism"]

        # Add vacancy features
        result = create_vacancy_features(
            housing_df=housing_df,
            demographics_df=demographics_df,
            tourism_df=tourism_df,
            airbnb_df=self._aggregate_airbnb(data["airbnb"]),
            territory_col="Comune_ISTAT",
        )

        # Save processed dataset
        output_path = self.processed_dir / "municipal_features.csv"
        result.to_csv(output_path, index=False)
        logger.info(f"Saved municipal dataset to {output_path}")

        return result

    def _merge_supplementary_with_prices(
        self,
        municipal_prices: pd.DataFrame,
        supplementary: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge supplementary data with OMI price data."""
        result = municipal_prices.copy()

        # OMI Comune_ISTAT is 7 digits (e.g., 1006003), ISTAT is 6 digits (e.g., 006003)
        # Remove the leading '1' from OMI codes to get standard 6-digit ISTAT codes
        result["Comune_ISTAT_std"] = (
            result["Comune_ISTAT"]
            .astype(int)
            .astype(str)
            .str.zfill(7)
            .str[1:]  # Remove first digit
        )

        # Merge on Comune_ISTAT
        supp_cols = [
            "Comune_ISTAT", "pop_2011", "pop_2021", "pop_10yr_change",
            "is_depopulating", "tourism_intensity", "tourism_level", "vacancy_rate"
        ]
        supp_subset = supplementary[[c for c in supp_cols if c in supplementary.columns]].copy()

        # Convert ISTAT codes to 6-digit string format
        supp_subset["Comune_ISTAT_std"] = supp_subset["Comune_ISTAT"].astype(str).str.zfill(6)
        supp_subset = supp_subset.drop(columns=["Comune_ISTAT"])

        result = result.merge(supp_subset, on="Comune_ISTAT_std", how="left")
        result = result.drop(columns=["Comune_ISTAT_std"])

        # Fill missing values with regional averages
        result["vacancy_rate"] = result["vacancy_rate"].fillna(0.272)
        result["pop_10yr_change"] = result["pop_10yr_change"].fillna(0)
        result["tourism_intensity"] = result["tourism_intensity"].fillna(5)

        return result

    def _prepare_housing_data(
        self,
        housing_df: pd.DataFrame,
        municipal_prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare housing data for feature creation.

        Merges census housing data with municipal price data.
        Uses regional vacancy rates as fallback when municipal data unavailable.
        """
        result = municipal_prices.copy()

        if not housing_df.empty:
            # Merge housing with prices
            result = result.merge(
                housing_df,
                on="Comune_ISTAT",
                how="left"
            )

            # Calculate vacancy rate if columns exist
            if "abitazioni_non_occupate" in result.columns and "abitazioni_totali" in result.columns:
                result["vacancy_rate"] = result["abitazioni_non_occupate"] / result["abitazioni_totali"]
                return result

        # Fallback: use regional vacancy rates from ISTAT 2021 estimates
        regional_file = self.raw_dir / "istat" / "regional_vacancy_2021.csv"
        if regional_file.exists():
            logger.info("Using regional vacancy rates as fallback")
            regional_vacancy = pd.read_csv(regional_file)

            # Create mapping from Regione name to vacancy rate
            vacancy_map = dict(zip(regional_vacancy["Regione"], regional_vacancy["vacancy_rate"], strict=False))

            # Apply regional vacancy rates based on Regione column
            if "Regione" in result.columns:
                result["vacancy_rate"] = result["Regione"].map(vacancy_map)
                # Fill any missing with national average
                result["vacancy_rate"] = result["vacancy_rate"].fillna(0.272)
            else:
                result["vacancy_rate"] = 0.272  # National average
        else:
            result["vacancy_rate"] = 0.272  # National average

        return result

    def _aggregate_airbnb(self, airbnb_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate Airbnb listings by municipality."""
        if airbnb_df.empty:
            return pd.DataFrame()

        from src.data.fetchers.inside_airbnb import aggregate_by_neighbourhood

        # First add metrics
        airbnb_with_metrics = self.airbnb_fetcher.calculate_metrics(airbnb_df)

        # Aggregate by neighbourhood (as proxy for municipality)
        return aggregate_by_neighbourhood(airbnb_with_metrics)

    def generate_vacancy_report(self, data: pd.DataFrame) -> dict:
        """Generate summary statistics on vacancy classification.

        Args:
            data: Municipal dataset with vacancy features

        Returns:
            Dictionary with summary statistics
        """
        if "vacancy_type" not in data.columns:
            return {"error": "No vacancy classification found"}

        report = {
            "total_municipalities": len(data),
            "vacancy_distribution": data["vacancy_type"].value_counts().to_dict(),
            "vacancy_distribution_pct": (data["vacancy_type"].value_counts() / len(data) * 100).round(1).to_dict(),
            "avg_vacancy_rate_by_type": data.groupby("vacancy_type")["vacancy_rate"].mean().round(3).to_dict(),
        }

        if "price_avg" in data.columns:
            report["avg_price_by_type"] = data.groupby("vacancy_type")["price_avg"].mean().round(0).to_dict()

        if "gross_yield" in data.columns:
            report["avg_yield_by_type"] = data.groupby("vacancy_type")["gross_yield"].mean().round(3).to_dict()

        # Top declining areas
        if "pop_10yr_change" in data.columns:
            declining = data.nsmallest(10, "pop_10yr_change")[
                ["Comune_descrizione", "Regione", "pop_10yr_change", "vacancy_rate", "vacancy_type"]
            ]
            report["top_declining_municipalities"] = declining.to_dict("records")

        # Top tourist areas
        if "tourism_intensity" in data.columns:
            tourist = data.nlargest(10, "tourism_intensity")[
                ["Comune_descrizione", "Regione", "tourism_intensity", "vacancy_rate", "vacancy_type"]
            ]
            report["top_tourist_municipalities"] = tourist.to_dict("records")

        return report


def main():
    """Run the full pipeline."""
    pipeline = RealEstatePipeline()

    # Fetch and process all data
    data = pipeline.fetch_all_data()

    # Create municipal dataset
    municipal_df = pipeline.create_municipal_dataset(data)

    # Generate report
    report = pipeline.generate_vacancy_report(municipal_df)

    # Print summary
    print("\n" + "=" * 60)
    print("VACANCY ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nTotal municipalities: {report.get('total_municipalities', 'N/A')}")

    print("\nVacancy Type Distribution:")
    for vtype, count in report.get("vacancy_distribution", {}).items():
        pct = report.get("vacancy_distribution_pct", {}).get(vtype, 0)
        print(f"  {vtype}: {count} ({pct}%)")

    print("\nAverage Vacancy Rate by Type:")
    for vtype, rate in report.get("avg_vacancy_rate_by_type", {}).items():
        print(f"  {vtype}: {rate:.1%}")

    if "avg_price_by_type" in report:
        print("\nAverage Price (€/m²) by Type:")
        for vtype, price in report["avg_price_by_type"].items():
            print(f"  {vtype}: €{price:,.0f}")

    print("\n" + "=" * 60)

    return municipal_df, report


if __name__ == "__main__":
    main()
