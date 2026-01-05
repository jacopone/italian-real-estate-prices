"""Airbnb/InsideAirbnb data processor.

Processes short-term rental (STR) data from InsideAirbnb.
STR density is a significant predictor of real estate prices,
especially in tourist areas where Airbnb competes with long-term rentals.

Data source: http://insideairbnb.com/get-the-data/
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

from src.config import AirbnbConfig
from src.utils.constants import INSIDEAIRBNB_CITIES


class AirbnbProcessor:
    """Process InsideAirbnb listing data.

    This processor handles:
    - Loading listing data from multiple cities
    - Geocoding listings to municipalities
    - Computing STR density metrics
    - Calculating Airbnb premium vs long-term rents

    Example:
        >>> processor = AirbnbProcessor(config.airbnb)
        >>> listings = processor.load_and_process(data_dir)
        >>> listings.columns
        ['istat_code', 'airbnb_listings', 'airbnb_price_median', 'str_density', ...]
    """

    def __init__(self, config: AirbnbConfig | None = None):
        """Initialize processor with configuration.

        Args:
            config: Airbnb configuration. Uses defaults if None.
        """
        self.config = config or AirbnbConfig()

    def load_city_listings(
        self,
        data_dir: Path,
        city: str,
    ) -> pd.DataFrame:
        """Load listings for a single city.

        Args:
            data_dir: Root data directory.
            city: City name (e.g., 'milan', 'florence').

        Returns:
            DataFrame with listings for the city.
        """
        # Check for gzipped or plain CSV
        listings_path = data_dir / "raw" / "airbnb" / city / "listings.csv.gz"
        if not listings_path.exists():
            listings_path = data_dir / "raw" / "airbnb" / city / "listings.csv"

        if not listings_path.exists():
            logger.debug(f"No Airbnb data found for {city}")
            return pd.DataFrame()

        logger.info(f"Loading Airbnb listings from {listings_path}")

        df = pd.read_csv(
            listings_path,
            usecols=[
                "id", "name", "latitude", "longitude", "price",
                "number_of_reviews", "room_type", "neighbourhood",
            ],
            low_memory=False,
        )

        # Parse price (remove $ and commas)
        if "price" in df.columns:
            df["price"] = (
                df["price"]
                .astype(str)
                .str.replace(r"[$,]", "", regex=True)
                .astype(float)
            )

        df["source_city"] = city
        logger.info(f"Loaded {len(df):,} listings for {city}")
        return df

    def load_all_cities(self, data_dir: Path) -> pd.DataFrame:
        """Load listings from all available cities.

        Args:
            data_dir: Root data directory.

        Returns:
            Combined DataFrame with listings from all cities.
        """
        all_listings = []

        for city in self.config.available_cities:
            city_df = self.load_city_listings(data_dir, city)
            if not city_df.empty:
                all_listings.append(city_df)

        if not all_listings:
            logger.warning("No Airbnb listings loaded from any city")
            return pd.DataFrame()

        combined = pd.concat(all_listings, ignore_index=True)
        logger.info(
            f"Combined {len(combined):,} listings from "
            f"{len(all_listings)} cities"
        )
        return combined

    def geocode_to_municipalities(
        self,
        listings: pd.DataFrame,
        municipalities_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """Assign listings to municipalities using spatial join.

        Args:
            listings: DataFrame with latitude/longitude columns.
            municipalities_gdf: GeoDataFrame with municipality boundaries.

        Returns:
            DataFrame with istat_code column added.
        """
        if listings.empty:
            return listings

        # Create GeoDataFrame from listings
        listings_gdf = gpd.GeoDataFrame(
            listings,
            geometry=gpd.points_from_xy(
                listings["longitude"], listings["latitude"]
            ),
            crs="EPSG:4326",
        )

        # Ensure municipalities have consistent CRS
        if municipalities_gdf.crs != "EPSG:4326":
            municipalities_gdf = municipalities_gdf.to_crs("EPSG:4326")

        # Rename conflicting columns in municipalities
        muni_cols_rename = {}
        if "name" in municipalities_gdf.columns:
            muni_cols_rename["name"] = "comune_name"
        if "prov_name" in municipalities_gdf.columns:
            pass  # Keep as is
        municipalities_gdf = municipalities_gdf.rename(columns=muni_cols_rename)

        # Spatial join
        joined = gpd.sjoin(
            listings_gdf,
            municipalities_gdf[["geometry", "pro_com_t", "comune_name"]],
            how="left",
            predicate="within",
        )

        # Rename ISTAT code column
        joined = joined.rename(columns={"pro_com_t": "istat_code"})

        # Drop geometry for regular DataFrame output
        result = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))

        matched = result["istat_code"].notna().sum()
        logger.info(
            f"Geocoded {matched:,}/{len(listings):,} listings "
            f"({matched/len(listings)*100:.1f}%)"
        )

        return result

    def aggregate_by_municipality(self, listings: pd.DataFrame) -> pd.DataFrame:
        """Aggregate listings to municipality level.

        Computes:
        - Count of listings
        - Median and mean nightly price
        - Total reviews (proxy for occupancy)

        Args:
            listings: DataFrame with istat_code column.

        Returns:
            Municipality-level aggregated DataFrame.
        """
        if listings.empty or "istat_code" not in listings.columns:
            return pd.DataFrame()

        # Filter to geocoded listings
        geocoded = listings[listings["istat_code"].notna()].copy()

        # Filter by minimum reviews if configured
        if self.config.min_reviews > 0:
            geocoded = geocoded[
                geocoded["number_of_reviews"] >= self.config.min_reviews
            ]

        # Aggregate
        aggregated = (
            geocoded.groupby("istat_code")
            .agg(
                airbnb_listings=("id", "count"),
                airbnb_price_median=("price", "median"),
                airbnb_price_mean=("price", "mean"),
                airbnb_reviews=("number_of_reviews", "sum"),
            )
            .reset_index()
        )

        # Add source city info
        city_info = (
            geocoded.groupby("istat_code")["source_city"]
            .first()
            .reset_index()
        )
        aggregated = aggregated.merge(city_info, on="istat_code", how="left")

        logger.info(
            f"Aggregated to {len(aggregated):,} municipalities, "
            f"total listings: {aggregated['airbnb_listings'].sum():,}"
        )

        return aggregated

    def aggregate_by_province(self, listings: pd.DataFrame) -> pd.DataFrame:
        """Aggregate listings to province level.

        Args:
            listings: DataFrame with istat_code column.

        Returns:
            Province-level aggregated DataFrame.
        """
        if listings.empty or "istat_code" not in listings.columns:
            return pd.DataFrame()

        # Extract province code from ISTAT code (first 3 digits)
        listings = listings.copy()
        listings["prov_code"] = listings["istat_code"].str[:3]

        geocoded = listings[listings["prov_code"].notna()]

        aggregated = (
            geocoded.groupby("prov_code")
            .agg(
                airbnb_listings=("id", "count"),
                airbnb_price_median=("price", "median"),
                airbnb_price_mean=("price", "mean"),
                airbnb_reviews=("number_of_reviews", "sum"),
            )
            .reset_index()
        )

        logger.info(
            f"Aggregated to {len(aggregated):,} provinces"
        )

        return aggregated

    def compute_str_density(
        self,
        aggregated: pd.DataFrame,
        population: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute STR density (listings per 1000 residents).

        Args:
            aggregated: Municipality/province-level Airbnb data.
            population: Population data with matching geography.

        Returns:
            DataFrame with str_density column.
        """
        if aggregated.empty:
            return aggregated

        # Determine join key
        if "istat_code" in aggregated.columns and "istat_code" in population.columns:
            join_key = "istat_code"
        elif "prov_code" in aggregated.columns and "prov_code" in population.columns:
            join_key = "prov_code"
        else:
            logger.warning("Cannot compute STR density: no matching geography key")
            return aggregated

        # Merge population
        pop_cols = [join_key, "popolazione"] if "popolazione" in population.columns else [join_key, "population"]
        pop_subset = population[pop_cols].drop_duplicates()

        merged = aggregated.merge(pop_subset, on=join_key, how="left")

        # Compute density
        pop_col = "popolazione" if "popolazione" in merged.columns else "population"
        merged["str_density"] = (
            merged["airbnb_listings"] / merged[pop_col] * 1000
        )
        merged["str_density"] = merged["str_density"].replace([np.inf, -np.inf], np.nan)

        logger.info(
            f"Computed STR density: median={merged['str_density'].median():.2f} "
            f"listings per 1000 residents"
        )

        return merged

    def compute_airbnb_premium(
        self,
        aggregated: pd.DataFrame,
        long_term_rents: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute Airbnb premium vs long-term rental prices.

        Premium = (Airbnb monthly revenue) / (Long-term monthly rent) - 1

        Args:
            aggregated: Airbnb data with airbnb_price_median.
            long_term_rents: Rent data with affitto_medio (EUR/sqm/month).

        Returns:
            DataFrame with airbnb_premium column.
        """
        if aggregated.empty or "airbnb_price_median" not in aggregated.columns:
            return aggregated

        # Assume 70 sqm average apartment and configured occupancy
        avg_sqm = 70
        days_per_month = self.config.days_per_month

        # Compute monthly Airbnb revenue (per sqm)
        aggregated = aggregated.copy()
        aggregated["airbnb_monthly_sqm"] = (
            aggregated["airbnb_price_median"] * days_per_month / avg_sqm
        )

        # Merge long-term rents
        if "istat_code" in aggregated.columns and "istat_code" in long_term_rents.columns:
            rent_cols = ["istat_code", "affitto_medio"]
            merged = aggregated.merge(
                long_term_rents[rent_cols],
                on="istat_code",
                how="left",
            )
        else:
            return aggregated

        # Compute premium
        merged["airbnb_premium"] = (
            merged["airbnb_monthly_sqm"] / merged["affitto_medio"] - 1
        ) * 100  # As percentage

        merged["airbnb_premium"] = merged["airbnb_premium"].replace([np.inf, -np.inf], np.nan)

        valid_premium = merged["airbnb_premium"].dropna()
        if not valid_premium.empty:
            logger.info(
                f"Airbnb premium: median={valid_premium.median():.0f}%, "
                f"range={valid_premium.min():.0f}%-{valid_premium.max():.0f}%"
            )

        return merged

    def create_str_proxy(
        self,
        tourism: pd.DataFrame,
        airbnb_by_province: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create STR proxy for provinces without Airbnb data.

        Uses calibration: str_proxy = tourism_intensity * (avg_airbnb_density / avg_tourism)

        Args:
            tourism: Tourism data with tourism_intensity.
            airbnb_by_province: Airbnb data for provinces that have it.

        Returns:
            Tourism DataFrame with str_proxy column added.
        """
        if tourism.empty or airbnb_by_province.empty:
            return tourism

        # Compute calibration ratio from provinces with both data
        merged = tourism.merge(
            airbnb_by_province[["prov_code", "str_density"]],
            on="prov_code",
            how="left",
        )

        # Calibration: ratio of airbnb_density to tourism_intensity
        has_both = merged["str_density"].notna() & merged["tourism_intensity"].notna()
        if has_both.sum() == 0:
            logger.warning("No overlapping provinces for STR proxy calibration")
            return tourism

        calibration_ratio = (
            merged.loc[has_both, "str_density"].mean() /
            merged.loc[has_both, "tourism_intensity"].mean()
        )

        logger.info(f"STR proxy calibration ratio: {calibration_ratio:.4f}")

        # Apply proxy where Airbnb data is missing
        tourism = tourism.copy()
        tourism["str_proxy"] = tourism["tourism_intensity"] * calibration_ratio

        return tourism

    def load_and_process(
        self,
        data_dir: Path,
        municipalities_gdf: gpd.GeoDataFrame | None = None,
    ) -> pd.DataFrame:
        """Full pipeline: load and process Airbnb data.

        Args:
            data_dir: Root data directory.
            municipalities_gdf: Optional municipality boundaries for geocoding.

        Returns:
            Municipality-level DataFrame with columns:
            - istat_code: ISTAT municipality code
            - airbnb_listings: Number of listings
            - airbnb_price_median: Median nightly price
            - airbnb_reviews: Total reviews
            - source_city: InsideAirbnb source city
        """
        # Load all listings
        listings = self.load_all_cities(data_dir)
        if listings.empty:
            return pd.DataFrame()

        # Geocode if boundaries provided
        if municipalities_gdf is not None:
            listings = self.geocode_to_municipalities(listings, municipalities_gdf)

        # Aggregate to municipality level
        aggregated = self.aggregate_by_municipality(listings)

        return aggregated


def load_airbnb_data(
    data_dir: Path,
    config: AirbnbConfig | None = None,
) -> pd.DataFrame:
    """Convenience function to load processed Airbnb data.

    Args:
        data_dir: Root data directory.
        config: Optional Airbnb configuration.

    Returns:
        Processed municipality-level Airbnb DataFrame.
    """
    processor = AirbnbProcessor(config)
    return processor.load_and_process(data_dir)
