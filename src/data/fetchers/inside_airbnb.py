"""Inside Airbnb data fetcher for short-term rental analysis.

Data source: https://insideairbnb.com/get-the-data/
License: Creative Commons CC0 1.0
"""

import gzip
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Inside Airbnb data URLs
INSIDE_AIRBNB_BASE = "http://data.insideairbnb.com"

# Known data dates (updated quarterly - add recent ones as they become available)
# Format: YYYY-MM-DD (typically first day of quarter months: Jan, Apr, Jul, Oct)
KNOWN_DATA_DATES = [
    "2025-10-04", "2025-07-02", "2025-04-09", "2025-01-03",
    "2024-09-13", "2024-06-21", "2024-03-22", "2024-01-03",
    "2023-12-13", "2023-09-05", "2023-06-08", "2023-03-14",
]

# Italian cities and regions available
ITALIAN_LOCATIONS = {
    # Cities
    "rome": {"country": "italy", "region": "lazio", "city": "rome"},
    "milan": {"country": "italy", "region": "lombardy", "city": "milan"},
    "florence": {"country": "italy", "region": "tuscany", "city": "florence"},
    "venice": {"country": "italy", "region": "veneto", "city": "venice"},
    "naples": {"country": "italy", "region": "campania", "city": "naples"},
    # Regions (aggregated data)
    "puglia": {"country": "italy", "region": "puglia", "city": None},
    "sicily": {"country": "italy", "region": "sicily", "city": None},
    "sardinia": {"country": "italy", "region": "sardinia", "city": None},
}


class InsideAirbnbFetcher:
    """Fetcher for Inside Airbnb listing data."""

    def __init__(self, cache_dir: Path = Path("data/raw/airbnb")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_listings(
        self,
        location: str,
        date: str | None = None
    ) -> pd.DataFrame:
        """Fetch Airbnb listings for an Italian location.

        Args:
            location: Location key (e.g., 'rome', 'milan', 'puglia')
            date: Data date in YYYY-MM-DD format (defaults to latest)

        Returns:
            DataFrame with listing data
        """
        if location not in ITALIAN_LOCATIONS:
            raise ValueError(f"Unknown location: {location}. Available: {list(ITALIAN_LOCATIONS.keys())}")

        loc_info = ITALIAN_LOCATIONS[location]

        # Build cache filename
        date_str = date or "latest"
        cache_file = self.cache_dir / f"{location}_listings_{date_str}.csv"

        if cache_file.exists():
            logger.info(f"Loading cached Airbnb listings from {cache_file}")
            return pd.read_csv(cache_file)

        logger.info(f"Fetching Airbnb listings for {location}...")

        # Build URL - Inside Airbnb URL structure
        # http://data.insideairbnb.com/{country}/{region}/{city}/{date}/data/listings.csv.gz
        if loc_info["city"]:
            url_path = f"{loc_info['country']}/{loc_info['region']}/{loc_info['city']}"
        else:
            url_path = f"{loc_info['country']}/{loc_info['region']}"

        # Try to find the latest data date
        if date is None:
            date = self._get_latest_date(url_path)

        if date is None:
            logger.error(f"Could not determine latest data date for {location}")
            return pd.DataFrame()

        url = f"{INSIDE_AIRBNB_BASE}/{url_path}/{date}/data/listings.csv.gz"

        try:
            df = self._download_and_parse(url)

            if not df.empty:
                df.to_csv(cache_file, index=False)
                logger.info(f"Cached {len(df)} listings to {cache_file}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch listings for {location}: {e}")
            return pd.DataFrame()

    def _get_latest_date(self, url_path: str) -> str | None:
        """Try to determine the latest data date available.

        Uses known dates first, then falls back to date generation.
        """
        # First try known dates (most recent first)
        for date_str in KNOWN_DATA_DATES:
            url = f"{INSIDE_AIRBNB_BASE}/{url_path}/{date_str}/data/listings.csv.gz"
            try:
                response = requests.head(url, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    logger.info(f"Found data for date: {date_str}")
                    return date_str
            except requests.RequestException:
                continue

        # Fallback: try first-of-month dates going back
        today = datetime.now()
        for months_back in range(0, 24):
            # Go back month by month
            year = today.year
            month = today.month - months_back
            while month <= 0:
                month += 12
                year -= 1

            # Try common day patterns: 1st, mid-month
            for day in [1, 15, 5, 10, 20]:
                try:
                    test_date = datetime(year, month, day)
                    date_str = test_date.strftime("%Y-%m-%d")
                    url = f"{INSIDE_AIRBNB_BASE}/{url_path}/{date_str}/data/listings.csv.gz"

                    response = requests.head(url, timeout=10, allow_redirects=True)
                    if response.status_code == 200:
                        logger.info(f"Found data for date: {date_str}")
                        return date_str
                except (ValueError, requests.RequestException):
                    continue

        return None

    def _download_and_parse(self, url: str) -> pd.DataFrame:
        """Download and parse gzipped or plain CSV."""
        logger.info(f"Downloading: {url}")
        response = requests.get(url, timeout=120, allow_redirects=True)
        response.raise_for_status()

        from io import BytesIO, StringIO

        # Try gzip first
        if url.endswith('.gz') or response.headers.get('Content-Encoding') == 'gzip':
            try:
                content = gzip.decompress(response.content)
                return pd.read_csv(StringIO(content.decode("utf-8")))
            except gzip.BadGzipFile:
                logger.warning("File not gzipped, trying plain CSV")

        # Fallback to plain CSV
        return pd.read_csv(BytesIO(response.content))

    def fetch_all_italian_cities(self) -> pd.DataFrame:
        """Fetch listings for all Italian cities and combine.

        Returns:
            Combined DataFrame with location column added
        """
        all_listings = []

        for location in ITALIAN_LOCATIONS:
            try:
                df = self.fetch_listings(location)
                if not df.empty:
                    df["source_location"] = location
                    all_listings.append(df)
            except Exception as e:
                logger.warning(f"Skipping {location}: {e}")

        if all_listings:
            return pd.concat(all_listings, ignore_index=True)
        return pd.DataFrame()

    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate key Airbnb metrics from listings data.

        Args:
            df: Raw listings DataFrame

        Returns:
            DataFrame with calculated metrics
        """
        if df.empty:
            return df

        metrics = df.copy()

        # Listing type breakdown
        if "room_type" in metrics.columns:
            metrics["is_entire_home"] = metrics["room_type"] == "Entire home/apt"

        # Availability metrics
        if "availability_365" in metrics.columns:
            metrics["high_availability"] = metrics["availability_365"] > 180

        # Estimated revenue (simplified)
        if "price" in metrics.columns and "availability_365" in metrics.columns:
            # Clean price column
            if metrics["price"].dtype == object:
                metrics["price_clean"] = (
                    metrics["price"]
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .astype(float)
                )
            else:
                metrics["price_clean"] = metrics["price"]

            # Estimate occupancy at 50% of available nights
            metrics["estimated_revenue"] = (
                metrics["price_clean"] * metrics["availability_365"] * 0.5
            )

        # Reviews as activity proxy
        if "number_of_reviews" in metrics.columns:
            metrics["active_listing"] = metrics["number_of_reviews"] > 0

        return metrics


def aggregate_by_neighbourhood(
    listings_df: pd.DataFrame,
    population_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Aggregate Airbnb listings by neighbourhood.

    Args:
        listings_df: Listings with neighbourhood column
        population_df: Optional population data for density calculation

    Returns:
        Aggregated metrics by neighbourhood
    """
    if "neighbourhood" not in listings_df.columns:
        logger.warning("No neighbourhood column found")
        return pd.DataFrame()

    agg = listings_df.groupby("neighbourhood").agg(
        listing_count=("id", "count"),
        entire_home_count=("is_entire_home", "sum") if "is_entire_home" in listings_df.columns else ("id", "count"),
        avg_price=("price_clean", "mean") if "price_clean" in listings_df.columns else ("id", "count"),
        total_reviews=("number_of_reviews", "sum") if "number_of_reviews" in listings_df.columns else ("id", "count"),
        avg_availability=("availability_365", "mean") if "availability_365" in listings_df.columns else ("id", "count"),
    ).reset_index()

    # Calculate entire home ratio
    if "entire_home_count" in agg.columns and "listing_count" in agg.columns:
        agg["entire_home_ratio"] = agg["entire_home_count"] / agg["listing_count"]

    # Calculate density if population available
    if population_df is not None and not population_df.empty:
        agg = agg.merge(population_df, on="neighbourhood", how="left")
        if "population" in agg.columns:
            agg["listings_per_1000"] = agg["listing_count"] / agg["population"] * 1000

    return agg
