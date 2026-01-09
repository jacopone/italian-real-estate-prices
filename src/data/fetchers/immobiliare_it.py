"""Immobiliare.it listing data fetcher.

Fetches real estate listing data from Immobiliare.it for model validation.
Uses the public search API to get aggregate pricing data.

Note: This fetcher is for validation purposes only, comparing listing prices
with OMI government valuations and model predictions.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# City configurations with Immobiliare.it search parameters and ISTAT codes
CITY_CONFIGS = {
    "roma": {
        "name": "Roma",
        "istat_code": "058091",
        "omi_code": "1058091",
        "region": "LAZIO",
        "province": "RM",
        "immobiliare_id": "roma",
    },
    "milano": {
        "name": "Milano",
        "istat_code": "015146",
        "omi_code": "1015146",
        "region": "LOMBARDIA",
        "province": "MI",
        "immobiliare_id": "milano",
    },
    "firenze": {
        "name": "Firenze",
        "istat_code": "048017",
        "omi_code": "1048017",
        "region": "TOSCANA",
        "province": "FI",
        "immobiliare_id": "firenze",
    },
    "napoli": {
        "name": "Napoli",
        "istat_code": "063049",
        "omi_code": "1063049",
        "region": "CAMPANIA",
        "province": "NA",
        "immobiliare_id": "napoli",
    },
    "bologna": {
        "name": "Bologna",
        "istat_code": "037006",
        "omi_code": "1037006",
        "region": "EMILIA-ROMAGNA",
        "province": "BO",
        "immobiliare_id": "bologna",
    },
    "torino": {
        "name": "Torino",
        "istat_code": "001272",
        "omi_code": "1001272",
        "region": "PIEMONTE",
        "province": "TO",
        "immobiliare_id": "torino",
    },
    "venezia": {
        "name": "Venezia",
        "istat_code": "027042",
        "omi_code": "1027042",
        "region": "VENETO",
        "province": "VE",
        "immobiliare_id": "venezia",
    },
    "palermo": {
        "name": "Palermo",
        "istat_code": "082053",
        "omi_code": "1082053",
        "region": "SICILIA",
        "province": "PA",
        "immobiliare_id": "palermo",
    },
    "genova": {
        "name": "Genova",
        "istat_code": "010025",
        "omi_code": "1010025",
        "region": "LIGURIA",
        "province": "GE",
        "immobiliare_id": "genova",
    },
    "bari": {
        "name": "Bari",
        "istat_code": "072006",
        "omi_code": "1072006",
        "region": "PUGLIA",
        "province": "BA",
        "immobiliare_id": "bari",
    },
}

# Rate limiting
DELAY_BETWEEN_REQUESTS = 3.0  # seconds


@dataclass
class ListingStats:
    """Aggregate statistics for a city's listings."""
    city: str
    listing_type: str
    count: int
    price_min: float
    price_max: float
    price_median: float
    price_mean: float
    price_sqm_median: float
    price_sqm_mean: float
    sqm_median: float
    fetch_date: str


class ImmobiliareItFetcher:
    """Fetcher for Immobiliare.it real estate listings.

    This fetcher retrieves listing data for major Italian cities
    to validate the price prediction model. It uses Immobiliare.it's
    public pages and extracts aggregate statistics.

    Example:
        >>> fetcher = ImmobiliareItFetcher()
        >>> sales = fetcher.fetch_all_cities(listing_type="sale")
        >>> print(sales)
    """

    BASE_URL = "https://www.immobiliare.it"

    # Headers to mimic a browser request
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    def __init__(self, cache_dir: Path = Path("data/raw/immobiliare")):
        """Initialize fetcher.

        Args:
            cache_dir: Directory to cache fetched data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < DELAY_BETWEEN_REQUESTS:
            time.sleep(DELAY_BETWEEN_REQUESTS - elapsed)
        self._last_request_time = time.time()

    def _get_search_url(self, city: str, listing_type: str, page: int = 1) -> str:
        """Build search URL for a city.

        Args:
            city: City key from CITY_CONFIGS
            listing_type: 'sale' or 'rent'
            page: Page number (1-indexed)

        Returns:
            Full URL for the search
        """
        config = CITY_CONFIGS[city]
        city_id = config["immobiliare_id"]

        if listing_type == "sale":
            path = f"/vendita-case/{city_id}/"
        else:
            path = f"/affitto-case/{city_id}/"

        if page > 1:
            path += f"?pag={page}"

        return f"{self.BASE_URL}{path}"

    def _extract_listing_data(self, html: str) -> list[dict]:
        """Extract listing data from HTML response.

        Parses the search results page to extract price and size information.
        Uses regex patterns to find JSON-LD data or listing cards.

        Args:
            html: Raw HTML content

        Returns:
            List of listing dictionaries
        """
        listings = []

        # Try to find JSON-LD structured data first (most reliable)
        json_ld_pattern = r'<script type="application/ld\+json">(.*?)</script>'
        json_matches = re.findall(json_ld_pattern, html, re.DOTALL)

        for match in json_matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and data.get("@type") == "ItemList":
                    for item in data.get("itemListElement", []):
                        if "item" in item:
                            listing = item["item"]
                            if "offers" in listing:
                                price = listing["offers"].get("price")
                                if price:
                                    listings.append({
                                        "price": float(price),
                                        "name": listing.get("name", ""),
                                        "url": listing.get("url", ""),
                                    })
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        # Fallback: extract from listing cards using patterns
        if not listings:
            # Pattern for price in listing cards
            price_pattern = r'data-price="(\d+)"'
            sqm_pattern = r'(\d+)\s*m²'

            prices = re.findall(price_pattern, html)
            sqms = re.findall(sqm_pattern, html)

            # Match prices with sizes
            for i, price in enumerate(prices[:50]):  # Limit to first 50
                listing = {"price": float(price)}
                if i < len(sqms):
                    listing["sqm"] = float(sqms[i])
                    listing["price_sqm"] = listing["price"] / listing["sqm"]
                listings.append(listing)

        # Alternative: try meta og:price patterns
        if not listings:
            og_price_pattern = r'€\s*([\d.,]+)'
            prices = re.findall(og_price_pattern, html)
            for price_str in prices[:50]:
                try:
                    price = float(price_str.replace(".", "").replace(",", "."))
                    if price > 10000:  # Filter out non-price matches
                        listings.append({"price": price})
                except ValueError:
                    continue

        return listings

    def _extract_listing_count(self, html: str) -> int:
        """Extract total listing count from search results.

        Args:
            html: Raw HTML content

        Returns:
            Total number of listings found
        """
        # Pattern: "1.234 annunci" or "1234 risultati"
        count_pattern = r'([\d.]+)\s*(?:annunci|risultati|immobili)'
        match = re.search(count_pattern, html, re.IGNORECASE)
        if match:
            count_str = match.group(1).replace(".", "")
            return int(count_str)
        return 0

    def fetch_city_listings(
        self,
        city: str,
        listing_type: Literal["sale", "rent"] = "sale",
        max_pages: int = 5,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch listings for a specific city.

        Args:
            city: City key (e.g., 'roma', 'milano')
            listing_type: 'sale' for purchases, 'rent' for rentals
            max_pages: Maximum pages to fetch (each ~25 listings)
            force_refresh: Bypass cache if True

        Returns:
            DataFrame with listing data
        """
        if city not in CITY_CONFIGS:
            raise ValueError(f"Unknown city: {city}. Available: {list(CITY_CONFIGS.keys())}")

        cache_file = self.cache_dir / f"{city}_{listing_type}s.csv"

        if cache_file.exists() and not force_refresh:
            logger.info(f"Loading cached {listing_type} listings for {city}")
            return pd.read_csv(cache_file)

        config = CITY_CONFIGS[city]
        logger.info(f"Fetching {listing_type} listings for {config['name']}...")

        all_listings = []

        for page in range(1, max_pages + 1):
            self._rate_limit()

            url = self._get_search_url(city, listing_type, page)
            logger.debug(f"Fetching page {page}: {url}")

            try:
                response = self._session.get(url, timeout=30)
                response.raise_for_status()

                listings = self._extract_listing_data(response.text)

                if not listings:
                    logger.debug(f"No more listings found at page {page}")
                    break

                # Add metadata
                for listing in listings:
                    listing["city"] = city
                    listing["city_name"] = config["name"]
                    listing["listing_type"] = listing_type
                    listing["istat_code"] = config["istat_code"]
                    listing["omi_code"] = config["omi_code"]
                    listing["region"] = config["region"]
                    listing["province"] = config["province"]
                    listing["page"] = page

                all_listings.extend(listings)

                # Check if we got fewer listings than expected (last page)
                if len(listings) < 20:
                    break

            except requests.RequestException as e:
                logger.warning(f"Failed to fetch page {page} for {city}: {e}")
                break

        if not all_listings:
            logger.warning(f"No listings found for {city}")
            return pd.DataFrame()

        df = pd.DataFrame(all_listings)
        df["fetch_date"] = pd.Timestamp.now().isoformat()

        # Calculate price_sqm if we have both price and sqm
        if "sqm" in df.columns and "price_sqm" not in df.columns:
            df["price_sqm"] = df["price"] / df["sqm"]

        # Cache results
        df.to_csv(cache_file, index=False)
        logger.info(f"Cached {len(df)} {listing_type} listings for {city}")

        return df

    def fetch_all_cities(
        self,
        listing_type: Literal["sale", "rent"] = "sale",
        max_pages_per_city: int = 5,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch listings for all configured cities.

        Args:
            listing_type: 'sale' or 'rent'
            max_pages_per_city: Maximum pages per city
            force_refresh: Bypass cache if True

        Returns:
            Combined DataFrame for all cities
        """
        all_data = []

        for city in CITY_CONFIGS:
            try:
                df = self.fetch_city_listings(
                    city=city,
                    listing_type=listing_type,
                    max_pages=max_pages_per_city,
                    force_refresh=force_refresh,
                )
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Failed to fetch {city}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        # Save combined file
        combined_file = self.cache_dir / f"all_cities_{listing_type}s.csv"
        combined.to_csv(combined_file, index=False)
        logger.info(f"Saved combined {listing_type} data: {len(combined)} listings")

        return combined

    def fetch_all(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch all data (sales and rentals for all cities).

        Args:
            force_refresh: Bypass cache if True

        Returns:
            Dictionary with 'sales' and 'rentals' DataFrames
        """
        return {
            "sales": self.fetch_all_cities("sale", force_refresh=force_refresh),
            "rentals": self.fetch_all_cities("rent", force_refresh=force_refresh),
        }

    def get_city_stats(self, city: str) -> dict | None:
        """Get quick stats for a city without full fetch.

        Fetches just the first page to get listing count and sample prices.

        Args:
            city: City key

        Returns:
            Dictionary with stats or None if failed
        """
        if city not in CITY_CONFIGS:
            return None

        config = CITY_CONFIGS[city]

        try:
            self._rate_limit()
            url = self._get_search_url(city, "sale", 1)
            response = self._session.get(url, timeout=30)
            response.raise_for_status()

            count = self._extract_listing_count(response.text)
            listings = self._extract_listing_data(response.text)

            if listings:
                prices = [item["price"] for item in listings if "price" in item]
                return {
                    "city": city,
                    "name": config["name"],
                    "total_listings": count,
                    "sample_count": len(prices),
                    "price_min": min(prices) if prices else None,
                    "price_max": max(prices) if prices else None,
                    "price_median": sorted(prices)[len(prices)//2] if prices else None,
                }
        except Exception as e:
            logger.warning(f"Failed to get stats for {city}: {e}")

        return None


def aggregate_by_city(listings_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate listing statistics by city.

    Args:
        listings_df: Raw listings DataFrame

    Returns:
        DataFrame with one row per city containing aggregate stats
    """
    if listings_df.empty:
        return pd.DataFrame()

    # Group by city and calculate statistics
    agg_funcs = {
        "price": ["count", "min", "max", "median", "mean", "std"],
    }

    if "price_sqm" in listings_df.columns:
        agg_funcs["price_sqm"] = ["median", "mean", "std"]

    if "sqm" in listings_df.columns:
        agg_funcs["sqm"] = ["median", "mean"]

    result = listings_df.groupby([
        "city", "city_name", "istat_code", "omi_code", "region", "province", "listing_type"
    ]).agg(agg_funcs).reset_index()

    # Flatten column names
    result.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in result.columns
    ]

    return result


def get_city_config(city: str) -> dict | None:
    """Get configuration for a city.

    Args:
        city: City key

    Returns:
        City configuration dict or None
    """
    return CITY_CONFIGS.get(city)


def list_available_cities() -> list[str]:
    """Get list of available city keys.

    Returns:
        List of city keys
    """
    return list(CITY_CONFIGS.keys())
