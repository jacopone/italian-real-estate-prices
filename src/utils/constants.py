"""Constants and mappings for Italian administrative geography.

This module centralizes all ISTAT codes, region/province mappings, and
other constants used throughout the pipeline. Using named constants
instead of magic numbers makes the code self-documenting.

Usage:
    from src.utils.constants import REGION_CODES, MAJOR_CITIES, haversine_distance
"""

from dataclasses import dataclass
from typing import Final

import numpy as np


# =============================================================================
# ISTAT REGION CODES
# =============================================================================

# Region code to name mapping (ISTAT standard)
REGION_CODES: Final[dict[int, str]] = {
    1: "Piemonte",
    2: "Valle d'Aosta",
    3: "Lombardia",
    4: "Trentino-Alto Adige",
    5: "Veneto",
    6: "Friuli-Venezia Giulia",
    7: "Liguria",
    8: "Emilia-Romagna",
    9: "Toscana",
    10: "Umbria",
    11: "Marche",
    12: "Lazio",
    13: "Abruzzo",
    14: "Molise",
    15: "Campania",
    16: "Puglia",
    17: "Basilicata",
    18: "Calabria",
    19: "Sicilia",
    20: "Sardegna",
}

# Reverse mapping: name to code
REGION_NAME_TO_CODE: Final[dict[str, int]] = {v: k for k, v in REGION_CODES.items()}

# Macro-regions for geographic analysis
MACRO_REGIONS: Final[dict[str, list[int]]] = {
    "Nord-Ovest": [1, 2, 3, 7],  # Piemonte, Valle d'Aosta, Lombardia, Liguria
    "Nord-Est": [4, 5, 6, 8],  # Trentino, Veneto, Friuli, Emilia-Romagna
    "Centro": [9, 10, 11, 12],  # Toscana, Umbria, Marche, Lazio
    "Sud": [13, 14, 15, 16, 17, 18],  # Abruzzo to Calabria
    "Isole": [19, 20],  # Sicilia, Sardegna
}

# Northern regions (for binary feature)
NORTHERN_REGIONS: Final[set[int]] = {1, 2, 3, 4, 5, 6, 7, 8}


# =============================================================================
# MAJOR CITIES (for distance calculations)
# =============================================================================


@dataclass(frozen=True)
class City:
    """Immutable city coordinates."""

    name: str
    lat: float
    lon: float
    istat_code: str  # 6-digit ISTAT code


# Major economic centers
MAJOR_CITIES: Final[dict[str, City]] = {
    "Milano": City("Milano", 45.4642, 9.1900, "015146"),
    "Roma": City("Roma", 41.9028, 12.4964, "058091"),
    "Napoli": City("Napoli", 40.8518, 14.2681, "063049"),
    "Torino": City("Torino", 45.0703, 7.6869, "001272"),
    "Firenze": City("Firenze", 43.7696, 11.2558, "048017"),
    "Bologna": City("Bologna", 44.4949, 11.3426, "037006"),
    "Venezia": City("Venezia", 45.4408, 12.3155, "027042"),
    "Genova": City("Genova", 44.4056, 8.9463, "010025"),
    "Palermo": City("Palermo", 38.1157, 13.3615, "082053"),
    "Bari": City("Bari", 41.1171, 16.8719, "072006"),
}

# Primary reference city for distance calculations
PRIMARY_CITY: Final[City] = MAJOR_CITIES["Milano"]


# =============================================================================
# OMI PROPERTY TYPES
# =============================================================================

# Property type codes used in OMI data
OMI_PROPERTY_TYPES: Final[dict[int, str]] = {
    20: "Abitazioni civili",
    21: "Abitazioni di tipo economico",
    22: "Ville e Villini",
    13: "Box",
    14: "Posti auto coperti",
    15: "Posti auto scoperti",
    16: "Autorimesse",
    30: "Negozi",
    31: "Centri commerciali",
    40: "Uffici",
    41: "Istituti di credito",
    50: "Capannoni industriali",
    51: "Capannoni artigianali",
    52: "Laboratori",
}

# Residential property types for price analysis
RESIDENTIAL_TYPES: Final[set[int]] = {20, 21, 22}

# Zone type hierarchy (B is most central)
OMI_ZONE_HIERARCHY: Final[list[str]] = ["B", "C", "D", "E", "R"]


# =============================================================================
# COLUMN NAME MAPPINGS
# =============================================================================

# Standardized column names (Italian -> English)
COLUMN_MAPPINGS: Final[dict[str, str]] = {
    # OMI columns
    "Comune_ISTAT": "istat_code",
    "Comune_descrizione": "municipality_name",
    "Compr_min": "price_min",
    "Compr_max": "price_max",
    "Loc_min": "rent_min",
    "Loc_max": "rent_max",
    "Descr_Tipologia": "property_type",
    "Fascia": "zone_type",
    # ISTAT columns
    "pro_com_t": "istat_code",
    "comune": "municipality_name",
    "den_prov": "province_name",
    "den_reg": "region_name",
    "cod_reg": "region_code",
    # Common
    "anno": "year",
    "popolazione": "population",
    "lat": "latitude",
    "long": "longitude",
}


# =============================================================================
# AIRBNB / INSIDEAIRBNB
# =============================================================================

# InsideAirbnb data URLs (as of 2024)
INSIDEAIRBNB_CITIES: Final[dict[str, dict[str, str]]] = {
    "milan": {
        "name": "Milan",
        "url": "http://data.insideairbnb.com/italy/lombardy/milan",
        "province_code": "015",
    },
    "florence": {
        "name": "Florence",
        "url": "http://data.insideairbnb.com/italy/tuscany/florence",
        "province_code": "048",
    },
    "bologna": {
        "name": "Bologna",
        "url": "http://data.insideairbnb.com/italy/emilia-romagna/bologna",
        "province_code": "037",
    },
    "naples": {
        "name": "Naples",
        "url": "http://data.insideairbnb.com/italy/campania/naples",
        "province_code": "063",
    },
    "rome": {
        "name": "Rome",
        "url": "http://data.insideairbnb.com/italy/lazio/rome",
        "province_code": "058",
    },
    "venice": {
        "name": "Venice",
        "url": "http://data.insideairbnb.com/italy/veneto/venice",
        "province_code": "027",
    },
}


# =============================================================================
# GEOGRAPHIC UTILITIES
# =============================================================================

# Earth radius in kilometers
EARTH_RADIUS_KM: Final[float] = 6371.0


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate great-circle distance between two points.

    Uses the Haversine formula for accurate distance on a sphere.

    Args:
        lat1, lon1: First point coordinates (degrees).
        lat2, lon2: Second point coordinates (degrees).

    Returns:
        Distance in kilometers.

    Example:
        >>> haversine_distance(45.4642, 9.1900, 41.9028, 12.4964)  # Milano to Roma
        477.5  # approximately
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


def distance_to_nearest_major_city(lat: float, lon: float) -> tuple[float, str]:
    """Calculate distance to nearest major city.

    Args:
        lat, lon: Point coordinates (degrees).

    Returns:
        Tuple of (distance_km, city_name).
    """
    min_dist = float("inf")
    nearest_city = ""

    for city in MAJOR_CITIES.values():
        dist = haversine_distance(lat, lon, city.lat, city.lon)
        if dist < min_dist:
            min_dist = dist
            nearest_city = city.name

    return min_dist, nearest_city


def distance_to_milan(lat: float, lon: float) -> float:
    """Calculate distance to Milan (primary economic center).

    Args:
        lat, lon: Point coordinates (degrees).

    Returns:
        Distance in kilometers.
    """
    return haversine_distance(lat, lon, PRIMARY_CITY.lat, PRIMARY_CITY.lon)


# =============================================================================
# MODEL FEATURE SETS
# =============================================================================

# Core features used in baseline model (without STR)
BASELINE_FEATURES: Final[list[str]] = [
    "log_population",
    "log_income",
    "pop_change_pct",
    "income_change_pct",
    "lat",
    "long",
    "dist_major_city",
    "dist_coast",
    "coastal",
    "northern",
    "alpine_zone",
    "urban",
    "tourism_intensity",
]

# Features added when STR data is available
STR_FEATURES: Final[list[str]] = [
    "log_str_density",
    "str_premium",
]

# Full feature set for STR-enhanced model
FULL_FEATURES: Final[list[str]] = BASELINE_FEATURES + STR_FEATURES


# =============================================================================
# ANALYSIS THRESHOLDS
# =============================================================================

# Population thresholds for urban classification
POPULATION_THRESHOLDS: Final[dict[str, int]] = {
    "village": 5_000,
    "town": 20_000,
    "small_city": 100_000,
    "large_city": 500_000,
}

# Tourism intensity thresholds (arrivals per 1000 residents)
TOURISM_THRESHOLDS: Final[dict[str, float]] = {
    "low": 100,
    "medium": 500,
    "high": 2000,
    "very_high": 10000,
}

# Undervaluation categories
VALUATION_CATEGORIES: Final[dict[str, tuple[float, float]]] = {
    "severely_undervalued": (-1.0, -0.30),  # More than 30% below fair value
    "undervalued": (-0.30, -0.15),  # 15-30% below
    "fair_value": (-0.15, 0.15),  # Within 15%
    "overvalued": (0.15, 0.30),  # 15-30% above
    "severely_overvalued": (0.30, 1.0),  # More than 30% above
}
