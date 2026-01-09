"""Data fetchers for Italian real estate price analysis."""

from .immobiliare_it import CITY_CONFIGS, ImmobiliareItFetcher, aggregate_by_city
from .inside_airbnb import InsideAirbnbFetcher
from .istat import ISTATFetcher
from .omi import OMIFetcher

__all__ = [
    "ISTATFetcher",
    "InsideAirbnbFetcher",
    "OMIFetcher",
    "ImmobiliareItFetcher",
    "aggregate_by_city",
    "CITY_CONFIGS",
]
