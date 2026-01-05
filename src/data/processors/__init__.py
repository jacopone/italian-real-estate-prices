"""Data processors for each source.

Each processor handles loading, cleaning, and transforming data from a specific source.
Processors follow a consistent interface and produce validated DataFrames.
"""

from src.data.processors.airbnb import AirbnbProcessor
from src.data.processors.irpef import IRPEFProcessor
from src.data.processors.istat import ISTATProcessor
from src.data.processors.omi import OMIProcessor
from src.data.processors.tourism import TourismProcessor

__all__ = [
    "OMIProcessor",
    "ISTATProcessor",
    "IRPEFProcessor",
    "TourismProcessor",
    "AirbnbProcessor",
]
