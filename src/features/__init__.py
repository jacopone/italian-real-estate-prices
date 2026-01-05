"""Feature engineering module.

Provides transformers for computing features from raw data,
following the scikit-learn fit/transform pattern.
"""

from src.features.base import (
    FeatureTransformer,
    IdentityTransformer,
    LogTransformer,
    StandardScaler,
    TransformerMetadata,
)
from src.features.demographic import AgingFeatures, DemographicFeatures
from src.features.economic import EconomicIndicators, IncomeFeatures
from src.features.geographic import CoordinateFeatures, DistanceFeatures, LocationFlags
from src.features.pipeline import (
    FeaturePipeline,
    FeaturePipelineResult,
    ModelReadyPipeline,
    create_features,
)
from src.features.tourism import STRFeatures, STRProxyFeatures, TourismFeatures

__all__ = [
    # Base
    "FeatureTransformer",
    "TransformerMetadata",
    "IdentityTransformer",
    "LogTransformer",
    "StandardScaler",
    # Demographic
    "DemographicFeatures",
    "AgingFeatures",
    # Economic
    "IncomeFeatures",
    "EconomicIndicators",
    # Geographic
    "DistanceFeatures",
    "LocationFlags",
    "CoordinateFeatures",
    # Tourism
    "TourismFeatures",
    "STRFeatures",
    "STRProxyFeatures",
    # Pipeline
    "FeaturePipeline",
    "FeaturePipelineResult",
    "ModelReadyPipeline",
    "create_features",
]
