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
from src.features.vacancy_classifier import (
    VacancyClassifier,
    VacancyThresholds,
    VacancyType,
    create_vacancy_features,
    get_vacancy_risk_score,
)
from src.features.vacancy_features import (
    create_vacancy_model_features,
    get_vacancy_feature_importance,
    prepare_model_features,
)

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
    # Vacancy
    "VacancyClassifier",
    "VacancyThresholds",
    "VacancyType",
    "create_vacancy_features",
    "get_vacancy_risk_score",
    # Vacancy Model Features
    "create_vacancy_model_features",
    "prepare_model_features",
    "get_vacancy_feature_importance",
]
