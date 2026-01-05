"""Configuration management using Pydantic models.

This module defines all configuration for the Italian Real Estate Risk Model.
Configuration is loaded from YAML files and validated at startup.

Usage:
    from src.config import load_config
    config = load_config("configs/default.yaml")
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class PathConfig(BaseModel):
    """Paths configuration for data directories."""

    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_dir: Path = Field(
        default=Path("data/processed"), description="Processed data directory"
    )
    outputs_dir: Path = Field(default=Path("outputs"), description="Outputs directory")
    models_dir: Path = Field(default=Path("models"), description="Saved models directory")

    @field_validator("data_dir", "raw_dir", "processed_dir", "outputs_dir", "models_dir")
    @classmethod
    def ensure_path(cls, v: Path | str) -> Path:
        """Convert string to Path if needed."""
        return Path(v) if isinstance(v, str) else v


class OMIConfig(BaseModel):
    """Configuration for OMI (Osservatorio Mercato Immobiliare) data processing."""

    # Property type filters
    property_types: list[str] = Field(
        default=["Abitazioni civili", "Abitazioni di tipo economico", "Ville e Villini"],
        description="Property types to include in analysis",
    )

    # Zone type mappings (B=central, C=semi-central, etc.)
    zone_priority: list[str] = Field(
        default=["B", "C", "D", "E", "R"],
        description="Zone types in priority order for aggregation",
    )

    # Year range for analysis
    min_year: int = Field(default=2015, ge=2004, description="Earliest year to include")
    max_year: int = Field(default=2024, le=2030, description="Latest year to include")

    # Price filtering
    min_price_sqm: float = Field(default=200.0, ge=0, description="Minimum price EUR/sqm")
    max_price_sqm: float = Field(default=20000.0, ge=0, description="Maximum price EUR/sqm")


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""

    # Log transformation targets
    log_transform_columns: list[str] = Field(
        default=["price_mid", "rent_mid", "income_avg", "population", "str_density"],
        description="Columns to log-transform (log1p for zero-safe)",
    )

    # Features to include in models
    demographic_features: list[str] = Field(
        default=["population_change_pct", "aging_index", "dependency_ratio"],
        description="Demographic features to compute",
    )

    economic_features: list[str] = Field(
        default=["income_avg", "income_per_capita", "income_gini"],
        description="Economic features to compute",
    )

    geographic_features: list[str] = Field(
        default=["lat", "lon", "distance_to_milan", "altitude"],
        description="Geographic features to include",
    )

    tourism_features: list[str] = Field(
        default=["tourism_intensity", "str_density", "airbnb_premium"],
        description="Tourism-related features to compute",
    )

    # Include STR (short-term rental) proxy
    include_str_proxy: bool = Field(
        default=True, description="Whether to compute STR proxy for non-Airbnb provinces"
    )


class ModelConfig(BaseModel):
    """Configuration for model training."""

    # Target variables
    price_target: str = Field(default="log_price_mid", description="Target for price model")
    rent_target: str = Field(default="log_rent_mid", description="Target for rent model")

    # Train/test split
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set proportion")
    random_state: int = Field(default=42, description="Random seed for reproducibility")

    # Cross-validation
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of CV folds")

    # Gradient Boosting hyperparameters (tuned values from analysis)
    gb_params: dict = Field(
        default_factory=lambda: {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "subsample": 0.8,
            "max_features": "sqrt",
            "random_state": 42,
        },
        description="Gradient Boosting hyperparameters",
    )

    # OLS regularization
    ols_alpha: float = Field(default=0.01, ge=0, description="Ridge regularization strength")


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation and valuation detection."""

    # Performance thresholds
    min_r2_price: float = Field(
        default=0.80, ge=0, le=1.0, description="Minimum acceptable R² for price model"
    )
    min_r2_rent: float = Field(
        default=0.70, ge=0, le=1.0, description="Minimum acceptable R² for rent model"
    )

    # Undervaluation detection
    undervaluation_threshold: float = Field(
        default=-0.15, le=0, description="Residual threshold for undervaluation (negative = cheaper)"
    )

    # Smart picks criteria
    min_yield_pct: float = Field(
        default=4.0, ge=0, description="Minimum gross yield % for smart picks"
    )
    max_price_gap_pct: float = Field(
        default=-15.0, le=0, description="Maximum price gap % (more negative = more undervalued)"
    )


class VisualizationConfig(BaseModel):
    """Configuration for visualizations and maps."""

    # Map settings
    map_crs: str = Field(default="EPSG:4326", description="Coordinate reference system")
    figsize_map: tuple[int, int] = Field(default=(12, 10), description="Figure size for maps")
    figsize_chart: tuple[int, int] = Field(default=(10, 6), description="Figure size for charts")

    # Color schemes
    colormap_sequential: str = Field(default="RdYlGn_r", description="Sequential colormap")
    colormap_diverging: str = Field(default="RdBu_r", description="Diverging colormap")

    # DPI for exports
    dpi: int = Field(default=150, ge=72, le=300, description="Resolution for saved figures")


class AirbnbConfig(BaseModel):
    """Configuration for Airbnb/short-term rental data."""

    # Cities with available data
    available_cities: list[str] = Field(
        default=["milan", "florence", "bologna", "naples"],
        description="Cities with InsideAirbnb data",
    )

    # Filtering
    min_reviews: int = Field(default=0, ge=0, description="Minimum reviews for listing inclusion")

    # Premium calculation
    days_per_month: int = Field(
        default=20, ge=1, le=30, description="Assumed occupied days for monthly revenue"
    )


class Config(BaseModel):
    """Root configuration model combining all config sections."""

    paths: PathConfig = Field(default_factory=PathConfig)
    omi: OMIConfig = Field(default_factory=OMIConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    airbnb: AirbnbConfig = Field(default_factory=AirbnbConfig)

    # Metadata
    project_name: str = Field(
        default="Italian Real Estate Demographic Risk Model",
        description="Project name for reports",
    )
    analysis_year: int = Field(default=2023, description="Primary year for analysis")


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config values are invalid.
    """
    if config_path is None:
        return Config()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw_config = yaml.safe_load(f)

    return Config.model_validate(raw_config or {})


def save_config(config: Config, config_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Config object to save.
        config_path: Destination path.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, handling Path objects
    config_dict = config.model_dump()

    def convert_paths(d: dict) -> dict:
        """Recursively convert Path objects to strings."""
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
            elif isinstance(value, dict):
                d[key] = convert_paths(value)
        return d

    config_dict = convert_paths(config_dict)

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
