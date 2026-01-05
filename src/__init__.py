"""Italian Real Estate Demographic Risk Model.

A machine learning system for analyzing Italian real estate prices,
identifying undervalued municipalities, and computing investment metrics.

Key Features:
- Hedonic pricing model using demographic, economic, and geographic features
- Short-term rental (Airbnb) integration for tourism market analysis
- Gradient Boosting regression achieving R² > 0.83
- Valuation analysis with smart investment picks

Usage:
    from src.config import load_config
    from src.features import create_features
    from src.models import GradientBoostingModel
    from src.evaluation import compute_smart_picks

Example:
    >>> config = load_config("configs/default.yaml")
    >>> features = create_features(Path("data"), config)
    >>> model = GradientBoostingModel.from_config(config.model)
    >>> model.fit(X_train, y_train)
"""

__version__ = "0.2.0"
__author__ = "Italian Real Estate Analysis Team"

# Main configuration
from src.config import Config, load_config

# Convenience imports
from src.features.pipeline import create_features
from src.models import GradientBoostingModel, ModelTrainer
from src.evaluation import compute_smart_picks, ValuationAnalyzer

__all__ = [
    # Version
    "__version__",
    # Config
    "Config",
    "load_config",
    # High-level API
    "create_features",
    "GradientBoostingModel",
    "ModelTrainer",
    "compute_smart_picks",
    "ValuationAnalyzer",
]
