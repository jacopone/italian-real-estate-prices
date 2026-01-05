"""Model evaluation and valuation analysis.

Provides metrics computation, residual diagnostics, and
undervaluation detection for investment analysis.
"""

from src.evaluation.metrics import (
    RegressionMetrics,
    ResidualDiagnostics,
    analyze_residuals,
    compare_models,
    compute_adjusted_r2,
    compute_feature_importance_ranking,
    compute_regression_metrics,
)
from src.evaluation.valuation import (
    ValuationAnalyzer,
    ValuationResult,
    compute_smart_picks,
    identify_undervalued_municipalities,
)

__all__ = [
    # Metrics
    "RegressionMetrics",
    "ResidualDiagnostics",
    "compute_regression_metrics",
    "analyze_residuals",
    "compute_adjusted_r2",
    "compare_models",
    "compute_feature_importance_ranking",
    # Valuation
    "ValuationAnalyzer",
    "ValuationResult",
    "identify_undervalued_municipalities",
    "compute_smart_picks",
]
