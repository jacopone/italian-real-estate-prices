"""Vacancy type classifier feature.

Classifies housing vacancy into:
- tourist_vacancy: Second homes, Airbnb, seasonal rentals (high tourism + stable pop)
- decline_vacancy: Abandonment, depopulation (low tourism + declining pop)
- mixed_vacancy: Both effects present

This helps separate "healthy" vacancy (tourism markets) from
"unhealthy" vacancy (declining markets) for price prediction models.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VacancyType(str, Enum):
    """Types of housing vacancy."""
    TOURIST = "tourist_vacancy"
    DECLINE = "decline_vacancy"
    MIXED = "mixed_vacancy"
    LOW = "low_vacancy"
    UNKNOWN = "unknown"


@dataclass
class VacancyThresholds:
    """Configurable thresholds for vacancy classification."""
    high_vacancy: float = 0.30  # > 30% non-occupied
    low_vacancy: float = 0.15  # < 15% non-occupied (healthy market)
    high_tourism: float = 10.0  # > 10 presenze per inhabitant per year
    low_tourism: float = 2.0  # < 2 presenze per inhabitant
    population_decline: float = -0.05  # > 5% decline over reference period
    population_growth: float = 0.02  # > 2% growth
    high_airbnb: float = 20.0  # > 20 listings per 1000 inhabitants
    low_airbnb: float = 5.0  # < 5 listings per 1000


class VacancyClassifier:
    """Classifies municipalities by vacancy type."""

    def __init__(self, thresholds: VacancyThresholds | None = None):
        self.thresholds = thresholds or VacancyThresholds()

    def classify(
        self,
        vacancy_rate: float,
        population_change: float,
        tourism_intensity: float | None = None,
        airbnb_density: float | None = None,
    ) -> VacancyType:
        """Classify a single location by vacancy type.

        Args:
            vacancy_rate: Proportion of non-occupied dwellings (0-1)
            population_change: Population change over reference period (e.g., 10 years)
            tourism_intensity: Tourist nights (presenze) per inhabitant per year
            airbnb_density: Airbnb listings per 1000 inhabitants

        Returns:
            VacancyType classification
        """
        t = self.thresholds

        # Low vacancy - healthy market
        if vacancy_rate < t.low_vacancy:
            return VacancyType.LOW

        # High vacancy - need to classify type
        if vacancy_rate >= t.high_vacancy:
            # Strong tourism signals
            has_tourism = (
                (tourism_intensity is not None and tourism_intensity >= t.high_tourism) or
                (airbnb_density is not None and airbnb_density >= t.high_airbnb)
            )

            # Population decline signal
            has_decline = population_change <= t.population_decline

            # Population stable/growing
            population_stable = population_change > t.population_decline

            if has_tourism and population_stable:
                return VacancyType.TOURIST
            elif has_decline and not has_tourism:
                return VacancyType.DECLINE
            elif has_tourism and has_decline:
                return VacancyType.MIXED
            elif has_decline:
                return VacancyType.DECLINE
            else:
                # High vacancy but no clear signal
                return VacancyType.MIXED

        # Moderate vacancy
        if tourism_intensity is not None and tourism_intensity >= t.high_tourism:
            return VacancyType.TOURIST
        elif population_change <= t.population_decline:
            return VacancyType.DECLINE
        else:
            return VacancyType.LOW

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        vacancy_col: str = "vacancy_rate",
        pop_change_col: str = "population_change",
        tourism_col: str | None = "tourism_intensity",
        airbnb_col: str | None = "airbnb_density",
    ) -> pd.DataFrame:
        """Classify all rows in a DataFrame.

        Args:
            df: Input DataFrame with required columns
            vacancy_col: Column name for vacancy rate
            pop_change_col: Column name for population change
            tourism_col: Column name for tourism intensity (optional)
            airbnb_col: Column name for Airbnb density (optional)

        Returns:
            DataFrame with vacancy_type column added
        """
        result = df.copy()

        def classify_row(row):
            try:
                vacancy = row.get(vacancy_col, np.nan)
                pop_change = row.get(pop_change_col, 0.0)
                tourism = row.get(tourism_col) if tourism_col else None
                airbnb = row.get(airbnb_col) if airbnb_col else None

                if pd.isna(vacancy):
                    return VacancyType.UNKNOWN.value

                return self.classify(
                    vacancy_rate=vacancy,
                    population_change=pop_change if not pd.isna(pop_change) else 0.0,
                    tourism_intensity=tourism if tourism is not None and not pd.isna(tourism) else None,
                    airbnb_density=airbnb if airbnb is not None and not pd.isna(airbnb) else None,
                ).value

            except Exception as e:
                logger.warning(f"Classification error: {e}")
                return VacancyType.UNKNOWN.value

        result["vacancy_type"] = result.apply(classify_row, axis=1)

        # Add numeric encoding for ML models
        type_encoding = {
            VacancyType.LOW.value: 0,
            VacancyType.TOURIST.value: 1,
            VacancyType.DECLINE.value: 2,
            VacancyType.MIXED.value: 3,
            VacancyType.UNKNOWN.value: -1,
        }
        result["vacancy_type_code"] = result["vacancy_type"].map(type_encoding)

        return result


def create_vacancy_features(
    housing_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    tourism_df: pd.DataFrame | None = None,
    airbnb_df: pd.DataFrame | None = None,
    territory_col: str = "Comune_ISTAT",
) -> pd.DataFrame:
    """Create vacancy classification features by merging data sources.

    Args:
        housing_df: ISTAT housing census data (may already include demographics/tourism)
        demographics_df: Population change data
        tourism_df: Tourism presenze data (optional)
        airbnb_df: Airbnb listings aggregated by area (optional)
        territory_col: Column to join on (ISTAT municipality code)

    Returns:
        DataFrame with vacancy classification features
    """
    logger.info("Creating vacancy features...")

    # Start with housing data
    result = housing_df.copy()

    # Log available columns
    available_features = []
    if "vacancy_rate" in result.columns:
        available_features.append("vacancy_rate")
    if "pop_10yr_change" in result.columns:
        available_features.append("pop_10yr_change")
    if "tourism_intensity" in result.columns:
        available_features.append("tourism_intensity")
    logger.info(f"  Available features in housing_df: {available_features}")

    # Check if demographics data is already in housing_df (from supplementary data)
    has_pop_change = "pop_10yr_change" in result.columns or "population_change" in result.columns

    # Merge demographics only if not already present
    if not has_pop_change and demographics_df is not None and not demographics_df.empty:
        demo_cols = [territory_col, "population_change", "pop_10yr_change"]
        available_cols = [c for c in demo_cols if c in demographics_df.columns]
        if available_cols:
            demo_subset = demographics_df[available_cols]
            result = result.merge(demo_subset, on=territory_col, how="left")

    # Check if tourism data is already in housing_df
    has_tourism = "tourism_intensity" in result.columns

    # Merge tourism only if not already present
    if not has_tourism and tourism_df is not None and not tourism_df.empty:
        tourism_cols = [territory_col, "tourism_intensity", "presenze", "arrivi"]
        available_cols = [c for c in tourism_cols if c in tourism_df.columns]
        if available_cols:
            tourism_subset = tourism_df[available_cols]
            result = result.merge(tourism_subset, on=territory_col, how="left")

    # Merge Airbnb if available and has matching column
    if airbnb_df is not None and not airbnb_df.empty:
        if territory_col in airbnb_df.columns:
            airbnb_cols = [territory_col, "airbnb_density", "listing_count", "listings_per_1000"]
            airbnb_subset = airbnb_df[[c for c in airbnb_cols if c in airbnb_df.columns]]

            # Rename for consistency
            if "listings_per_1000" in airbnb_subset.columns:
                airbnb_subset = airbnb_subset.rename(columns={"listings_per_1000": "airbnb_density"})

            result = result.merge(airbnb_subset, on=territory_col, how="left")
        else:
            logger.info(f"Airbnb data lacks {territory_col} column, skipping Airbnb merge")
            result["airbnb_density"] = None  # Will be filled for cities with data

    # Apply classifier
    classifier = VacancyClassifier()

    # Determine which columns are available
    pop_change_col = "pop_10yr_change" if "pop_10yr_change" in result.columns else "population_change"
    tourism_col = "tourism_intensity" if "tourism_intensity" in result.columns else None
    airbnb_col = "airbnb_density" if "airbnb_density" in result.columns else None

    result = classifier.classify_dataframe(
        result,
        vacancy_col="vacancy_rate",
        pop_change_col=pop_change_col,
        tourism_col=tourism_col,
        airbnb_col=airbnb_col,
    )

    logger.info(f"Classified {len(result)} municipalities:")
    logger.info(f"  Distribution: {result['vacancy_type'].value_counts().to_dict()}")

    return result


def get_vacancy_risk_score(
    vacancy_type: str,
    vacancy_rate: float,
    price_trend: float | None = None
) -> float:
    """Calculate a vacancy risk score for investment analysis.

    Higher score = higher risk for real estate investment.

    Args:
        vacancy_type: Classified vacancy type
        vacancy_rate: Raw vacancy rate
        price_trend: Price change over time (positive = appreciation)

    Returns:
        Risk score from 0 (low risk) to 1 (high risk)
    """
    base_scores = {
        VacancyType.LOW.value: 0.1,
        VacancyType.TOURIST.value: 0.3,  # Moderate risk (seasonal, regulatory)
        VacancyType.MIXED.value: 0.6,
        VacancyType.DECLINE.value: 0.9,  # High risk (depopulation)
        VacancyType.UNKNOWN.value: 0.5,
    }

    score = base_scores.get(vacancy_type, 0.5)

    # Adjust for vacancy rate intensity
    if vacancy_rate > 0.5:
        score = min(1.0, score + 0.2)
    elif vacancy_rate > 0.4:
        score = min(1.0, score + 0.1)

    # Adjust for price trend if available
    if price_trend is not None:
        if price_trend < -0.05:  # Declining prices
            score = min(1.0, score + 0.15)
        elif price_trend > 0.05:  # Rising prices
            score = max(0.0, score - 0.1)

    return score
