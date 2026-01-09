"""Find undervalued properties using the vacancy-aware price model.

Identifies municipalities where actual prices are below model predictions,
filtered for investment quality based on vacancy type and risk factors.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.vacancy_features import create_vacancy_model_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UndervaluedPropertyFinder:
    """Find undervalued properties using trained price model."""

    def __init__(
        self,
        model_path: Path = Path("outputs/price_model_with_vacancy.joblib"),
        data_path: Path = Path("data/processed/municipal_features.csv"),
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.df = None
        self.predictions = None

    def load_model(self):
        """Load the trained price prediction model."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info(f"Loaded {self.model.name} with {len(self.model.feature_names)} features")
        return self

    def load_data(self) -> pd.DataFrame:
        """Load and prepare municipal data."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)

        # Clean data
        self.df = self.df.dropna(subset=["price_avg"])
        self.df = self.df[(self.df["price_avg"] >= 100) & (self.df["price_avg"] <= 10000)]

        logger.info(f"Loaded {len(self.df)} municipalities")
        return self.df

    def prepare_features(self) -> pd.DataFrame:
        """Prepare features matching model training."""
        df = create_vacancy_model_features(self.df.copy(), include_interactions=True)

        # Get features used by model
        feature_names = self.model.feature_names

        # Check for missing features
        missing = [f for f in feature_names if f not in df.columns]

        if missing:
            logger.warning(f"Missing features: {missing}")
            for f in missing:
                df[f] = 0  # Fill missing with 0

        return df

    def predict_prices(self) -> pd.DataFrame:
        """Generate price predictions and calculate gaps."""
        df = self.prepare_features()

        # Get features in correct order
        X = df[self.model.feature_names]

        # Make predictions (log scale)
        log_predictions = self.model.predict(X)

        # Convert back to price scale
        df["predicted_price"] = np.expm1(log_predictions)

        # Calculate price gap (negative = undervalued)
        df["price_gap"] = df["price_avg"] - df["predicted_price"]
        df["price_gap_pct"] = (df["price_gap"] / df["predicted_price"]) * 100

        # Undervaluation score (more negative = more undervalued)
        df["undervaluation_score"] = -df["price_gap_pct"]

        self.predictions = df
        logger.info(f"Generated predictions for {len(df)} municipalities")

        return df

    def find_undervalued(
        self,
        min_undervaluation_pct: float = 15.0,
        max_vacancy_risk: float = 0.6,
        min_yield: float = 4.0,
        exclude_decline: bool = True,
        top_n: int = 50,
    ) -> pd.DataFrame:
        """Find undervalued properties with quality filters.

        Args:
            min_undervaluation_pct: Minimum % below predicted price
            max_vacancy_risk: Maximum acceptable vacancy risk score
            min_yield: Minimum gross yield %
            exclude_decline: Whether to exclude decline_vacancy areas
            top_n: Number of top results to return

        Returns:
            DataFrame of undervalued municipalities
        """
        if self.predictions is None:
            self.predict_prices()

        df = self.predictions.copy()

        # Filter criteria
        filters = []

        # 1. Undervaluation threshold
        undervalued_mask = df["price_gap_pct"] <= -min_undervaluation_pct
        filters.append(("undervaluation", undervalued_mask.sum()))

        # 2. Vacancy risk threshold
        if "vacancy_risk_score" in df.columns:
            risk_mask = df["vacancy_risk_score"] <= max_vacancy_risk
        else:
            risk_mask = pd.Series([True] * len(df))
        filters.append(("risk", risk_mask.sum()))

        # 3. Minimum yield
        if "gross_yield" in df.columns:
            yield_mask = df["gross_yield"] >= (min_yield / 100)
        else:
            yield_mask = pd.Series([True] * len(df))
        filters.append(("yield", yield_mask.sum()))

        # 4. Exclude decline areas
        if exclude_decline and "vacancy_type" in df.columns:
            decline_mask = df["vacancy_type"] != "decline_vacancy"
        else:
            decline_mask = pd.Series([True] * len(df))
        filters.append(("non-decline", decline_mask.sum()))

        # Apply all filters
        combined_mask = undervalued_mask & risk_mask & yield_mask & decline_mask
        filtered = df[combined_mask].copy()

        logger.info(f"Filter results: {filters}")
        logger.info(f"Found {len(filtered)} undervalued properties")

        # Sort by undervaluation score
        filtered = filtered.sort_values("undervaluation_score", ascending=False)

        # Select output columns
        output_cols = [
            "Comune_descrizione",
            "Regione",
            "Prov",
            "price_avg",
            "predicted_price",
            "price_gap",
            "price_gap_pct",
            "undervaluation_score",
            "gross_yield",
            "vacancy_rate",
            "vacancy_type",
            "vacancy_risk_score",
            "pop_10yr_change",
            "tourism_intensity",
        ]

        available_cols = [c for c in output_cols if c in filtered.columns]

        return filtered[available_cols].head(top_n)

    def find_smart_picks(self) -> pd.DataFrame:
        """Find smart investment picks: undervalued + low risk + tourist areas.

        These are properties that are:
        - At least 20% below predicted price
        - In tourist or low vacancy areas (not declining)
        - Have reasonable yield (> 5%)
        - Low vacancy risk (< 0.4)
        """
        return self.find_undervalued(
            min_undervaluation_pct=20.0,
            max_vacancy_risk=0.4,
            min_yield=5.0,
            exclude_decline=True,
            top_n=30,
        )

    def find_high_yield_opportunities(self) -> pd.DataFrame:
        """Find high yield opportunities with moderate risk tolerance.

        These are properties with:
        - High yield (> 6%)
        - Some undervaluation (> 10%)
        - Moderate risk tolerance (including mixed areas)
        """
        return self.find_undervalued(
            min_undervaluation_pct=10.0,
            max_vacancy_risk=0.7,
            min_yield=6.0,
            exclude_decline=True,
            top_n=30,
        )

    def find_tourist_area_deals(self) -> pd.DataFrame:
        """Find undervalued properties in tourist areas.

        Tourist vacancy areas often have seasonal dynamics that
        can create buying opportunities.
        """
        if self.predictions is None:
            self.predict_prices()

        df = self.predictions.copy()

        # Filter for tourist areas
        tourist_mask = df["vacancy_type"] == "tourist_vacancy"
        undervalued_mask = df["price_gap_pct"] <= -10.0

        filtered = df[tourist_mask & undervalued_mask].copy()
        filtered = filtered.sort_values("undervaluation_score", ascending=False)

        output_cols = [
            "Comune_descrizione",
            "Regione",
            "Prov",
            "price_avg",
            "predicted_price",
            "price_gap_pct",
            "gross_yield",
            "vacancy_rate",
            "tourism_intensity",
        ]

        available_cols = [c for c in output_cols if c in filtered.columns]
        return filtered[available_cols].head(30)

    def generate_report(self) -> dict:
        """Generate comprehensive undervaluation report."""
        if self.predictions is None:
            self.predict_prices()

        df = self.predictions

        # Overall statistics
        report = {
            "total_municipalities": len(df),
            "undervalued_count": (df["price_gap_pct"] <= -15).sum(),
            "overvalued_count": (df["price_gap_pct"] >= 15).sum(),
            "fairly_valued_count": ((df["price_gap_pct"] > -15) & (df["price_gap_pct"] < 15)).sum(),
        }

        # By vacancy type
        if "vacancy_type" in df.columns:
            by_type = df.groupby("vacancy_type").agg({
                "price_gap_pct": ["mean", "std", "count"],
                "undervaluation_score": "mean",
            }).round(2)
            report["by_vacancy_type"] = by_type.to_dict()

        # By region
        if "Regione" in df.columns:
            by_region = df.groupby("Regione").agg({
                "price_gap_pct": "mean",
                "undervaluation_score": "mean",
                "price_avg": "mean",
            }).round(2)
            report["undervalued_regions"] = by_region.nsmallest(5, "price_gap_pct").to_dict()
            report["overvalued_regions"] = by_region.nlargest(5, "price_gap_pct").to_dict()

        # Top undervalued
        report["top_undervalued"] = self.find_undervalued(top_n=10).to_dict("records")
        report["smart_picks"] = self.find_smart_picks().to_dict("records")

        return report

    def save_results(self, output_dir: Path = Path("outputs")):
        """Save undervaluation analysis results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full predictions
        if self.predictions is not None:
            self.predictions.to_csv(output_dir / "price_predictions.csv", index=False)

        # Save undervalued list
        undervalued = self.find_undervalued(top_n=100)
        undervalued.to_csv(output_dir / "undervalued_properties.csv", index=False)

        # Save smart picks
        smart = self.find_smart_picks()
        smart.to_csv(output_dir / "smart_picks.csv", index=False)

        # Save tourist deals
        tourist = self.find_tourist_area_deals()
        tourist.to_csv(output_dir / "tourist_area_deals.csv", index=False)

        logger.info(f"Results saved to {output_dir}")


def main():
    """Find undervalued properties using vacancy-aware model."""
    finder = UndervaluedPropertyFinder()
    finder.load_model()
    finder.load_data()
    finder.predict_prices()

    # Print summary statistics
    df = finder.predictions
    print("\n" + "="*80)
    print("PRICE PREDICTION ANALYSIS")
    print("="*80)

    print(f"\nTotal municipalities analyzed: {len(df)}")
    print("\nPrice Gap Distribution:")
    print(f"  Significantly undervalued (<-20%): {(df['price_gap_pct'] <= -20).sum()}")
    print(f"  Moderately undervalued (-20% to -10%): {((df['price_gap_pct'] > -20) & (df['price_gap_pct'] <= -10)).sum()}")
    print(f"  Fairly valued (-10% to +10%): {((df['price_gap_pct'] > -10) & (df['price_gap_pct'] < 10)).sum()}")
    print(f"  Moderately overvalued (+10% to +20%): {((df['price_gap_pct'] >= 10) & (df['price_gap_pct'] < 20)).sum()}")
    print(f"  Significantly overvalued (>+20%): {(df['price_gap_pct'] >= 20).sum()}")

    # By vacancy type
    print("\n" + "="*80)
    print("AVERAGE PRICE GAP BY VACANCY TYPE")
    print("="*80)
    if "vacancy_type" in df.columns:
        by_type = df.groupby("vacancy_type").agg({
            "price_gap_pct": ["mean", "count"],
            "price_avg": "mean",
        }).round(1)
        print(by_type.to_string())

    # Top undervalued (quality filtered)
    print("\n" + "="*80)
    print("TOP 20 UNDERVALUED PROPERTIES (Quality Filtered)")
    print("="*80)
    print("Criteria: >15% below predicted, low risk, >4% yield, non-declining areas")
    print()

    undervalued = finder.find_undervalued(top_n=20)
    if len(undervalued) > 0:
        display_cols = [
            "Comune_descrizione", "Regione", "Prov",
            "price_avg", "predicted_price", "price_gap_pct",
            "gross_yield", "vacancy_type"
        ]
        available = [c for c in display_cols if c in undervalued.columns]
        print(undervalued[available].to_string(index=False))
    else:
        print("No properties match strict criteria. Relaxing filters...")
        undervalued = finder.find_undervalued(
            min_undervaluation_pct=10.0,
            max_vacancy_risk=0.7,
            min_yield=3.0,
            top_n=20
        )
        display_cols = [
            "Comune_descrizione", "Regione", "Prov",
            "price_avg", "predicted_price", "price_gap_pct",
            "gross_yield", "vacancy_type"
        ]
        available = [c for c in display_cols if c in undervalued.columns]
        print(undervalued[available].to_string(index=False))

    # Smart picks
    print("\n" + "="*80)
    print("SMART PICKS: Best Risk-Adjusted Opportunities")
    print("="*80)
    print("Criteria: >20% undervalued, tourist/low vacancy areas, >5% yield, low risk")
    print()

    smart = finder.find_smart_picks()
    if len(smart) > 0:
        display_cols = [
            "Comune_descrizione", "Regione",
            "price_avg", "price_gap_pct",
            "gross_yield", "vacancy_type", "tourism_intensity"
        ]
        available = [c for c in display_cols if c in smart.columns]
        print(smart[available].head(15).to_string(index=False))
    else:
        print("No smart picks found with strict criteria.")

    # Tourist area deals
    print("\n" + "="*80)
    print("TOURIST AREA DEALS")
    print("="*80)

    tourist = finder.find_tourist_area_deals()
    if len(tourist) > 0:
        display_cols = [
            "Comune_descrizione", "Regione",
            "price_avg", "price_gap_pct",
            "gross_yield", "tourism_intensity"
        ]
        available = [c for c in display_cols if c in tourist.columns]
        print(tourist[available].head(15).to_string(index=False))
    else:
        print("No tourist area deals found.")

    # Regional summary
    print("\n" + "="*80)
    print("REGIONAL UNDERVALUATION SUMMARY")
    print("="*80)

    if "Regione" in df.columns:
        regional = df.groupby("Regione").agg({
            "price_gap_pct": "mean",
            "price_avg": "mean",
            "vacancy_rate": "mean",
        }).round(1)
        regional = regional.sort_values("price_gap_pct")
        print("\nMost Undervalued Regions (avg price gap):")
        print(regional.head(10).to_string())

    # Save results
    finder.save_results()
    print("\n" + "="*80)
    print("Results saved to outputs/")
    print("  - undervalued_properties.csv")
    print("  - smart_picks.csv")
    print("  - tourist_area_deals.csv")
    print("  - price_predictions.csv")

    return finder


if __name__ == "__main__":
    main()
