"""Validate price prediction model using listing data from Idealista/Immobiliare.it.

This script:
1. Loads listing price data for 10 major Italian cities (static Idealista data)
2. Compares listing prices with OMI government valuations
3. Compares listing prices with model predictions
4. Generates validation report and GeoJSON for map visualization

Note: Uses static Idealista Q4 2025 price data since Immobiliare.it blocks programmatic access.
Data source: https://www.idealista.it/news/immobiliare/residenziale/2025/12/29/305127
"""

import logging
from pathlib import Path

import pandas as pd

from src.validation import ImmobiliareValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_listing_data() -> pd.DataFrame:
    """Load listing price data from static Idealista file or cached scrape data.

    Returns:
        DataFrame with city-level price data
    """
    # Priority 1: Static Idealista data (most reliable)
    idealista_file = Path("data/external/idealista_prices_2025.csv")
    if idealista_file.exists():
        logger.info("Loading static Idealista price data")
        df = pd.read_csv(idealista_file)
        df["listing_type"] = "sale"
        df["source"] = "idealista"
        return df

    # Priority 2: Cached Immobiliare.it data (if available from manual download)
    cache_file = Path("data/raw/immobiliare/all_cities_sales.csv")
    if cache_file.exists():
        logger.info("Loading cached Immobiliare.it data")
        return pd.read_csv(cache_file)

    return pd.DataFrame()


def main():
    """Run full validation pipeline."""
    print("\n" + "="*80)
    print("LISTING DATA VALIDATION")
    print("Comparing listing prices with OMI data and model predictions")
    print("="*80)

    # Load listing data
    sales_df = load_listing_data()

    if sales_df.empty:
        print("\nERROR: No listing data available.")
        print("Please ensure data/external/idealista_prices_2025.csv exists.")
        return

    # Show data source
    source = sales_df["source"].iloc[0] if "source" in sales_df.columns else "unknown"
    print(f"\nData source: {source}")
    print(f"  Loaded price data for {len(sales_df)} cities")

    # Show prices by city
    print("\n" + "-"*60)
    print("LISTING PRICES BY CITY")
    print("-"*60)
    for _, row in sales_df.sort_values("price_sqm", ascending=False).iterrows():
        print(f"  {row['city_name']}: {row['price_sqm']:,.0f} €/m²")

    # Load OMI data
    print("\n" + "-"*60)
    print("LOADING OMI DATA")
    print("-"*60)

    omi_path = Path("data/processed/municipal_features.csv")
    if not omi_path.exists():
        print(f"ERROR: OMI data not found at {omi_path}")
        return

    omi_df = pd.read_csv(omi_path)
    print(f"  Loaded OMI data: {len(omi_df)} municipalities")

    # Load model predictions
    predictions_path = Path("outputs/price_predictions.csv")
    predictions_df = None
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
        print(f"  Loaded model predictions: {len(predictions_df)} municipalities")
    else:
        print("  WARNING: No model predictions found, skipping model validation")

    # Run validation
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    validator = ImmobiliareValidator(
        immobiliare_df=sales_df,
        omi_df=omi_df,
        predictions_df=predictions_df,
    )

    results = validator.validate_all()

    # Print OMI correlation results
    print("\n" + "-"*60)
    print("1. IMMOBILIARE.IT vs OMI GOVERNMENT PRICES")
    print("-"*60)

    omi_results = results.get("omi_correlation", {})
    if "error" in omi_results:
        print(f"  Error: {omi_results['error']}")
    else:
        print(f"  Pearson correlation:  r = {omi_results['pearson_r']:.3f} (p = {omi_results['pearson_p']:.4f})")
        print(f"  Spearman correlation: ρ = {omi_results['spearman_r']:.3f} (p = {omi_results['spearman_p']:.4f})")
        print(f"  Interpretation: {omi_results['correlation_interpretation']}")
        print(f"  Cities compared: {omi_results['n_cities']}")
        print(f"\n  Average price ratio (Immobiliare/OMI): {omi_results['mean_price_ratio']:.2f}")
        print(f"  Average listing premium: {omi_results['mean_premium_pct']:.1f}%")

        print("\n  By City:")
        print("  " + "-"*56)
        print(f"  {'City':<15} {'Immo €/m²':>12} {'OMI €/m²':>12} {'Ratio':>8} {'Premium':>10}")
        print("  " + "-"*56)

        for city_data in sorted(omi_results.get("by_city", []), key=lambda x: x.get("city_name", "")):
            city = city_data.get("city_name", city_data.get("city", ""))[:15]
            immo = city_data.get("immobiliare_price_sqm", city_data.get("price_total_median", 0))
            omi = city_data.get("price_avg", 0)
            ratio = city_data.get("price_ratio", 0)
            premium = city_data.get("price_premium_pct", 0)
            print(f"  {city:<15} {immo:>12,.0f} {omi:>12,.0f} {ratio:>8.2f} {premium:>+9.1f}%")

    # Print model correlation results
    print("\n" + "-"*60)
    print("2. IMMOBILIARE.IT vs MODEL PREDICTIONS")
    print("-"*60)

    model_results = results.get("model_correlation")
    if model_results is None:
        print("  Skipped: No model predictions available")
    elif "error" in model_results:
        print(f"  Error: {model_results['error']}")
    else:
        print(f"  Pearson correlation:  r = {model_results['pearson_r']:.3f} (p = {model_results['pearson_p']:.4f})")
        print(f"  Spearman correlation: ρ = {model_results['spearman_r']:.3f} (p = {model_results['spearman_p']:.4f})")
        print(f"  Interpretation: {model_results['correlation_interpretation']}")
        print(f"  Cities compared: {model_results['n_cities']}")

        if model_results.get("undervalued_confirmation_rate") is not None:
            print(f"\n  Undervalued confirmation: {model_results['undervalued_confirmation_rate']*100:.1f}%")
            print("  (% of model-identified undervalued cities also showing lower listings)")

    # Save results
    print("\n" + "-"*60)
    print("SAVING RESULTS")
    print("-"*60)

    output_dir = Path("outputs/validation")
    validator.save_report(output_dir)

    print("  Saved validation_summary.csv")
    print("  Saved validation_results.json")
    print("  Saved validation_cities.geojson")

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    if omi_results and "error" not in omi_results:
        r = omi_results["pearson_r"]
        if r >= 0.7:
            verdict = "STRONG correlation - model aligns well with market listings"
        elif r >= 0.5:
            verdict = "MODERATE correlation - model captures general trends"
        elif r >= 0.3:
            verdict = "WEAK correlation - model may need improvement"
        else:
            verdict = "VERY WEAK correlation - investigate data quality"

        print(f"\n  {verdict}")
        print(f"  Listing prices are typically {omi_results['mean_premium_pct']:.0f}% above OMI valuations")
        print("  (This premium is expected - listings are asking prices, not transactions)")

    print("\n  Files saved to outputs/validation/")
    print("  Add validation_cities.geojson to the Leaflet map for visualization")
    print()

    return results


if __name__ == "__main__":
    main()
