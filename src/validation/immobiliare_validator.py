"""Validate model predictions against Immobiliare.it listing data.

Compares listing prices from Immobiliare.it with:
1. OMI government valuations (ground truth)
2. Model predictions (what we're validating)

This helps assess whether our vacancy-aware price model
produces predictions that correlate with actual market listings.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ImmobiliareValidator:
    """Validates price predictions against Immobiliare.it listings.

    Compares listing prices with OMI government data and model predictions
    to assess model accuracy and identify systematic biases.

    Example:
        >>> validator = ImmobiliareValidator(
        ...     immobiliare_df=sales_listings,
        ...     omi_df=omi_prices,
        ...     predictions_df=model_predictions
        ... )
        >>> results = validator.validate_all()
        >>> print(f"Correlation: {results['omi_correlation']['pearson_r']:.3f}")
    """

    def __init__(
        self,
        immobiliare_df: pd.DataFrame,
        omi_df: pd.DataFrame,
        predictions_df: pd.DataFrame | None = None,
    ):
        """Initialize validator.

        Args:
            immobiliare_df: Listings from ImmobiliareItFetcher
            omi_df: OMI price data (municipal features)
            predictions_df: Model predictions with price_gap columns
        """
        self.immobiliare = immobiliare_df
        self.omi = omi_df
        self.predictions = predictions_df
        self.results = {}

    def _aggregate_immobiliare(self) -> pd.DataFrame:
        """Aggregate Immobiliare.it/Idealista listings by city.

        Handles both raw listing data (needs aggregation) and
        pre-aggregated static data (single row per city).
        """
        if self.immobiliare.empty:
            return pd.DataFrame()

        df = self.immobiliare.copy()

        # Check if data is already aggregated (one row per city with price_sqm)
        is_preaggregated = (
            "price_sqm" in df.columns and
            "city" in df.columns and
            len(df) == df["city"].nunique()  # One row per city
        )

        if is_preaggregated:
            # Data is already aggregated, just rename columns
            df["immobiliare_price_sqm"] = df["price_sqm"]
            df["listing_count"] = 1  # Static data doesn't have listing count
            return df

        # Raw listing data - needs aggregation
        if "price_sqm" not in df.columns and "sqm" in df.columns:
            df["price_sqm"] = df["price"] / df["sqm"]

        # Group by city
        agg_cols = ["city", "city_name", "istat_code", "omi_code", "region"]
        available_cols = [c for c in agg_cols if c in df.columns]

        agg = df.groupby(available_cols).agg({
            "price": ["count", "median", "mean", "std"],
            "price_sqm": ["median", "mean", "std"] if "price_sqm" in df.columns else ["count"],
        }).reset_index()

        # Flatten columns
        agg.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in agg.columns
        ]

        # Rename for clarity
        rename_map = {
            "price_count": "listing_count",
            "price_median": "price_total_median",
            "price_mean": "price_total_mean",
            "price_sqm_median": "immobiliare_price_sqm",
            "price_sqm_mean": "immobiliare_price_sqm_mean",
        }
        agg = agg.rename(columns={k: v for k, v in rename_map.items() if k in agg.columns})

        return agg

    def _get_omi_city_prices(self) -> pd.DataFrame:
        """Extract OMI prices for the target cities.

        Matches cities by name since ISTAT codes have varying prefixes in OMI data.
        """
        from src.data.fetchers.immobiliare_it import CITY_CONFIGS

        omi = self.omi.copy()
        matches = []

        for city, config in CITY_CONFIGS.items():
            city_name = config["name"].upper()
            istat_code = config["istat_code"]

            # Match by city name (most reliable)
            mask = omi["Comune_descrizione"].str.upper() == city_name

            # Also try matching ISTAT code as substring (handles variable prefixes)
            if "Comune_ISTAT" in omi.columns:
                omi_code_str = omi["Comune_ISTAT"].astype(str)
                mask = mask | omi_code_str.str.endswith(istat_code)

            city_data = omi[mask].copy()
            if not city_data.empty:
                city_data["city"] = city
                city_data["city_name"] = config["name"]
                matches.append(city_data.iloc[[0]])  # Take first match
                logger.debug(f"Matched {config['name']}: price={city_data['price_avg'].iloc[0]:.0f} €/m²")
            else:
                logger.warning(f"Could not match OMI data for {config['name']}")

        if matches:
            return pd.concat(matches, ignore_index=True)

        return pd.DataFrame()

    def validate_against_omi(self) -> dict:
        """Compare Immobiliare.it prices to OMI government valuations.

        Tests hypothesis: Listing prices should correlate with
        official government valuations, typically with a premium.

        Returns:
            Dictionary with correlation metrics and by-city comparison
        """
        logger.info("Validating Immobiliare.it vs OMI prices...")

        # Aggregate Immobiliare.it by city
        immo_agg = self._aggregate_immobiliare()
        if immo_agg.empty:
            return {"error": "No Immobiliare.it data"}

        # Get OMI prices for target cities
        omi_cities = self._get_omi_city_prices()
        if omi_cities.empty:
            return {"error": "No OMI data for target cities"}

        # Merge on city
        merged = immo_agg.merge(
            omi_cities[["city", "price_avg"]].drop_duplicates(),
            on="city",
            how="inner",
        )

        if merged.empty or len(merged) < 3:
            return {"error": f"Insufficient matched data: {len(merged)} cities"}

        # Get the price columns
        immo_col = "immobiliare_price_sqm" if "immobiliare_price_sqm" in merged.columns else "price_total_median"
        omi_col = "price_avg"

        # Filter valid values
        valid = merged[[immo_col, omi_col, "city", "city_name"]].dropna()

        if len(valid) < 3:
            return {"error": f"Insufficient valid data: {len(valid)} cities"}

        # Calculate correlations
        r_pearson, p_pearson = stats.pearsonr(valid[immo_col], valid[omi_col])
        r_spearman, p_spearman = stats.spearmanr(valid[immo_col], valid[omi_col])

        # Calculate price ratios (Immobiliare / OMI)
        valid["price_ratio"] = valid[immo_col] / valid[omi_col]
        valid["price_premium_pct"] = (valid["price_ratio"] - 1) * 100

        results = {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_r": float(r_spearman),
            "spearman_p": float(p_spearman),
            "mean_price_ratio": float(valid["price_ratio"].mean()),
            "mean_premium_pct": float(valid["price_premium_pct"].mean()),
            "n_cities": len(valid),
            "correlation_interpretation": self._interpret_correlation(r_pearson),
            "by_city": valid.to_dict("records"),
        }

        self.results["omi_correlation"] = results
        logger.info(f"OMI validation: r={r_pearson:.3f}, p={p_pearson:.4f}, n={len(valid)}")

        return results

    def validate_against_model(self) -> dict:
        """Compare Immobiliare.it prices to model predictions.

        Tests whether our model predictions correlate with actual
        market listing prices.

        Returns:
            Dictionary with correlation metrics
        """
        if self.predictions is None or self.predictions.empty:
            return {"error": "No predictions data provided"}

        logger.info("Validating Immobiliare.it vs model predictions...")

        # Aggregate Immobiliare.it by city
        immo_agg = self._aggregate_immobiliare()
        if immo_agg.empty:
            return {"error": "No Immobiliare.it data"}

        # Get predictions for target cities
        from src.data.fetchers.immobiliare_it import CITY_CONFIGS

        predictions = self.predictions.copy()

        # Match predictions to cities
        merged_data = []
        for city, config in CITY_CONFIGS.items():
            # Find city in predictions
            city_pred = predictions[
                (predictions["Comune_ISTAT"].astype(str) == config["omi_code"]) |
                (predictions["Comune_ISTAT"].astype(str) == config["istat_code"]) |
                (predictions.get("Comune_descrizione", pd.Series()).str.upper() == config["name"].upper())
            ]

            if not city_pred.empty:
                row = city_pred.iloc[0]
                merged_data.append({
                    "city": city,
                    "city_name": config["name"],
                    "predicted_price": row.get("predicted_price", row.get("price_avg")),
                    "price_gap_pct": row.get("price_gap_pct", 0),
                    "omi_price": row.get("price_avg"),
                })

        if not merged_data:
            return {"error": "Could not match predictions to cities"}

        pred_df = pd.DataFrame(merged_data)

        # Merge with Immobiliare.it
        merged = immo_agg.merge(pred_df, on="city", how="inner", suffixes=("_immo", "_pred"))

        if len(merged) < 3:
            return {"error": f"Insufficient matched data: {len(merged)} cities"}

        immo_col = "immobiliare_price_sqm" if "immobiliare_price_sqm" in merged.columns else "price_total_median"

        # Get city_name from either source
        city_name_col = "city_name" if "city_name" in merged.columns else "city_name_immo"
        if city_name_col not in merged.columns:
            city_name_col = "city_name_pred"

        valid_cols = [immo_col, "predicted_price", "price_gap_pct", "city"]
        if city_name_col in merged.columns:
            valid_cols.append(city_name_col)
            merged["city_name"] = merged[city_name_col]
            valid_cols[-1] = "city_name"

        valid = merged[valid_cols].dropna()

        if len(valid) < 3:
            return {"error": f"Insufficient valid data: {len(valid)} cities"}

        # Calculate correlations
        r_pearson, p_pearson = stats.pearsonr(valid[immo_col], valid["predicted_price"])
        r_spearman, p_spearman = stats.spearmanr(valid[immo_col], valid["predicted_price"])

        # Compare model "undervalued" calls with actual listing prices
        valid["immo_vs_pred_pct"] = (valid[immo_col] - valid["predicted_price"]) / valid["predicted_price"] * 100

        # Do undervalued picks (negative price_gap) also show lower listings?
        undervalued = valid[valid["price_gap_pct"] < -10]
        undervalued_confirmation = None
        if len(undervalued) > 0:
            # If model says undervalued, listings should also be below prediction
            undervalued_confirmation = (undervalued["immo_vs_pred_pct"] < 0).mean()

        results = {
            "pearson_r": float(r_pearson),
            "pearson_p": float(p_pearson),
            "spearman_r": float(r_spearman),
            "spearman_p": float(p_spearman),
            "n_cities": len(valid),
            "correlation_interpretation": self._interpret_correlation(r_pearson),
            "undervalued_confirmation_rate": float(undervalued_confirmation) if undervalued_confirmation else None,
            "by_city": valid.to_dict("records"),
        }

        self.results["model_correlation"] = results
        logger.info(f"Model validation: r={r_pearson:.3f}, p={p_pearson:.4f}, n={len(valid)}")

        return results

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient strength."""
        r_abs = abs(r)
        if r_abs >= 0.9:
            strength = "very strong"
        elif r_abs >= 0.7:
            strength = "strong"
        elif r_abs >= 0.5:
            strength = "moderate"
        elif r_abs >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"

        direction = "positive" if r >= 0 else "negative"
        return f"{strength} {direction}"

    def validate_all(self) -> dict:
        """Run all validations.

        Returns:
            Dictionary with all validation results
        """
        results = {
            "omi_correlation": self.validate_against_omi(),
            "model_correlation": self.validate_against_model() if self.predictions is not None else None,
        }

        self.results = results
        return results

    def generate_validation_geojson(self, output_path: Path) -> dict:
        """Generate GeoJSON with validation markers for the Leaflet map.

        Creates point markers for each validated city showing the comparison
        between Immobiliare.it, OMI, and model predictions.

        Args:
            output_path: Path to save the GeoJSON file

        Returns:
            GeoJSON feature collection dict
        """
        from src.data.fetchers.immobiliare_it import CITY_CONFIGS

        features = []

        # City coordinates (approximate center points)
        city_coords = {
            "roma": [12.4964, 41.9028],
            "milano": [9.1900, 45.4642],
            "firenze": [11.2558, 43.7696],
            "napoli": [14.2681, 40.8518],
            "bologna": [11.3426, 44.4949],
            "torino": [7.6869, 45.0703],
            "venezia": [12.3155, 45.4408],
            "palermo": [13.3615, 38.1157],
            "genova": [8.9463, 44.4056],
            "bari": [16.8719, 41.1171],
        }

        # Get aggregated data
        immo_agg = self._aggregate_immobiliare()

        for city, config in CITY_CONFIGS.items():
            coords = city_coords.get(city)
            if not coords:
                continue

            properties = {
                "city": config["name"],
                "city_key": city,
                "region": config["region"],
                "province": config["province"],
                "istat_code": config["istat_code"],
            }

            # Add Immobiliare.it data
            city_immo = immo_agg[immo_agg["city"] == city]
            if not city_immo.empty:
                row = city_immo.iloc[0]
                properties["immobiliare_price_sqm"] = float(row.get("immobiliare_price_sqm", 0))
                properties["immobiliare_listing_count"] = int(row.get("listing_count", 0))

            # Add OMI data
            if "omi_correlation" in self.results and "by_city" in self.results["omi_correlation"]:
                for city_data in self.results["omi_correlation"]["by_city"]:
                    if city_data.get("city") == city:
                        properties["omi_price_sqm"] = float(city_data.get("price_avg", 0))
                        properties["price_ratio"] = float(city_data.get("price_ratio", 1))
                        properties["price_premium_pct"] = float(city_data.get("price_premium_pct", 0))
                        break

            # Add model prediction data
            if "model_correlation" in self.results and self.results["model_correlation"]:
                if "by_city" in self.results["model_correlation"]:
                    for city_data in self.results["model_correlation"]["by_city"]:
                        if city_data.get("city") == city:
                            properties["predicted_price"] = float(city_data.get("predicted_price", 0))
                            properties["price_gap_pct"] = float(city_data.get("price_gap_pct", 0))
                            properties["immo_vs_pred_pct"] = float(city_data.get("immo_vs_pred_pct", 0))
                            break

            # Determine validation status
            if properties.get("price_ratio"):
                ratio = properties["price_ratio"]
                if 0.85 <= ratio <= 1.25:
                    properties["validation_status"] = "confirmed"
                elif ratio < 0.85:
                    properties["validation_status"] = "listings_lower"
                else:
                    properties["validation_status"] = "listings_higher"
            else:
                properties["validation_status"] = "no_data"

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coords,
                },
                "properties": properties,
            }

            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "validation_summary": {
                    "omi_pearson_r": self.results.get("omi_correlation", {}).get("pearson_r"),
                    "model_pearson_r": self.results.get("model_correlation", {}).get("pearson_r") if self.results.get("model_correlation") else None,
                    "n_cities": len(features),
                }
            }
        }

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"Saved validation GeoJSON to {output_path}")

        return geojson

    def save_report(self, output_dir: Path = Path("outputs/validation")):
        """Save validation results to files.

        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary CSV
        if self.results:
            summary_data = []
            for name, result in self.results.items():
                if result and isinstance(result, dict) and "error" not in result:
                    summary_data.append({
                        "validation": name,
                        "pearson_r": result.get("pearson_r"),
                        "pearson_p": result.get("pearson_p"),
                        "spearman_r": result.get("spearman_r"),
                        "n_cities": result.get("n_cities"),
                        "interpretation": result.get("correlation_interpretation"),
                    })

            if summary_data:
                pd.DataFrame(summary_data).to_csv(
                    output_dir / "validation_summary.csv", index=False
                )

            # Save detailed results as JSON
            with open(output_dir / "validation_results.json", "w") as f:
                # Convert numpy types for JSON serialization
                def convert(obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj

                json.dump(self.results, f, indent=2, default=convert)

        # Save GeoJSON for map
        self.generate_validation_geojson(output_dir / "validation_cities.geojson")

        logger.info(f"Saved validation report to {output_dir}")


def compare_with_omi(
    immobiliare_df: pd.DataFrame,
    omi_df: pd.DataFrame,
) -> dict:
    """Quick comparison of Immobiliare.it with OMI data.

    Args:
        immobiliare_df: Listings from ImmobiliareItFetcher
        omi_df: OMI municipal data

    Returns:
        Validation results dictionary
    """
    validator = ImmobiliareValidator(immobiliare_df, omi_df)
    return validator.validate_against_omi()


def compare_with_model(
    immobiliare_df: pd.DataFrame,
    omi_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> dict:
    """Quick comparison of Immobiliare.it with model predictions.

    Args:
        immobiliare_df: Listings from ImmobiliareItFetcher
        omi_df: OMI municipal data
        predictions_df: Model predictions

    Returns:
        Validation results dictionary
    """
    validator = ImmobiliareValidator(immobiliare_df, omi_df, predictions_df)
    return validator.validate_all()
