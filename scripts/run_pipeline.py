#!/usr/bin/env python3
"""Main CLI entry point for the Italian Real Estate Risk Model.

This script provides a command-line interface for running the full
analysis pipeline, from data loading through model training to
valuation analysis.

Usage:
    python scripts/run_pipeline.py train --config configs/default.yaml
    python scripts/run_pipeline.py evaluate --model models/price_gb.pkl
    python scripts/run_pipeline.py smart-picks --top 50
"""

from pathlib import Path
from typing import Optional

import typer
from loguru import logger

app = typer.Typer(
    name="italian-real-estate",
    help="Italian Real Estate Demographic Risk Model",
    add_completion=False,
)


@app.command()
def train(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir", "-d",
        help="Root data directory",
    ),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir", "-o",
        help="Output directory for results",
    ),
    model_type: str = typer.Option(
        "gradient_boosting",
        "--model", "-m",
        help="Model type: gradient_boosting, ols, ridge",
    ),
    include_str: bool = typer.Option(
        True,
        "--str/--no-str",
        help="Include short-term rental features",
    ),
    year: Optional[int] = typer.Option(
        None,
        "--year", "-y",
        help="Specific year to analyze",
    ),
) -> None:
    """Train price and rent prediction models.

    Runs the full training pipeline:
    1. Load and process data from all sources
    2. Engineer features
    3. Train models
    4. Evaluate and save results
    """
    from src.config import load_config
    from src.features.pipeline import create_features
    from src.models.training import PriceRentTrainer
    from src.utils.logging import setup_logging

    setup_logging(level="INFO", log_file=output_dir / "training.log")
    logger.info("Starting training pipeline")

    # Load configuration
    config = load_config(config_path)

    # Create features
    logger.info("Creating features...")
    features = create_features(data_dir, config, year=year)
    logger.info(f"Created {len(features):,} feature records")

    # Train models
    trainer = PriceRentTrainer(config)

    logger.info("Training price model...")
    price_result = trainer.train_price_model(features, include_str=include_str)
    logger.info(f"Price model: R²={price_result.test_result.r_squared:.4f}")

    logger.info("Training rent model...")
    rent_result = trainer.train_rent_model(features, include_str=include_str)
    if rent_result:
        logger.info(f"Rent model: R²={rent_result.test_result.r_squared:.4f}")

    # Save models
    output_dir.mkdir(parents=True, exist_ok=True)
    price_result.model.save(output_dir / "price_model.pkl")
    if rent_result:
        rent_result.model.save(output_dir / "rent_model.pkl")

    # Print summary
    typer.echo("\n" + "=" * 50)
    typer.echo("Training Complete")
    typer.echo("=" * 50)
    typer.echo(price_result.summary())
    if rent_result:
        typer.echo("\n" + rent_result.summary())


@app.command()
def evaluate(
    model_path: Path = typer.Argument(
        ...,
        help="Path to trained model file",
    ),
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir", "-d",
        help="Root data directory",
    ),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir", "-o",
        help="Output directory",
    ),
) -> None:
    """Evaluate a trained model on data."""
    from src.models.base import BaseModel
    from src.utils.logging import setup_logging

    setup_logging(level="INFO")

    # Load model
    model = BaseModel.load(model_path)
    logger.info(f"Loaded model: {model.name}")

    # TODO: Implement evaluation
    typer.echo(f"Model: {model.name}")
    typer.echo(f"Features: {len(model.feature_names)}")


@app.command("smart-picks")
def smart_picks(
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir", "-d",
        help="Root data directory",
    ),
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir", "-o",
        help="Output directory",
    ),
    top_n: int = typer.Option(
        50,
        "--top", "-n",
        help="Number of picks to show",
    ),
    min_yield: float = typer.Option(
        4.0,
        "--min-yield",
        help="Minimum gross yield percentage",
    ),
    max_gap: float = typer.Option(
        -15.0,
        "--max-gap",
        help="Maximum price gap percentage (negative = undervalued)",
    ),
) -> None:
    """Identify smart investment opportunities.

    Smart picks are municipalities that are:
    - Undervalued according to the model
    - Have attractive rental yields
    """
    import pandas as pd

    from src.utils.logging import setup_logging

    setup_logging(level="INFO")

    # Load pre-computed valuations if available
    valuations_path = data_dir / "processed" / "valuations_with_str_model.csv"
    if not valuations_path.exists():
        typer.echo(f"Error: Valuations file not found at {valuations_path}")
        typer.echo("Run 'train' command first to generate valuations.")
        raise typer.Exit(1)

    valuations = pd.read_csv(valuations_path)

    # Filter for smart picks
    picks = valuations[
        (valuations["price_gap_with_str"] <= max_gap)
        & (valuations["gross_yield_pct"] >= min_yield)
    ].sort_values("price_gap_with_str").head(top_n)

    # Display results
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Top {len(picks)} Smart Picks")
    typer.echo(f"Criteria: Gap <= {max_gap}%, Yield >= {min_yield}%")
    typer.echo(f"{'=' * 60}\n")

    display_cols = ["nome", "regione", "price_gap_with_str", "gross_yield_pct", "prezzo_medio"]
    if all(c in picks.columns for c in display_cols):
        for _, row in picks.iterrows():
            typer.echo(
                f"{row['nome']:<30} | "
                f"Gap: {row['price_gap_with_str']:>6.1f}% | "
                f"Yield: {row['gross_yield_pct']:>5.1f}% | "
                f"Price: {row['prezzo_medio']:>7.0f} EUR/sqm"
            )

    # Save to CSV
    output_path = output_dir / "smart_picks.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    picks.to_csv(output_path, index=False)
    typer.echo(f"\nSaved to {output_path}")


@app.command()
def info(
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir", "-d",
        help="Root data directory",
    ),
) -> None:
    """Show information about available data."""
    import os

    typer.echo("\nData Directory Structure:")
    typer.echo("=" * 40)

    for subdir in ["raw", "processed"]:
        path = data_dir / subdir
        if path.exists():
            files = list(path.rglob("*"))
            csv_files = [f for f in files if f.suffix == ".csv"]
            typer.echo(f"\n{subdir}/")
            for f in sorted(csv_files)[:10]:
                size = os.path.getsize(f) / 1_000_000
                typer.echo(f"  {f.relative_to(path)}: {size:.1f} MB")
            if len(csv_files) > 10:
                typer.echo(f"  ... and {len(csv_files) - 10} more files")


@app.command()
def validate(
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir", "-d",
        help="Root data directory",
    ),
) -> None:
    """Validate data files against schemas."""
    from src.data.schemas import validate_dataframe, OMIValoriSchema
    from src.utils.logging import setup_logging
    import pandas as pd

    setup_logging(level="INFO")
    typer.echo("Validating data files...")

    # Check OMI data
    omi_path = data_dir / "raw" / "omi" / "valori.csv"
    if omi_path.exists():
        typer.echo(f"\nValidating {omi_path}...")
        try:
            df = pd.read_csv(omi_path, sep=";", nrows=1000)
            # Note: Full validation would use the schema
            typer.echo(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            typer.echo(f"  ✗ Error: {e}")
    else:
        typer.echo(f"  ✗ Not found: {omi_path}")


if __name__ == "__main__":
    app()
