# Italian Real Estate Demographic Risk Model

A machine learning system for analyzing Italian real estate prices, identifying undervalued municipalities, and computing investment metrics based on demographic, economic, and tourism data.

## Key Results

| Model | Target | R² Score | Key Features |
|-------|--------|----------|--------------|
| Price (with STR) | EUR/sqm | **83.2%** | STR density, income, location |
| Rent (with STR) | EUR/sqm/month | **74.3%** | STR density, tourism, population |
| Price (baseline) | EUR/sqm | 80.9% | Demographics only |

**Top Finding**: Short-term rental (STR) density is the #1 predictor of real estate prices, explaining 21-31% of variance in Italian municipalities with tourism activity.

## Features

- **Hedonic Pricing Model**: Log-log specification capturing price elasticities
- **Multi-source Data Integration**: OMI prices, ISTAT demographics, IRPEF income, InsideAirbnb
- **Gradient Boosting Regression**: Optimized ensemble model with feature importance
- **Undervaluation Detection**: Residual-based identification of mispriced municipalities
- **Smart Investment Picks**: Combined undervaluation + yield scoring

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full pipeline
python scripts/run_pipeline.py train --config configs/default.yaml

# Find smart investment picks
python scripts/run_pipeline.py smart-picks --top 50 --min-yield 4.0
```

## Project Structure

```
italian-real-estate-risk/
├── configs/
│   └── default.yaml          # Main configuration
├── src/
│   ├── config.py             # Pydantic configuration
│   ├── data/
│   │   ├── schemas.py        # Pandera validation schemas
│   │   ├── loaders.py        # Data loading utilities
│   │   └── processors/       # Source-specific processors
│   │       ├── omi.py        # OMI price data
│   │       ├── istat.py      # Demographics
│   │       ├── irpef.py      # Income
│   │       ├── tourism.py    # Tourism statistics
│   │       └── airbnb.py     # Short-term rentals
│   ├── features/
│   │   ├── base.py           # Transformer base classes
│   │   ├── demographic.py    # Population features
│   │   ├── economic.py       # Income features
│   │   ├── geographic.py     # Distance features
│   │   ├── tourism.py        # STR features
│   │   └── pipeline.py       # Feature orchestration
│   ├── models/
│   │   ├── base.py           # Model interface
│   │   ├── regression.py     # OLS, Ridge, Lasso
│   │   ├── ensemble.py       # Gradient Boosting
│   │   └── training.py       # Training orchestration
│   ├── evaluation/
│   │   ├── metrics.py        # R², RMSE, diagnostics
│   │   └── valuation.py      # Undervaluation detection
│   └── utils/
│       ├── constants.py      # ISTAT codes, mappings
│       └── logging.py        # Loguru configuration
├── scripts/
│   └── run_pipeline.py       # CLI entry point
├── tests/
│   ├── conftest.py           # Pytest fixtures
│   └── unit/                 # Unit tests
├── data/
│   ├── raw/                  # Source data files
│   └── processed/            # Computed features
├── outputs/                  # Maps, charts, dashboards
└── docs/
    ├── METHODOLOGY.md        # Statistical methodology
    └── data_dictionary.md    # Column definitions
```

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| **OMI** | Real estate quotations (Agenzia delle Entrate) | 7,904 municipalities, 2015-2024 |
| **ISTAT** | Demographics, population trends | All municipalities |
| **IRPEF** | Income tax declarations (MEF) | Municipality-level averages |
| **InsideAirbnb** | Short-term rental listings | Milan, Florence, Bologna, Naples |
| **Tourism** | Tourist arrivals by province | Province-level |

## Model Features

### Demographic Features
- `log_population` - Log-transformed population
- `pop_change_pct` - Population change 2011-2023
- `pop_declining` - Binary flag for declining areas

### Economic Features
- `log_income` - Log-transformed average income
- `income_change_pct` - Income growth rate
- `income_ratio` - Ratio to national average

### Geographic Features
- `lat`, `lon` - Coordinates
- `dist_major_city` - Distance to nearest major city (km)
- `dist_coast` - Distance to coast (km)
- `northern`, `coastal`, `urban` - Location flags

### Tourism Features
- `tourism_intensity` - Arrivals per 1000 residents
- `str_density` - Airbnb listings per 1000 residents
- `str_premium` - Airbnb monthly revenue vs long-term rent

## Usage Examples

### Training a Model

```python
from src.config import load_config
from src.features.pipeline import create_features
from src.models import GradientBoostingModel, ModelTrainer

# Load configuration
config = load_config("configs/default.yaml")

# Create features from raw data
features = create_features(Path("data"), config)

# Train model
trainer = ModelTrainer(config.model)
result = trainer.train(
    features,
    target="log_price_mid",
    model_type="gradient_boosting",
)

print(f"Test R²: {result.test_result.r_squared:.4f}")
```

### Finding Undervalued Properties

```python
from src.evaluation import ValuationAnalyzer

analyzer = ValuationAnalyzer(config.evaluation)
valuations = analyzer.compute_valuations(features, predictions)

# Get smart picks: undervalued + high yield
smart_picks = analyzer.get_smart_picks(
    valuations,
    min_yield_pct=4.0,
    max_price_gap_pct=-15.0,
    top_n=50,
)
```

### CLI Commands

```bash
# Train models
python scripts/run_pipeline.py train --config configs/default.yaml

# Evaluate saved model
python scripts/run_pipeline.py evaluate models/price_model.pkl

# Find investment opportunities
python scripts/run_pipeline.py smart-picks --top 50 --min-yield 4.0

# Show data info
python scripts/run_pipeline.py info
```

## Configuration

All parameters are configured in `configs/default.yaml`:

```yaml
# Model hyperparameters
model:
  test_size: 0.2
  random_state: 42
  gb_params:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 6

# Evaluation thresholds
evaluation:
  undervaluation_threshold: -0.15  # 15% below fair value
  min_yield_pct: 4.0               # Minimum gross yield
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/ -v
```

## Key Findings

1. **STR Density is #1 Predictor**: Short-term rental density explains 21-31% of price variance, more than any traditional demographic factor.

2. **Tourism Premium on Prices > Rents**: Tourism intensity coefficient is 3x higher for prices (β=0.089) than rents (β=0.029), suggesting capital appreciation expectations.

3. **Airbnb Premium**: STR revenue is 470-1,117% higher than long-term rents in major cities.

4. **Regional Disparities**: Northern Italy prices are 40-60% higher than South, controlling for income and demographics.

## Requirements

- Python 3.12+
- Key dependencies: pandas, scikit-learn, geopandas, pydantic, pandera, loguru, typer

See `pyproject.toml` for full dependency list.

## License

This project uses data from:
- **OMI**: CC-BY (cite: Agenzia delle Entrate)
- **ISTAT**: CC-BY 3.0
- **InsideAirbnb**: For research purposes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/`
4. Run linting: `ruff check src/`
5. Submit a pull request

## Authors

Italian Real Estate Analysis Team
