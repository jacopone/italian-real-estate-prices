# Italian Real Estate Demographic Risk Model

A machine learning system for analyzing Italian real estate prices, identifying undervalued municipalities, and computing investment metrics based on demographic, economic, and tourism data.

## Key Results

| Model | Target | R² Score | Validation | Notes |
|-------|--------|----------|------------|-------|
| Price (GB + Vacancy) | EUR/sqm | **48.2%** | Random split | Vacancy-aware model |
| Price (GB + STR, Optuna) | EUR/sqm | **84.8%** | Spatial CV | Holdout municipalities |
| Price (GB + STR) | EUR/sqm | **92.2%** | Temporal | Train 2014-21, test 22-23 |
| Price (GB + lag) | EUR/sqm | **99.4%** | Spatial CV | For forecasting |
| Rent (GB + STR) | EUR/sqm/month | **74.3%** | Random split | |

### Model Validation (Idealista Q4 2025)

| Comparison | Pearson r | p-value | Interpretation |
|------------|-----------|---------|----------------|
| Idealista vs OMI | **0.883** | 0.0016 | Very strong positive |
| Idealista vs Model | **0.756** | 0.0185 | Strong positive |

- Average listing premium: **+15.5%** above OMI (expected for asking vs transaction prices)
- Undervalued confirmation rate: **100%** (model's undervalued picks confirmed by market)

**Key Findings**:
1. STR density is the #1 predictor (65% feature importance) for price levels
2. Vacancy classification identifies 4 distinct market types across 7,850 municipalities
3. Model predictions correlate strongly (r=0.756) with actual Idealista listing prices
4. Model captures 91% of spatial autocorrelation (Moran's I: 0.76 → 0.07)

## Features

- **Hedonic Pricing Model**: Log-log specification capturing price elasticities
- **Multi-source Data Integration**: OMI prices, ISTAT demographics, IRPEF income, InsideAirbnb
- **Vacancy Classification**: 4-type classification (low/decline/tourist/mixed vacancy)
- **Gradient Boosting Regression**: Optimized ensemble model with feature importance
- **Undervaluation Detection**: Residual-based identification of mispriced municipalities
- **Market Validation**: Correlation analysis with Idealista listing prices
- **Interactive Map**: Leaflet visualization with vacancy and validation layers
- **Smart Investment Picks**: Combined undervaluation + yield scoring

## Quick Start

```bash
# Install dependencies (using devenv or uv)
devenv shell
# or: uv sync

# Run the full pipeline with vacancy classification
python -m src.pipeline

# Train vacancy-aware price model
python src/train_with_vacancy.py

# Find undervalued municipalities
python src/find_undervalued.py

# Validate model against Idealista listings
python src/validate_with_immobiliare.py

# View interactive map
python -m http.server 8080
# Open http://localhost:8080/vacancy_map.html
```

## Project Structure

```
italian-real-estate-prices/
├── configs/
│   ├── default.yaml          # Main configuration
│   └── data_sources.yaml     # Data source URLs and settings
├── src/
│   ├── config.py             # Pydantic configuration
│   ├── pipeline.py           # Main data pipeline
│   ├── train_with_vacancy.py # Vacancy-aware model training
│   ├── find_undervalued.py   # Undervalued municipality detection
│   ├── validate_with_immobiliare.py  # Market validation script
│   ├── data/
│   │   ├── fetchers/         # Data fetchers
│   │   │   ├── omi.py        # OMI price data
│   │   │   ├── istat.py      # ISTAT demographics
│   │   │   ├── inside_airbnb.py  # Airbnb data
│   │   │   └── immobiliare_it.py # Immobiliare.it/Idealista
│   │   ├── schemas.py        # Pandera validation schemas
│   │   └── loaders.py        # Data loading utilities
│   ├── features/
│   │   ├── vacancy_classifier.py  # 4-type vacancy classification
│   │   ├── vacancy_features.py    # Vacancy feature engineering
│   │   ├── demographic.py    # Population features
│   │   ├── economic.py       # Income features
│   │   └── tourism.py        # STR features
│   ├── validation/           # Model validation
│   │   └── immobiliare_validator.py  # Listing price correlation
│   ├── models/
│   │   ├── regression.py     # OLS, Ridge, Lasso
│   │   └── ensemble.py       # Gradient Boosting
│   └── utils/
│       └── constants.py      # ISTAT codes, mappings
├── data/
│   ├── raw/                  # Source data files
│   ├── processed/            # Computed features, GeoJSON
│   └── external/             # External validation data
├── outputs/
│   ├── validation/           # Validation results
│   └── price_predictions.csv # Model predictions
├── vacancy_map.html          # Interactive Leaflet map
└── docs/
    └── METHODOLOGY.md        # Statistical methodology
```

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| **OMI** | Real estate quotations (Agenzia delle Entrate) | 7,850 municipalities, 2014-2023 |
| **ISTAT** | Demographics, population, vacancy rates | All municipalities, Census 2021 |
| **IRPEF** | Income tax declarations (MEF) | Municipality-level, 2012-2023 |
| **InsideAirbnb** | Short-term rental listings | Milan, Florence, Bologna, Naples |
| **Idealista** | Listing prices for validation | 10 major cities, Q4 2025 |

## Vacancy Classification

The model classifies 7,850 Italian municipalities into 4 vacancy types based on ISTAT Census 2021 data:

| Type | Count | Avg Price | Characteristics |
|------|-------|-----------|-----------------|
| **Low Vacancy** | 5,595 | €987/m² | Normal market, <15% vacancy |
| **Tourist Vacancy** | 931 | €1,306/m² | High tourism, seasonal homes |
| **Decline Vacancy** | 1,195 | €859/m² | Depopulating, >20% vacancy |
| **Mixed Vacancy** | 129 | €1,651/m² | Complex markets (tourism + decline) |

Classification criteria:
- `vacancy_rate`: Total vacancy rate from ISTAT Census
- `pop_change_10yr`: Population change over 10 years
- `tourism_index`: Tourism intensity (arrivals per resident)

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

### Vacancy Features
- `vacancy_rate` - Total housing vacancy rate (%)
- `vacancy_type` - Classification (low/decline/tourist/mixed)
- `pop_change_10yr` - Population change over 10 years (%)

## Interactive Map

The project includes an interactive Leaflet map (`vacancy_map.html`) with three views:

1. **Regions View**: Aggregated vacancy classification by region
2. **Municipalities View**: Individual municipality polygons (7,850 total)
3. **Validation View**: 10 major cities with price comparison markers

**Validation markers show**:
- Idealista listing price (€/m²)
- OMI government valuation (€/m²)
- Listing premium vs OMI (%)
- Model predicted price (€/m²)

**Color coding**:
- 🟢 Green: Confirmed (listing ≈ OMI price)
- 🟠 Orange: Listings higher (premium market)
- 🔴 Red: Listings lower (declining market)

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

1. **STR Density is #1 Predictor**: Short-term rental density accounts for ~65% of feature importance in the Gradient Boosting model, far exceeding traditional demographic factors.

2. **Spatial Structure Captured**: Model reduces spatial autocorrelation by 91% (Moran's I: 0.76 → 0.07), indicating location features effectively capture geographic patterns.

3. **Prices are Persistent**: Year-over-year price correlation is ~0.99. Adding lagged price increases R² from 84% to 99%, useful for forecasting.

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
