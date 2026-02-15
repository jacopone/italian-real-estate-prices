# Italian Real Estate Demographic Risk Model

ML system for Italian real estate price analysis using demographic, economic, and tourism data across 7,850 municipalities.

## What This Does

Predicts residential real estate prices per square meter for every municipality in Italy by combining government transaction data (OMI), census demographics (ISTAT), tax income records (IRPEF), and short-term rental activity (InsideAirbnb). The system identifies undervalued municipalities and scores investment opportunities by comparing predicted fair value against observed prices.

## Results

| Model | Target | R² Score | Validation | Notes |
|-------|--------|----------|------------|-------|
| Price (GB + Vacancy) | EUR/sqm | **48.2%** | Random split | Vacancy-aware model |
| Price (GB + STR, Optuna) | EUR/sqm | **84.8%** | Spatial CV | Holdout municipalities |
| Price (GB + STR) | EUR/sqm | **92.2%** | Temporal | Train 2014-21, test 22-23 |
| Price (GB + lag) | EUR/sqm | **99.4%** | Spatial CV | For forecasting |
| Rent (GB + STR) | EUR/sqm/month | **74.3%** | Random split | |

### Validation Against Idealista Listings (Q4 2025)

| Comparison | Pearson r | p-value | Interpretation |
|------------|-----------|---------|----------------|
| Idealista vs OMI | **0.883** | 0.0016 | Very strong positive |
| Idealista vs Model | **0.756** | 0.0185 | Strong positive |

- Average listing premium: **+15.5%** above OMI (expected for asking vs transaction prices)
- Undervalued confirmation rate: **100%** (model's undervalued picks confirmed by market)

### Key Findings

1. STR density is the top predictor (65% feature importance), far exceeding traditional demographic factors
2. Vacancy classification identifies 4 distinct market types across 7,850 municipalities
3. Model predictions correlate strongly (r=0.756) with Idealista listing prices
4. Model captures 91% of spatial autocorrelation (Moran's I: 0.76 to 0.07)

## Methodology

**Data sources**: OMI real estate quotations (7,850 municipalities, 2014-2023), ISTAT Census 2021 demographics, IRPEF income declarations (2012-2023), InsideAirbnb short-term rental data, Idealista listing prices for validation.

**Feature engineering**: Log-log hedonic pricing specification with demographic features (population, growth rate), economic features (income level, growth, national ratio), geographic features (coordinates, distance to coast/city), tourism features (STR density, revenue premium), and a 4-type vacancy classification (low/decline/tourist/mixed).

**Models**: Gradient Boosting regression with Optuna hyperparameter tuning. Benchmarked against OLS, Ridge, and Lasso baselines.

**Validation**: Three strategies -- random split, spatial cross-validation with held-out municipalities, and temporal split (train 2014-2021, test 2022-2023). External validation against Idealista listing prices in 10 major cities.

## Vacancy Classification

| Type | Count | Avg Price | Characteristics |
|------|-------|-----------|-----------------|
| Low Vacancy | 5,595 | 987 EUR/sqm | Normal market, <15% vacancy |
| Tourist Vacancy | 931 | 1,306 EUR/sqm | High tourism, seasonal homes |
| Decline Vacancy | 1,195 | 859 EUR/sqm | Depopulating, >20% vacancy |
| Mixed Vacancy | 129 | 1,651 EUR/sqm | Complex markets (tourism + decline) |

## Quick Start

```bash
# Install dependencies
devenv shell
# or: uv sync

# Run the full pipeline
python -m src.pipeline

# Train vacancy-aware price model
python src/train_with_vacancy.py

# Find undervalued municipalities
python src/find_undervalued.py

# Validate against Idealista listings
python src/validate_with_immobiliare.py

# View interactive map
python -m http.server 8080
# Open http://localhost:8080/vacancy_map.html
```

## Project Structure

```
configs/           Configuration (model params, data sources)
src/
  pipeline.py      Main data pipeline
  data/fetchers/   OMI, ISTAT, InsideAirbnb, Idealista fetchers
  features/        Vacancy classification, demographic/economic/tourism features
  models/          OLS, Ridge, Lasso, Gradient Boosting
  validation/      Idealista listing price correlation
data/              Raw, processed, and external validation data
outputs/           Predictions and validation results
tests/             Unit and integration tests
docs/              Methodology and data guides
```

## Requirements

- Python 3.12+
- Key dependencies: pandas, scikit-learn, geopandas, pydantic, pandera, loguru

See `pyproject.toml` for the full dependency list.

## Testing

```bash
pytest tests/
pytest tests/ --cov=src --cov-report=html
```

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| OMI | Real estate quotations (Agenzia delle Entrate) | 7,850 municipalities, 2014-2023 |
| ISTAT | Demographics, population, vacancy rates | All municipalities, Census 2021 |
| IRPEF | Income tax declarations (MEF) | Municipality-level, 2012-2023 |
| InsideAirbnb | Short-term rental listings | Milan, Florence, Bologna, Naples |
| Idealista | Listing prices for validation | 10 major cities, Q4 2025 |

## License

Data sourced from OMI (CC-BY, Agenzia delle Entrate), ISTAT (CC-BY 3.0), and InsideAirbnb (research use).
