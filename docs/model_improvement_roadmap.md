# Real Estate Price Prediction Model: Improvement Roadmap

## Current State: POC Model

**R² = 0.016 (1.6%)** - Demographics alone explain almost nothing

```
price_change = 0.49% + 0.25×italian_pop_change + 0.03×foreign_pop_change
```

---

## Why Our Model Has Low Explanatory Power

According to academic literature, real estate prices are determined by **multiple categories of factors**:

| Category | Our Model | Required |
|----------|-----------|----------|
| Demographics | ✅ Basic | Need age structure, household formation |
| Structural (property) | ❌ Missing | Size, quality, type, age of buildings |
| Location/Spatial | ❌ Missing | Distance to CBD, amenities, accessibility |
| Economic | ❌ Missing | Income, employment, GDP |
| Financial | ❌ Missing | Interest rates, mortgage availability |
| Environmental | ❌ Missing | Climate, pollution, natural amenities |

---

## Literature Review: What Works

### Best Performing Models

| Model Type | Typical R² | Notes |
|------------|------------|-------|
| Linear Regression | 0.60-0.75 | Baseline, easy to interpret |
| Random Forest | 0.80-0.90 | Good for non-linear relationships |
| XGBoost | 0.85-0.95 | Best overall performance |
| GWR (Spatial) | 0.88-0.93 | Captures geographic heterogeneity |
| GTWR (Spatial+Temporal) | 0.92-0.95 | Best for panel data |
| Neural Networks | 0.85-0.95 | Complex but powerful |

### Most Important Features (from ML studies)

1. **OverallQual** - Property quality rating (strongest predictor)
2. **GrLivArea** - Living area square footage
3. **Location/Latitude** - Spatial position
4. **TotalBsmtSF** - Basement size
5. **GarageCars** - Garage capacity
6. **YearBuilt** - Age of property
7. **Neighborhood** - Categorical location

### Italian-Specific Research

From MDPI (2025) study on Italian provinces:
- **Variables used**: house stock, incomes, population, employment, inflation, interest rates
- **OMI data**: Average price of civil properties in "normal" conditions, Zone B
- **Model**: MLP Neural Network on 99 provinces (2005-2020)

---

## Recommended Variable Categories

### Tier 1: Essential (High Impact)

| Variable | Source | Expected Impact |
|----------|--------|-----------------|
| **Household income** | ISTAT | +++ (strongest economic driver) |
| **Employment rate** | ISTAT | ++ |
| **Interest rates** | Bank of Italy | ++ (affordability) |
| **Population density** | ISTAT | ++ |
| **Age structure** (% elderly) | ISTAT | ++ (demographic quality) |

### Tier 2: Important (Medium Impact)

| Variable | Source | Expected Impact |
|----------|--------|-----------------|
| **Accessibility index** | ISTAT/Ministry | + (transport infrastructure) |
| **Service availability** | ISTAT | + (schools, hospitals) |
| **Tourism intensity** | ISTAT | + (demand driver) |
| **Building stock age** | Censimento | + (quality proxy) |
| **Rental yield** | OMI | + (investment attractiveness) |

### Tier 3: Refinement (Lower but Significant)

| Variable | Source | Expected Impact |
|----------|--------|-----------------|
| **Crime rate** | ISTAT/Ministry | - |
| **Environmental quality** | ISPRA | + |
| **University presence** | MIUR | + |
| **Coastal/Mountain** | GIS | +/- |
| **Climate index** | ISTAT | + |

---

## Spatial Modeling Requirements

### Why Standard OLS Fails

Real estate exhibits **spatial autocorrelation** - prices in nearby areas are correlated. This violates OLS assumptions and requires:

1. **Spatial Lag Model (SLM)**: Price depends on neighbors' prices
   ```
   Y = ρWY + Xβ + ε
   ```

2. **Spatial Error Model (SEM)**: Errors are spatially correlated
   ```
   Y = Xβ + λWu + ε
   ```

3. **Geographically Weighted Regression (GWR)**: Coefficients vary by location
   - Allows different relationships in different regions
   - North vs South Italy may have different dynamics

### Recommended: GTWR (Geographically and Temporally Weighted Regression)

From literature: **GTWR reduced errors by 46.4%** vs standard OLS and achieved **R² = 0.93** for housing prices.

---

## Data Sources for Italy

### Available from ISTAT

| Dataset | Coverage | Variables |
|---------|----------|-----------|
| Censimento popolazione | Municipal | Demographics, housing stock |
| Redditi IRPEF | Municipal | Declared income |
| Occupazione | Provincial | Employment rates |
| Imprese | Municipal | Business density |
| Turismo | Municipal | Tourism arrivals |
| Indicatori territoriali | Municipal | Quality of life composite |

### Available from Other Sources

| Source | Data |
|--------|------|
| **OMI (Agenzia Entrate)** | Property prices, rental values |
| **Bank of Italy** | Interest rates, mortgage data |
| **ISPRA** | Environmental indicators |
| **Ministero Infrastrutture** | Accessibility, transport |
| **OpenStreetMap** | POI density, amenities |

---

## Implementation Roadmap

### Phase 1: Data Enrichment (2-3 weeks)

```
Priority data to add:
1. Household income by municipality (ISTAT IRPEF)
2. Employment rate by province (ISTAT)
3. National interest rates (Bank of Italy)
4. Population density and urban/rural classification
5. Age dependency ratio (from existing ISTAT data)
```

### Phase 2: Spatial Model (1-2 weeks)

```python
# Upgrade from OLS to spatial models
from pysal.model import spreg

# Spatial Lag Model
model_slm = spreg.GM_Lag(y, X, w=weights_matrix)

# Geographically Weighted Regression
from mgwr.gwr import GWR
model_gwr = GWR(coords, y, X, bw=bandwidth)
```

### Phase 3: Machine Learning (1-2 weeks)

```python
# XGBoost for price prediction
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8
)

# Feature importance analysis
importance = model.feature_importances_
```

### Phase 4: Temporal Dynamics (1 week)

```python
# Panel data models with fixed effects
from linearmodels.panel import PanelOLS

# GTWR for spatio-temporal variation
# Requires specialized implementation
```

---

## Expected Improvement

| Model Stage | Expected R² | Improvement |
|-------------|-------------|-------------|
| Current (demographics only) | 0.016 | Baseline |
| + Economic variables | 0.15-0.25 | 10-15x |
| + Spatial structure | 0.40-0.60 | 25-40x |
| + ML ensemble | 0.70-0.85 | 45-55x |
| + Temporal dynamics | 0.80-0.90 | 50-60x |

---

## Key References

1. Rosen (1974) - Hedonic Prices and Implicit Markets (foundational theory)
2. Anselin (1988) - Spatial Econometrics: Methods and Models
3. [MDPI 2024](https://www.mdpi.com/2813-2203/3/1/3) - XGBoost for house prices
4. [MDPI 2025](https://www.mdpi.com/2813-8090/2/4/16) - MLP for Italian housing market
5. [Tandfonline](https://www.tandfonline.com/doi/abs/10.1080/13658810802672469) - GTWR methodology
6. [ScienceDirect Istanbul](https://www.sciencedirect.com/science/article/abs/pii/S0264837722002101) - GWR case study

---

## Quick Wins (Can Do Now)

1. **Add income data** - Single biggest improvement potential
2. **Add spatial weights** - Account for neighbor effects
3. **Try Random Forest** - Quick ML baseline
4. **Segment by region** - Separate North/South models
5. **Add lagged variables** - Previous year's price change
