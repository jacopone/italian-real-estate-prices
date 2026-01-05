# Real Estate Price Prediction Model: Status & Roadmap

## Current State: Production Model

| Model | Target | R² Score | Validation | Status |
|-------|--------|----------|------------|--------|
| **Price (XGBoost + STR)** | EUR/sqm | **84.4%** | Spatial CV | ✅ Production |
| **Price (GB + STR + Lag)** | EUR/sqm | **99.4%** | Spatial CV | ✅ Forecasting |
| **Rent (Gradient Boosting + STR)** | EUR/sqm/month | **74.3%** | Spatial CV | ✅ Production |
| Price (Temporal Split) | EUR/sqm | 92.2% | 2014-21→22-23 | Validation |

**Key Findings**:
1. STR density is the #1 predictor (64.7% feature importance) for price levels
2. Lagged price dominates (98.7%) for forecasting - prices are highly persistent
3. Spatial autocorrelation reduced by 91% (Moran's I: 0.76 → 0.07)

---

## Model Evolution

| Version | Model | R² | Features | Notes |
|---------|-------|-------|----------|-------|
| v0.1 | OLS (pop_change → price_change) | 0.3% | 1 | Initial POC - useless |
| v0.2 | OLS (log_income + log_pop → log_price) | 32.3% | 2 | Hedonic baseline |
| v0.3 | OLS + Year FE | 35.5% | 11 | Time controls |
| v0.4 | GWR (spatially varying coefficients) | ~45% | 2 | Spatial heterogeneity |
| v0.5 | Gradient Boosting (full features) | 80.9% | 25+ | ML ensemble |
| **v1.0** | **GB + STR features** | **83.2%** | 29 | **Current production** |

---

## Current Feature Set (29 features)

### Demographics
- `log_population` - Log-transformed population
- `pop_change_pct` - Population change 2011-2023
- `pop_declining` - Binary flag for declining municipalities
- `pop_growing_fast` - Binary flag for fast-growing municipalities

### Economics
- `log_income` - Log-transformed average IRPEF income
- `income_change_pct` - Income growth rate
- `income_ratio` - Ratio to national average

### Geography
- `lat`, `long` - Coordinates
- `dist_major_city` - Distance to nearest major city (km)
- `dist_coast` - Distance to coast (km)
- `northern`, `coastal`, `urban`, `alpine_zone` - Location flags

### Tourism & STR (Short-Term Rentals)
- `tourism_intensity` - Arrivals per 1000 residents
- `str_density` - Airbnb listings per 1000 residents
- `log_str_density` - Log-transformed STR density
- `str_price` - Median Airbnb nightly rate
- `str_premium` - STR revenue vs long-term rent ratio

---

## Completed Work

### ✅ Data Integration
- OMI prices (7,904 municipalities, 2004-2024)
- ISTAT demographics (population, age structure)
- IRPEF income (2012-2023)
- InsideAirbnb STR data (Milan, Florence, Bologna, Naples)
- Tourism arrivals by province

### ✅ Feature Engineering
- Hedonic pricing specification (log-log)
- Geographic distance features (haversine)
- STR density aggregation to municipality level
- Population trajectory classification

### ✅ Spatial Analysis
- GWR (Geographically Weighted Regression) implemented
- Coefficients vary by region (income β: 0.18–0.39)
- North/South heterogeneity captured

### ✅ Time Series Analysis
- Panel cross-correlation (lag=0 optimal)
- Distributed lag model (cumulative effect: 0.20)
- Trajectory divergence analysis (decliners vs growers)

### ✅ ML Models
- Gradient Boosting (n_estimators=500, max_depth=6)
- XGBoost and LightGBM benchmarked
- SHAP feature importance computed
- Cross-validation (5-fold)

### ✅ Validation & Diagnostics (2026-01-05)
- **Temporal split validation**: Train 2014-2021 → Test 2022-2023 (R²=92.2%)
- **Spatial CV (GroupKFold by municipality)**: R²=84.4% ± 2.1%
- **Moran's I spatial autocorrelation**: Reduced from 0.76 to 0.07 (91% improvement)
- **Lagged price feature**: R²=99.4% for forecasting (prices highly persistent)

---

## Remaining Improvements

### Priority 1: Model Enhancements

| Task | Expected Impact | Effort |
|------|-----------------|--------|
| XGBoost comparison | +1-3% R² | Low |
| LightGBM comparison | +1-2% R² | Low |
| Hyperparameter tuning (Optuna) | +1-2% R² | Medium |
| Stacked ensemble | +2-4% R² | Medium |

### Priority 2: Spatial Refinements

| Task | Expected Impact | Effort |
|------|-----------------|--------|
| Moran's I spatial autocorrelation | Diagnostic | Low |
| Spatial lag model (SAR) | +2-5% R² | Medium |
| GTWR (temporal + spatial) | +3-7% R² | High |
| Neighbor-based features | +1-3% R² | Medium |

### Priority 3: Feature Expansion

| Task | Expected Impact | Effort |
|------|-----------------|--------|
| Age dependency ratio | +1-2% R² | Low |
| University presence | +0.5-1% R² | Low |
| Crime rate (ISTAT) | +0.5-1% R² | Medium |
| Environmental quality (ISPRA) | +0.5-1% R² | Medium |
| Accessibility index | +1-2% R² | Medium |
| Interest rate interaction | +0.5-1% R² | Low |

### Priority 4: Temporal Dynamics

| Task | Expected Impact | Effort |
|------|-----------------|--------|
| Panel fixed effects (municipality FE) | Better causal ID | Medium |
| Lagged dependent variable | +1-2% R² | Low |
| Rolling window features | +1-2% R² | Medium |

---

## Quick Wins (Can Do Now)

1. **XGBoost benchmark** - Drop-in replacement, often 1-3% better
2. **Moran's I test** - Verify spatial autocorrelation in residuals
3. **Age dependency ratio** - Already available in ISTAT data
4. **Lagged price feature** - Previous year's price as predictor
5. **Region fixed effects** - Simple dummy encoding

---

## Technical Debt

- [ ] No saved model artifacts (`.pkl` files)
- [ ] Missing integration tests for full pipeline
- [ ] GWR code was ad-hoc, not in `src/models/`
- [ ] SHAP analysis not reproducible from CLI

---

## Data Sources Status

| Source | Status | Coverage |
|--------|--------|----------|
| **OMI** | ✅ Complete | 7,904 municipalities, 2004-2024 |
| **ISTAT Population** | ✅ Complete | All municipalities, 2002-2025 |
| **IRPEF Income** | ✅ Complete | All municipalities, 2012-2023 |
| **InsideAirbnb** | ⚠️ Partial | 4 cities only |
| **Tourism** | ⚠️ Province-level | Not municipality-level |
| **Crime** | ❌ Not integrated | Available from ISTAT |
| **Environment** | ❌ Not integrated | Available from ISPRA |
| **Accessibility** | ❌ Not integrated | Available from MIT |

---

## Key References

1. Rosen (1974) - Hedonic Prices and Implicit Markets
2. Anselin (1988) - Spatial Econometrics
3. [MDPI 2024](https://www.mdpi.com/2813-2203/3/1/3) - XGBoost for house prices
4. [MDPI 2025](https://www.mdpi.com/2813-8090/2/4/16) - MLP for Italian housing
5. [Tandfonline](https://www.tandfonline.com/doi/abs/10.1080/13658810802672469) - GTWR methodology

---

## Changelog

- **2026-01-05**: Updated to reflect actual model performance (R²=83.2%)
- **2026-01-03**: Initial POC documented (R²=0.3%, superseded)
