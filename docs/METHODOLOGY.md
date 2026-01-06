# Statistical Methodology

This document describes the statistical methodology used in the Italian Real Estate Demographic Risk Model.

## 1. Hedonic Pricing Framework

### 1.1 Theoretical Foundation

The model uses a **hedonic pricing approach**, which decomposes property prices into implicit prices of individual attributes. Following Rosen (1974), property price P is modeled as a function of characteristics:

```
P = f(structural, location, neighborhood, market)
```

### 1.2 Functional Form

We use a **log-log specification** for the hedonic model:

```
log(Price) = β₀ + β₁·log(Population) + β₂·log(Income) + β₃·log(STR_Density)
           + β₄·Lat + β₅·Lon + β₆·Dist_City + ... + ε
```

**Rationale for log transformation**:
- Prices and many features are right-skewed
- Coefficients represent elasticities (% change interpretation)
- Reduces heteroscedasticity
- Multiplicative relationships become additive

## 2. Data Processing

### 2.1 OMI Price Aggregation

OMI provides price ranges (min/max) for multiple zones within each municipality. We aggregate to municipality level using:

1. **Price midpoint**: `price_mid = (Compr_min + Compr_max) / 2`
2. **Zone prioritization**: Weight central zones (B) more than peripheral (E)
3. **Property type filtering**: Focus on residential ("Abitazioni civili", "Abitazioni di tipo economico")
4. **Outlier removal**: Filter prices outside 200-20,000 EUR/sqm range

### 2.2 ISTAT Code Harmonization

ISTAT municipality codes require careful handling:
- OMI uses 7-digit codes (region prefix + 6-digit code)
- Other sources use 6-digit codes
- Solution: Extract last 6 digits: `istat_code = full_code % 1000000`

### 2.3 Short-Term Rental Proxy

For provinces without InsideAirbnb data, we create an STR proxy:

```
STR_proxy = Tourism_Intensity × Calibration_Ratio
```

Where:
```
Calibration_Ratio = mean(STR_density) / mean(Tourism_Intensity)
```

Calculated from provinces with both data sources (Milan, Florence, Bologna, Naples).

## 3. Feature Engineering

### 3.1 Demographic Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `pop_change_pct` | (Pop₂₀₂₃ - Pop₂₀₁₁) / Pop₂₀₁₁ × 100 | Demographic trend |
| `aging_index` | Pop₆₅₊ / Pop₀₋₁₄ | Age structure pressure |
| `dependency_ratio` | (Pop₀₋₁₄ + Pop₆₅₊) / Pop₁₅₋₆₄ | Economic burden |

### 3.2 Economic Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `income_ratio` | Income_local / Income_national | Relative wealth |
| `taxpayer_ratio` | Taxpayers / Population | Formal economy participation |
| `affordability_index` | Price × 70sqm / Annual_Income | Housing affordability |

### 3.3 Geographic Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `dist_major_city` | Haversine(location, nearest_city) | Access to economic centers |
| `dist_coast` | Haversine(location, nearest_coast) | Coastal premium |
| `northern` | 1 if region_code ∈ {1-8} else 0 | North-South divide |

**Haversine Formula**:
```
d = 2R × arcsin(√(sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)))
```

### 3.4 Tourism Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `tourism_intensity` | Arrivals / (Population × 1000) | Tourist pressure |
| `str_density` | Airbnb_listings / (Population × 1000) | STR market depth |
| `str_premium` | (Airbnb_nightly × 20 / 70) / LT_rent × 100 | STR vs long-term |

## 4. Model Specification

### 4.1 Baseline OLS Model

```
log(price_mid) = β₀ + Σ βᵢ·Xᵢ + ε

where X includes:
- log(population), pop_change_pct
- log(income), income_change_pct
- lat, lon, dist_major_city, dist_coast
- northern, coastal, urban (binary)
- tourism_intensity
```

**Results**: R² = 43.8% (prices), 21.0% (rents)

### 4.2 Gradient Boosting Model

**Algorithm**: Sequential ensemble of decision trees

**Hyperparameters** (tuned via Optuna, 30 trials):
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_estimators | 500 | Sufficient convergence |
| learning_rate | 0.072 | Optimized via Bayesian search |
| max_depth | 6 | Prevent overfitting |
| min_child_weight | 17 | Stronger regularization |
| subsample | 0.69 | Stochastic gradient boosting |
| colsample_bytree | 0.67 | Feature randomization |
| reg_alpha | 0.054 | L1 regularization |
| reg_lambda | 0.00008 | L2 regularization |

**Results** (Spatial Cross-Validation):
- Without STR: R² = 80.9% (prices), 64.6% (rents)
- With STR (baseline): R² = 81.9% ± 0.5% (prices)
- With STR (Optuna-tuned): R² = **84.8% ± 0.3%** (prices), 74.3% (rents)
- Temporal split (train 2014-21, test 2022-23): R² = 92.2%

### 4.3 Model Comparison

| Model | Price R² | Validation | Interpretation |
|-------|----------|------------|----------------|
| OLS | 37.8% | Random split | Linear relationships only |
| OLS + STR | 44.0% | Random split | STR adds ~6 points |
| GB | 80.9% | Random split | Captures nonlinearities |
| GB + STR (baseline) | 81.9% | Spatial CV | Proper holdout municipalities |
| GB + STR (Optuna) | **84.8%** | Spatial CV | +3.5% from hyperparameter tuning |
| GB + STR + lag | **99.4%** | Spatial CV | For forecasting (prices persistent) |

## 5. Feature Importance Analysis

### 5.1 Gradient Boosting Importance

Feature importance is measured by **impurity reduction** (mean decrease in Gini/MSE):

```
Importance(feature) = Σ (ΔImpurity at splits using feature) / n_trees
```

**Top 10 Features (Price Model with STR)**:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | str_density | 64.8% |
| 2 | urban | 9.3% |
| 3 | income_ratio | 8.5% |
| 4 | tourism_intensity | 3.6% |
| 5 | log_population | 2.6% |
| 6 | dist_coast | 2.2% |
| 7 | long | 1.9% |
| 8 | lat | 1.9% |
| 9 | dist_major_city | 1.8% |
| 10 | str_premium | 1.7% |

*Note: Feature importance varies with feature set and data subset. STR density dominates when STR features are included.*

### 5.2 STR vs Tourism Effect

The model reveals that STR density captures price effects **beyond** general tourism:

```
Tourism coefficient:     β = 0.029 (rent), 0.089 (price)
STR density coefficient: β = 0.156 (rent), 0.187 (price)
```

**Interpretation**: STR activity has a stronger marginal effect on prices than tourist arrivals alone, suggesting the Airbnb market creates additional housing demand beyond tourism amenity value.

## 6. Valuation Methodology

### 6.1 Residual-Based Undervaluation

Properties are classified as undervalued based on prediction residuals:

```
residual = log(actual_price) - log(predicted_price)
price_gap_pct = (exp(residual) - 1) × 100
```

**Classification Thresholds**:
| Category | Price Gap |
|----------|-----------|
| Severely undervalued | < -30% |
| Undervalued | -30% to -15% |
| Fair value | -15% to +15% |
| Overvalued | +15% to +30% |
| Severely overvalued | > +30% |

### 6.2 Smart Picks Scoring

Smart investment picks combine undervaluation with yield:

```
Combined_Score = α × (-price_gap_pct/100) + (1-α) × (gross_yield_pct/20)
```

Where α = 0.5 weights both equally.

**Selection Criteria**:
1. Price gap ≤ -15% (undervalued)
2. Gross yield ≥ 4% (cash-flow positive)
3. Ranked by combined score

### 6.3 Gross Yield Calculation

```
Gross_Yield = (Annual_Rent_per_sqm × 12) / Price_per_sqm × 100
```

Assuming 70 sqm typical apartment and 100% occupancy for long-term rental.

## 7. Validation

### 7.1 Comprehensive Regression Metrics

**Spatial Cross-Validation** (GroupKFold by municipality, 5 folds):

| Metric | Mean | Std | Interpretation |
|--------|------|-----|----------------|
| **R²** | 0.831 | ±0.010 | Explains 83.1% of price variance |
| **RMSE (EUR/sqm)** | 226 | ±38 | Typical prediction error |
| **MAE (EUR/sqm)** | 107 | ±7 | Average absolute error |
| **MAPE (%)** | 11.9 | ±0.2 | Average percentage error |
| **MedAE (EUR/sqm)** | 57 | ±1 | Median absolute error |

**Why Spatial CV matters**: Panel data (same municipalities across years) creates data leakage in random splits. Spatial CV holds out entire municipalities, testing true generalization to unseen locations.

**Temporal Validation** (train 2014-2021, test 2022-2023):

| Metric | Value |
|--------|-------|
| R² | 0.896 |
| RMSE (EUR/sqm) | 146 |
| MAE (EUR/sqm) | 84 |
| MAPE (%) | 11.0 |

### 7.2 Performance by Price Segment

| Price Range (EUR/sqm) | N | R² | MAE | MAPE |
|-----------------------|---|----|----|------|
| 0-500 (low) | 15,185 | 0.37 | 41 | 10.2% |
| 500-1,000 (medium) | 44,096 | 0.71 | 56 | **7.9%** |
| 1,000-2,000 (high) | 13,459 | 0.39 | 125 | 9.6% |
| 2,000+ (luxury) | 1,631 | -1.16 | 463 | 17.3% |

**Observations**:
- Best performance in 500-1,000 EUR range (most common prices)
- Poor performance in luxury segment due to small sample size and high variance
- Low-end properties harder to predict (heterogeneous quality)

### 7.3 Residual Diagnostics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean residual | 0.00001 | ✓ Unbiased predictions |
| Std residual | 0.123 | Typical log-scale error |
| Skewness | 0.658 | ⚠ Slight right skew |
| Kurtosis | 5.95 | ⚠ Heavy tails (outliers) |
| 5th percentile | -0.182 | - |
| 95th percentile | 0.199 | 90% within ±0.2 log units |

**Interpretation**: Residuals are approximately normal but with slight right skew and heavy tails, indicating some extreme underpredictions (actual > predicted) for outlier municipalities.

### 7.4 Spatial Autocorrelation (Moran's I)

| Metric | Raw Prices | Model Residuals | Reduction |
|--------|------------|-----------------|-----------|
| Moran's I (k=10) | 0.763 | 0.066 | **91%** |
| Z-score | 154.7 | 23.5 | - |

The model captures most spatial structure through location features (lat, long, distances). Remaining autocorrelation is statistically significant but small in magnitude.

### 7.5 Model Limitations

1. **Residual spatial autocorrelation**: Moran's I = 0.07 remains significant; spatial lag models could add 2-5% R²
2. **Micro-location**: Municipality-level misses neighborhood variation within cities
3. **STR data coverage**: Only 4 cities have direct Airbnb data; tourism intensity used as proxy elsewhere
4. **Price persistence**: High correlation (r=0.99) year-over-year limits detection of rapid changes

### 7.6 Model Improvement Experiments

Several approaches were tested to improve model performance:

| Approach | Overall R² | Luxury R² | Notes |
|----------|------------|-----------|-------|
| Baseline (XGBoost) | 81.9% | -140% | Poor luxury predictions |
| + Interaction features | 81.8% | -142% | No improvement |
| + Huber loss | 77.8% | - | Worse R², no skewness fix |
| + MAE loss | 77.1% | - | Worse R², worse skewness |
| + 5x Luxury weighting | 81.8% | **+48%** | Major luxury improvement |
| **+ Optuna tuning** | **84.8%** | - | **+3.5% improvement** |

**Key Findings**:

1. **Optuna hyperparameter tuning** provided the largest improvement (+3 percentage points R²). Key changes:
   - Increased `min_child_weight` from 1 to 17 (stronger regularization)
   - Reduced `colsample_bytree` from 0.8 to 0.67 (more feature sampling)
   - Added L1/L2 regularization (`reg_alpha=0.054`, `reg_lambda=0.00008`)

2. **Sample weighting** dramatically improved the luxury segment from R² = -140% (worse than mean prediction) to R² = +48% (useful predictions). This is recommended when luxury segment performance matters.

3. **Robust loss functions** (Huber, MAE, Quantile) did not improve residual skewness and reduced overall R². The skewness is inherent to the price distribution, not caused by the loss function.

4. **Interaction features** (coastal×STR, urban×income, etc.) provided no improvement, suggesting XGBoost already captures these nonlinear relationships.

## 8. Reproducibility

All random processes use `random_state=42`:
- Train/test split
- Gradient Boosting initialization
- Cross-validation folds

Pipeline produces identical results when run with same configuration.

## References

1. Rosen, S. (1974). Hedonic Prices and Implicit Markets. *Journal of Political Economy*, 82(1), 34-55.
2. Barron, K., Kung, E., & Proserpio, D. (2021). The Effect of Home-Sharing on House Prices and Rents. *Marketing Science*, 40(1), 23-47.
3. Garcia-López, M. À., et al. (2020). Do Short-Term Rental Platforms Affect Housing Markets? *Journal of Urban Economics*, 119.
