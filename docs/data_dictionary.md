# Data Dictionary

This document defines all columns in the processed datasets.

## Primary Keys

| Column | Type | Description |
|--------|------|-------------|
| `istat_code` | string(6) | ISTAT municipality code, zero-padded to 6 digits |
| `anno` | int | Year of observation |
| `prov_code` | string(3) | Province code, zero-padded to 3 digits |

---

## Price Data (OMI)

Source: Agenzia delle Entrate - Osservatorio Mercato Immobiliare

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `prezzo_medio` | float | EUR/sqm | Average purchase price (midpoint of OMI range) |
| `prezzo_min` | float | EUR/sqm | Minimum purchase price from OMI |
| `prezzo_max` | float | EUR/sqm | Maximum purchase price from OMI |
| `affitto_medio` | float | EUR/sqm/month | Average rental price |
| `n_zones` | int | count | Number of OMI zones in municipality |

---

## Demographic Data (ISTAT)

Source: Italian National Institute of Statistics

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `popolazione` | int | persons | Total resident population |
| `pop_change_pct` | float | % | Population change since base year |
| `pop_declining` | int | 0/1 | 1 if population declined >2% |
| `pop_growing_fast` | int | 0/1 | 1 if population grew >5% |
| `pop_0_14` | int | persons | Population aged 0-14 |
| `pop_15_64` | int | persons | Population aged 15-64 (working age) |
| `pop_65_plus` | int | persons | Population aged 65+ |
| `aging_index` | float | ratio | Pop 65+ / Pop 0-14 |
| `dependency_ratio` | float | ratio | (Pop 0-14 + 65+) / Pop 15-64 |

---

## Economic Data (IRPEF)

Source: Ministry of Economy and Finance

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `avg_income` | float | EUR/year | Average declared income per taxpayer |
| `income_change_pct` | float | % | Income change since base year |
| `n_contribuenti` | int | count | Number of taxpayers |
| `reddito_totale` | float | EUR | Total declared income in municipality |
| `income_ratio` | float | ratio | Local income / National average |
| `taxpayer_ratio` | float | ratio | Taxpayers / Population |

---

## Geographic Data

Source: Computed from coordinates using Haversine formula

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `lat` | float | degrees | Latitude (WGS84) |
| `long` | float | degrees | Longitude (WGS84) |
| `dist_major_city` | float | km | Distance to nearest major city (Milan, Rome, Naples, etc.) |
| `dist_coast` | float | km | Distance to nearest coastline point |
| `nearest_city` | string | name | Name of nearest major city |

### Location Flags

| Column | Type | Description |
|--------|------|-------------|
| `northern` | int (0/1) | 1 if in Northern Italy (regions 1-8) |
| `coastal` | int (0/1) | 1 if within 20km of coast |
| `alpine_zone` | int (0/1) | 1 if in Alpine region (Piemonte, Valle d'Aosta, Trentino) |
| `urban` | int (0/1) | 1 if population > 50,000 |
| `large_city` | int (0/1) | 1 if population > 500,000 |

---

## Tourism Data

Source: ISTAT Tourism Statistics

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `arrivals` | int | persons | Tourist arrivals per year |
| `nights` | int | nights | Tourist overnight stays per year |
| `tourism_intensity` | float | arrivals/1000 pop | Tourist arrivals per 1,000 residents |
| `tourism_level` | string | category | low/medium/high/very_high |

---

## Short-Term Rental Data (Airbnb)

Source: InsideAirbnb (Milan, Florence, Bologna, Naples)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `airbnb_listings` | int | count | Number of Airbnb listings |
| `airbnb_price_median` | float | EUR/night | Median nightly price |
| `airbnb_price_mean` | float | EUR/night | Mean nightly price |
| `airbnb_reviews` | int | count | Total reviews (proxy for bookings) |
| `str_density` | float | listings/1000 pop | STR listings per 1,000 residents |
| `str_premium` | float | % | (STR monthly revenue / LT rent - 1) × 100 |
| `source_city` | string | name | InsideAirbnb source city |

### STR Proxy (for areas without Airbnb data)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `str_proxy` | float | estimated density | Tourism-calibrated STR proxy |
| `has_str_data` | int (0/1) | | 1 if direct Airbnb data available |

---

## Log-Transformed Features

All log-transformed features use `log1p(x) = log(1 + x)` for zero-safety.

| Column | Source | Description |
|--------|--------|-------------|
| `log_population` | log1p(popolazione) | Log population |
| `log_income` | log1p(avg_income) | Log average income |
| `log_price_mid` | log1p(prezzo_medio) | Log price (model target) |
| `log_rent_mid` | log1p(affitto_medio) | Log rent (model target) |
| `log_str_density` | log1p(str_density) | Log STR density |
| `log_str_price` | log1p(airbnb_price_median) | Log Airbnb price |
| `log_tourism` | log1p(tourism_intensity) | Log tourism intensity |

---

## Model Outputs

### Predictions

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `predicted_price` | float | EUR/sqm | Model-predicted price |
| `predicted_rent` | float | EUR/sqm/month | Model-predicted rent |
| `residual` | float | log units | actual - predicted (log scale) |
| `price_gap_pct` | float | % | (exp(residual) - 1) × 100 |

### Valuation Categories

| Column | Type | Description |
|--------|------|-------------|
| `valuation_category` | string | severely_undervalued / undervalued / fair_value / overvalued / severely_overvalued |
| `valuation_score` | float | Combined undervaluation + yield score |
| `gross_yield_pct` | float | (Annual rent × 12) / Price × 100 |

---

## Administrative Hierarchy

| Column | Type | Description |
|--------|------|-------------|
| `nome` | string | Municipality name |
| `provincia` | string | Province name |
| `sigla_provincia` | string(2) | Province abbreviation (MI, RM, NA, etc.) |
| `regione` | string | Region name |
| `cod_regione` | int | Region code (1-20) |

### Region Codes

| Code | Region | Macro-region |
|------|--------|--------------|
| 1 | Piemonte | Nord-Ovest |
| 2 | Valle d'Aosta | Nord-Ovest |
| 3 | Lombardia | Nord-Ovest |
| 4 | Trentino-Alto Adige | Nord-Est |
| 5 | Veneto | Nord-Est |
| 6 | Friuli-Venezia Giulia | Nord-Est |
| 7 | Liguria | Nord-Ovest |
| 8 | Emilia-Romagna | Nord-Est |
| 9 | Toscana | Centro |
| 10 | Umbria | Centro |
| 11 | Marche | Centro |
| 12 | Lazio | Centro |
| 13 | Abruzzo | Sud |
| 14 | Molise | Sud |
| 15 | Campania | Sud |
| 16 | Puglia | Sud |
| 17 | Basilicata | Sud |
| 18 | Calabria | Sud |
| 19 | Sicilia | Isole |
| 20 | Sardegna | Isole |

---

## File Descriptions

### Raw Data (`data/raw/`)

| File | Description |
|------|-------------|
| `omi/valori.csv` | OMI price quotations (semicolon-separated) |
| `omi/zone.csv` | OMI zone definitions |
| `omi/comuni.csv` | OMI municipality list |
| `istat/main.csv` | Municipality metadata |
| `istat/population_trends_2018_2021.csv` | Population time series |
| `irpef/*.csv` | Income data by year |
| `airbnb/{city}/listings.csv.gz` | Airbnb listings by city |
| `geo/comuni.geojson` | Municipality boundaries |

### Processed Data (`data/processed/`)

| File | Description |
|------|-------------|
| `enhanced_features_with_str.csv` | All features, municipality-year level |
| `price_model_with_str.csv` | Price predictions and residuals |
| `rent_model_with_str.csv` | Rent predictions and residuals |
| `valuations_with_str_model.csv` | Full valuation analysis |
| `smart_picks_with_str.csv` | Top investment opportunities |
| `model_metrics.csv` | R² scores for all model variants |
| `airbnb_by_municipality.csv` | Aggregated Airbnb data |
| `airbnb_by_province.csv` | Province-level Airbnb summary |
| `tourism_intensity_by_province.csv` | Tourism statistics |
