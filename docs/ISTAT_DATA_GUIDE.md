# ISTAT Data Acquisition Guide

## Required Datasets for Demographic Risk Analysis

To test the hypothesis that demographic decline predicts real estate depreciation, we need the following ISTAT datasets:

### 1. Population by Municipality (DCIS_POPRES)

**Source:** http://dati.istat.it/Index.aspx?DataSetCode=DCIS_POPRES

**Required fields:**
- Municipality (ITTER107)
- Year (TIME)
- Age class (ETA1)
- Sex (optional)
- Total population

**Download steps:**
1. Go to http://dati.istat.it/?lang=en
2. Navigate to "Population and Households" → "Population"
3. Select "Resident population on 1st January"
4. Filter:
   - Territory: All municipalities (or select regions of interest)
   - Time: 2010-2024
   - Age: All ages OR age bands (0-14, 15-64, 65+)
5. Export as CSV

### 2. Demographic Balance (DCIS_BILPOP)

**Source:** http://dati.istat.it/Index.aspx?DataSetCode=DCIS_BILPOP

**Required fields:**
- Municipality
- Year
- Births (nati vivi)
- Deaths (morti)
- Internal migration in/out (iscritti/cancellati da altri comuni)
- International migration in/out (iscritti/cancellati dall'estero)

### 3. ISTAT Municipality Codes

**Source:** https://www.istat.it/storage/codici-unita-amministrative/

Download the latest "Elenco comuni" file to map:
- ISTAT municipality code → Municipality name
- Province code
- Region code
- Population
- Altitude (mountain/hill/plain classification)
- Coastal flag

## File Naming Convention

Place downloaded files in `data/raw/istat/` with these names:
- `population_by_municipality.csv` - Population by age and year
- `demographic_balance.csv` - Births, deaths, migration
- `municipality_codes.csv` - Administrative codes and metadata

## API Alternative

ISTAT provides a JSON-stat API at: http://dati.istat.it/

Example API call:
```
http://sdmx.istat.it/SDMXWS/rest/data/DCIS_POPRES/.TOTAL.9.1....?format=jsondata
```

This requires understanding SDMX query syntax. Consider using the `pyjstat` library for parsing.

## Quick Start: Manual Download

For quick testing, download just these two things:

1. **Population summary (2010-2023):**
   - Go to https://demo.istat.it/app/?l=en&a=2024&i=D2P
   - Download "Resident population by age, sex and year of birth"
   - Select all municipalities

2. **ISTAT codes:**
   - Download from https://www.istat.it/storage/codici-unita-amministrative/Elenco-comuni-italiani.csv

## Expected Schema

After processing, the demographic data should have this structure:

```csv
istat_code,year,population,pop_0_14,pop_15_64,pop_65_plus,births,deaths,migration_internal,migration_international
001001,2010,44103,5432,28901,9770,312,498,145,-23
001001,2011,43987,5298,28756,9933,298,512,132,-18
...
```
