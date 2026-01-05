# OMI Historical Data Download Guide

## Overview

Download 20 years of real estate price quotations (2004-2024) from Agenzia delle Entrate.

**Required**: SPID, CIE, or Fisconline/Entratel credentials

---

## Step-by-Step Instructions

### Step 1: Access the Portal

1. Go to: https://telematici.agenziaentrate.gov.it/Main/index.jsp
2. Click **"Accedi"** (Login)
3. Authenticate with SPID, CIE, or credentials

### Step 2: Navigate to OMI Service

1. After login, look for: **"Servizi ipotecari e catastali, Osservatorio Mercato Immobiliare"**
2. Click on it
3. Select: **"Forniture OMI - Quotazioni Immobiliari"**

### Step 3: Request Data Download

You can download data by:
- **Entire national territory** (recommended for full analysis)
- By region
- By province
- By single municipality

**For our analysis, download: ENTIRE NATIONAL TERRITORY**

### Step 4: Select Time Period

Download ALL available semesters:

| Year | Semesters |
|------|-----------|
| 2004 | S1, S2 |
| 2005 | S1, S2 |
| 2006 | S1, S2 |
| 2007 | S1, S2 |
| 2008 | S1, S2 |
| 2009 | S1, S2 |
| 2010 | S1, S2 |
| 2011 | S1, S2 |
| 2012 | S1, S2 |
| 2013 | S1, S2 |
| 2014 | S1, S2 |
| 2015 | S1, S2 |
| 2016 | S1, S2 |
| 2017 | S1, S2 |
| 2018 | S1, S2 |
| 2019 | S1, S2 |
| 2020 | S1, S2 |
| 2021 | S1, S2 |
| 2022 | S1, S2 |
| 2023 | S1, S2 |
| 2024 | S1, S2 (if available) |

**Total: ~42 files** (21 years × 2 semesters)

### Step 5: Download Format

- Select **CSV format** if available
- Files will be named like: `quotazioni_YYYYS.csv`

### Step 6: Save Files

Save all downloaded files to:
```
/home/guyfawkes/price-analysis/data/raw/omi/historical/
```

Create the directory structure:
```bash
mkdir -p data/raw/omi/historical
```

---

## Expected File Structure

After download, you should have:
```
data/raw/omi/historical/
├── quotazioni_2004S1.csv
├── quotazioni_2004S2.csv
├── quotazioni_2005S1.csv
├── ...
├── quotazioni_2024S1.csv
└── quotazioni_2024S2.csv
```

## File Format

Each CSV should contain columns similar to:
- `Comune_ISTAT` - Municipality code
- `Zona` - OMI zone
- `Tipologia` - Property type
- `Stato` - Condition (normal/excellent)
- `Compr_min` - Min purchase price (EUR/sqm)
- `Compr_max` - Max purchase price (EUR/sqm)
- `Loc_min` - Min rent (EUR/sqm/month)
- `Loc_max` - Max rent (EUR/sqm/month)

---

## After Download

Once files are saved, run:
```bash
python scripts/ingest_historical_omi.py
```

This will:
1. Consolidate all files into a single dataset
2. Parse year/semester from filenames
3. Standardize column names
4. Save to `data/processed/omi_historical.parquet`

---

## Troubleshooting

### "Service not available"
- Try during business hours (Italian time)
- The portal may have maintenance windows

### Download limit reached
- There may be daily download limits
- Try downloading by region if national fails

### File format issues
- Older files (2004-2010) may have different column names
- The ingestion script handles format variations

---

## Data Size Estimates

| Period | Estimated Size |
|--------|---------------|
| Per semester | ~15-20 MB |
| Full 20 years | ~600-800 MB |
| After compression | ~100-150 MB |

---

## Questions?

If you encounter issues, note:
- The exact error message
- Which semester/year failed
- Screenshot of the portal page

We can adapt the ingestion pipeline accordingly.
