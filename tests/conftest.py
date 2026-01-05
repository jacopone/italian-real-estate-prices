"""Pytest fixtures for testing the Italian Real Estate Risk Model.

Provides reusable test fixtures with sample data for unit and integration tests.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Return data directory."""
    return project_root / "data"


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create temporary test data directory."""
    (tmp_path / "raw" / "omi").mkdir(parents=True)
    (tmp_path / "raw" / "istat").mkdir(parents=True)
    (tmp_path / "processed").mkdir(parents=True)
    return tmp_path


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_omi_data() -> pd.DataFrame:
    """Create sample OMI price data."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        "Area_territoriale": np.random.choice(
            ["NORD-OVEST", "NORD-EST", "CENTRO", "SUD"], n
        ),
        "Regione": np.random.choice(
            ["LOMBARDIA", "TOSCANA", "LAZIO", "CAMPANIA"], n
        ),
        "Prov": np.random.choice(["MI", "FI", "RM", "NA"], n),
        "Comune_ISTAT": [f"0{i:05d}" for i in range(n)],
        "Comune_descrizione": [f"Comune_{i}" for i in range(n)],
        "Fascia": np.random.choice(["B", "C", "D", "E"], n),
        "Zona": [f"Z{i % 5}" for i in range(n)],
        "Cod_Tip": np.random.choice([20, 21, 22], n),
        "Descr_Tipologia": np.random.choice(
            ["Abitazioni civili", "Abitazioni di tipo economico"], n
        ),
        "Stato": "NORMALE",
        "Stato_prev": "P",
        "Compr_min": np.random.uniform(500, 2000, n),
        "Compr_max": np.random.uniform(2000, 5000, n),
        "Loc_min": np.random.uniform(3, 8, n),
        "Loc_max": np.random.uniform(8, 15, n),
        "file": "QI_test_20231_VALORI.csv",
    })


@pytest.fixture
def sample_municipality_data() -> pd.DataFrame:
    """Create sample municipality metadata."""
    np.random.seed(42)
    n = 50

    return pd.DataFrame({
        "pro_com_t": [f"{i:06d}" for i in range(n)],
        "comune": [f"Comune_{i}" for i in range(n)],
        "lat": np.random.uniform(36, 47, n),
        "long": np.random.uniform(7, 18, n),
        "den_prov": np.random.choice(["Milano", "Firenze", "Roma", "Napoli"], n),
        "sigla": np.random.choice(["MI", "FI", "RM", "NA"], n),
        "den_reg": np.random.choice(
            ["Lombardia", "Toscana", "Lazio", "Campania"], n
        ),
        "cod_reg": np.random.choice([3, 9, 12, 15], n),
    })


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create sample feature DataFrame for model training."""
    np.random.seed(42)
    n = 200

    return pd.DataFrame({
        "istat_code": [f"{i:06d}" for i in range(n)],
        "anno": np.random.choice([2022, 2023], n),
        "prezzo_medio": np.random.uniform(800, 5000, n),
        "affitto_medio": np.random.uniform(5, 15, n),
        "popolazione": np.random.uniform(1000, 500000, n),
        "pop_change_pct": np.random.uniform(-10, 10, n),
        "avg_income": np.random.uniform(15000, 40000, n),
        "income_change_pct": np.random.uniform(-5, 15, n),
        "lat": np.random.uniform(36, 47, n),
        "long": np.random.uniform(7, 18, n),
        "dist_major_city": np.random.uniform(5, 300, n),
        "dist_coast": np.random.uniform(0, 200, n),
        "coastal": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "northern": np.random.choice([0, 1], n, p=[0.5, 0.5]),
        "urban": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "tourism_intensity": np.random.uniform(0, 5000, n),
        "str_density": np.random.uniform(0, 50, n),
    })


@pytest.fixture
def sample_predictions(sample_features: pd.DataFrame) -> pd.Series:
    """Create sample model predictions."""
    # Simulate predictions close to actual with some noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, len(sample_features))
    return pd.Series(
        np.log(sample_features["prezzo_medio"]) + noise,
        index=sample_features.index,
        name="prediction",
    )


# =============================================================================
# CONFIG FIXTURES
# =============================================================================


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Create sample configuration dictionary."""
    return {
        "paths": {
            "data_dir": "data",
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "outputs_dir": "outputs",
        },
        "omi": {
            "property_types": ["Abitazioni civili"],
            "min_year": 2020,
            "max_year": 2023,
        },
        "model": {
            "test_size": 0.2,
            "random_state": 42,
            "gb_params": {
                "n_estimators": 100,
                "max_depth": 4,
            },
        },
    }


# =============================================================================
# MODEL FIXTURES
# =============================================================================


@pytest.fixture
def trained_ols_model(sample_features: pd.DataFrame):
    """Create a fitted OLS model."""
    from src.models.regression import OLSModel

    feature_cols = ["lat", "long", "popolazione", "avg_income"]
    X = sample_features[feature_cols].fillna(0)
    y = np.log(sample_features["prezzo_medio"])

    model = OLSModel()
    model.fit(X, y)
    return model


@pytest.fixture
def trained_gb_model(sample_features: pd.DataFrame):
    """Create a fitted Gradient Boosting model."""
    from src.models.ensemble import GradientBoostingModel

    feature_cols = ["lat", "long", "popolazione", "avg_income"]
    X = sample_features[feature_cols].fillna(0)
    y = np.log(sample_features["prezzo_medio"])

    model = GradientBoostingModel(n_estimators=50, max_depth=3)
    model.fit(X, y)
    return model


# =============================================================================
# TRANSFORMER FIXTURES
# =============================================================================


@pytest.fixture
def fitted_log_transformer(sample_features: pd.DataFrame):
    """Create a fitted log transformer."""
    from src.features.base import LogTransformer

    transformer = LogTransformer(columns=["prezzo_medio", "popolazione"])
    transformer.fit(sample_features)
    return transformer


@pytest.fixture
def fitted_standard_scaler(sample_features: pd.DataFrame):
    """Create a fitted standard scaler."""
    from src.features.base import StandardScaler

    scaler = StandardScaler(columns=["lat", "long"])
    scaler.fit(sample_features)
    return scaler


# =============================================================================
# INTEGRATION TEST HELPERS
# =============================================================================


@pytest.fixture
def mock_data_files(test_data_dir: Path, sample_omi_data: pd.DataFrame) -> Path:
    """Create mock data files for integration tests."""
    # Write OMI data
    sample_omi_data.to_csv(
        test_data_dir / "raw" / "omi" / "valori.csv",
        sep=";",
        index=False,
    )
    return test_data_dir


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
