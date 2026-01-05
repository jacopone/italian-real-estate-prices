"""Tests for base feature transformers."""

import numpy as np
import pandas as pd
import pytest

from src.features.base import (
    IdentityTransformer,
    LogTransformer,
    StandardScaler,
)


class TestLogTransformer:
    """Tests for LogTransformer."""

    def test_fit_transform(self, sample_features: pd.DataFrame):
        """Test fit_transform produces correct output."""
        transformer = LogTransformer(columns=["prezzo_medio", "popolazione"])
        result = transformer.fit_transform(sample_features)

        assert "log_prezzo_medio" in result.columns
        assert "log_popolazione" in result.columns
        assert len(result) == len(sample_features)

    def test_log_values_correct(self, sample_features: pd.DataFrame):
        """Test that log values are mathematically correct."""
        transformer = LogTransformer(columns=["prezzo_medio"])
        result = transformer.fit_transform(sample_features)

        expected = np.log1p(sample_features["prezzo_medio"])
        np.testing.assert_array_almost_equal(
            result["log_prezzo_medio"].values,
            expected.values,
        )

    def test_not_fitted_raises(self, sample_features: pd.DataFrame):
        """Test that transform before fit raises error."""
        transformer = LogTransformer(columns=["prezzo_medio"])
        with pytest.raises(ValueError, match="not been fitted"):
            transformer.transform(sample_features)

    def test_missing_column_raises(self, sample_features: pd.DataFrame):
        """Test that missing column raises error."""
        transformer = LogTransformer(columns=["nonexistent"])
        with pytest.raises(KeyError):
            transformer.fit(sample_features)


class TestStandardScaler:
    """Tests for StandardScaler."""

    def test_fit_transform(self, sample_features: pd.DataFrame):
        """Test fit_transform produces standardized output."""
        scaler = StandardScaler(columns=["lat", "long"])
        result = scaler.fit_transform(sample_features)

        assert "lat_std" in result.columns
        assert "long_std" in result.columns

    def test_zero_mean_unit_variance(self, sample_features: pd.DataFrame):
        """Test that output has approximately zero mean and unit variance."""
        scaler = StandardScaler(columns=["lat"])
        result = scaler.fit_transform(sample_features)

        # Mean should be close to 0
        assert abs(result["lat_std"].mean()) < 0.01
        # Std should be close to 1
        assert abs(result["lat_std"].std() - 1.0) < 0.01

    def test_inverse_transform(self, sample_features: pd.DataFrame):
        """Test that inverse transform recovers original values."""
        scaler = StandardScaler(columns=["lat", "long"])
        scaled = scaler.fit_transform(sample_features)

        recovered = scaler.inverse_transform(scaled)

        np.testing.assert_array_almost_equal(
            recovered["lat"].values,
            sample_features["lat"].values,
        )


class TestIdentityTransformer:
    """Tests for IdentityTransformer."""

    def test_passes_columns_unchanged(self, sample_features: pd.DataFrame):
        """Test that columns are passed through unchanged."""
        transformer = IdentityTransformer(columns=["lat", "long"])
        result = transformer.fit_transform(sample_features)

        pd.testing.assert_series_equal(result["lat"], sample_features["lat"])
        pd.testing.assert_series_equal(result["long"], sample_features["long"])

    def test_only_specified_columns(self, sample_features: pd.DataFrame):
        """Test that only specified columns are returned."""
        transformer = IdentityTransformer(columns=["lat"])
        result = transformer.fit_transform(sample_features)

        assert list(result.columns) == ["lat"]
