"""Tests for ensemble models."""

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import GradientBoostingModel, RandomForestModel


class TestGradientBoostingModel:
    """Tests for GradientBoostingModel."""

    def test_fit_predict(self, sample_features: pd.DataFrame):
        """Test basic fit and predict."""
        feature_cols = ["lat", "long", "popolazione"]
        X = sample_features[feature_cols].fillna(0)
        y = np.log(sample_features["prezzo_medio"])

        model = GradientBoostingModel(n_estimators=50, max_depth=3)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert model.is_fitted

    def test_feature_importance(self, trained_gb_model):
        """Test that feature importance sums to 1."""
        importance = trained_gb_model.get_feature_importance()

        assert len(importance) > 0
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_evaluate_metrics(self, sample_features: pd.DataFrame):
        """Test that evaluation returns valid metrics."""
        feature_cols = ["lat", "long", "popolazione"]
        X = sample_features[feature_cols].fillna(0)
        y = np.log(sample_features["prezzo_medio"])

        model = GradientBoostingModel(n_estimators=50, max_depth=3)
        model.fit(X, y)

        result = model.evaluate(X, y)

        assert 0 <= result.r_squared <= 1
        assert result.rmse >= 0
        assert result.mae >= 0

    def test_top_features(self, trained_gb_model):
        """Test get_top_features returns correct format."""
        top = trained_gb_model.get_top_features(n=3)

        assert len(top) <= 3
        assert all(isinstance(t, tuple) for t in top)
        assert all(len(t) == 2 for t in top)

    def test_not_fitted_raises(self, sample_features: pd.DataFrame):
        """Test that predict before fit raises error."""
        model = GradientBoostingModel()
        X = sample_features[["lat", "long"]].fillna(0)

        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(X)


class TestRandomForestModel:
    """Tests for RandomForestModel."""

    def test_fit_predict(self, sample_features: pd.DataFrame):
        """Test basic fit and predict."""
        feature_cols = ["lat", "long", "popolazione"]
        X = sample_features[feature_cols].fillna(0)
        y = np.log(sample_features["prezzo_medio"])

        model = RandomForestModel(n_estimators=20, max_depth=5)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_predict_with_std(self, sample_features: pd.DataFrame):
        """Test prediction with uncertainty estimates."""
        feature_cols = ["lat", "long"]
        X = sample_features[feature_cols].fillna(0)
        y = np.log(sample_features["prezzo_medio"])

        model = RandomForestModel(n_estimators=20)
        model.fit(X, y)

        mean_pred, std_pred = model.predict_with_std(X)

        assert len(mean_pred) == len(X)
        assert len(std_pred) == len(X)
        assert all(std_pred >= 0)  # Std should be non-negative
