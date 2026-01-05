"""Tests for configuration module."""

import pytest

from src.config import Config, FeatureConfig, ModelConfig, OMIConfig, load_config


class TestConfig:
    """Tests for the Config class."""

    def test_default_config(self):
        """Test that default config loads without errors."""
        config = Config()
        assert config.analysis_year == 2023
        assert config.paths.data_dir.name == "data"

    def test_omi_config_defaults(self):
        """Test OMI configuration defaults."""
        config = OMIConfig()
        assert "Abitazioni civili" in config.property_types
        assert config.min_year >= 2004
        assert config.max_year <= 2030

    def test_model_config_defaults(self):
        """Test model configuration defaults."""
        config = ModelConfig()
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert "n_estimators" in config.gb_params

    def test_feature_config_defaults(self):
        """Test feature configuration defaults."""
        config = FeatureConfig()
        assert config.include_str_proxy is True
        assert len(config.log_transform_columns) > 0

    def test_config_validation(self):
        """Test that invalid config values are rejected."""
        with pytest.raises(ValueError):
            ModelConfig(test_size=1.5)  # Must be <= 0.5

    def test_load_config_nonexistent(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_config_none(self):
        """Test loading with None returns defaults."""
        config = load_config(None)
        assert isinstance(config, Config)
