"""Base classes for feature transformers.

This module defines the abstract interface for feature transformers,
following the scikit-learn transformer pattern (fit/transform).

All feature transformers inherit from FeatureTransformer and implement:
- fit(): Learn parameters from training data
- transform(): Apply transformation to data
- get_feature_names(): Return list of output feature names
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class TransformerMetadata:
    """Metadata about a fitted transformer.

    Attributes:
        name: Transformer name.
        input_features: List of input column names.
        output_features: List of output column names.
        parameters: Learned parameters (e.g., means, stds for normalization).
        is_fitted: Whether the transformer has been fitted.
    """

    name: str
    input_features: list[str] = field(default_factory=list)
    output_features: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    is_fitted: bool = False


class FeatureTransformer(ABC):
    """Abstract base class for feature transformers.

    Transformers follow the scikit-learn pattern:
    1. fit(df) - Learn parameters from data
    2. transform(df) - Apply transformation
    3. fit_transform(df) - Convenience method

    Example:
        >>> transformer = DemographicFeatures()
        >>> transformer.fit(train_df)
        >>> train_features = transformer.transform(train_df)
        >>> test_features = transformer.transform(test_df)
    """

    def __init__(self, name: str | None = None):
        """Initialize transformer.

        Args:
            name: Optional name for the transformer.
        """
        self._name = name or self.__class__.__name__
        self._metadata = TransformerMetadata(name=self._name)

    @property
    def name(self) -> str:
        """Get transformer name."""
        return self._name

    @property
    def is_fitted(self) -> bool:
        """Check if transformer has been fitted."""
        return self._metadata.is_fitted

    @property
    def metadata(self) -> TransformerMetadata:
        """Get transformer metadata."""
        return self._metadata

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        """Learn parameters from training data.

        Args:
            df: Training DataFrame.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to data.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame with new features.

        Raises:
            ValueError: If transformer has not been fitted.
        """
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Get list of output feature names.

        Returns:
            List of feature column names produced by this transformer.
        """
        pass

    def _check_fitted(self) -> None:
        """Check if transformer is fitted, raise if not."""
        if not self.is_fitted:
            raise ValueError(
                f"{self.name} has not been fitted. "
                "Call fit() or fit_transform() first."
            )

    def _check_columns(self, df: pd.DataFrame, required: list[str]) -> None:
        """Check that required columns exist in DataFrame.

        Args:
            df: DataFrame to check.
            required: List of required column names.

        Raises:
            KeyError: If any required column is missing.
        """
        missing = set(required) - set(df.columns)
        if missing:
            raise KeyError(
                f"{self.name} requires columns {missing} which are not in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )


class IdentityTransformer(FeatureTransformer):
    """Transformer that passes through specified columns unchanged.

    Useful for including raw columns in feature pipelines.

    Example:
        >>> transformer = IdentityTransformer(columns=['lat', 'lon'])
        >>> features = transformer.fit_transform(df)
    """

    def __init__(self, columns: list[str], name: str | None = None):
        """Initialize with columns to pass through.

        Args:
            columns: Column names to include.
            name: Optional transformer name.
        """
        super().__init__(name or "IdentityTransformer")
        self._columns = columns

    def fit(self, df: pd.DataFrame) -> "IdentityTransformer":
        """Verify columns exist.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        self._check_columns(df, self._columns)
        self._metadata.input_features = self._columns
        self._metadata.output_features = self._columns
        self._metadata.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return specified columns.

        Args:
            df: DataFrame to transform.

        Returns:
            DataFrame with only the specified columns.
        """
        self._check_fitted()
        self._check_columns(df, self._columns)
        return df[self._columns].copy()

    def get_feature_names(self) -> list[str]:
        """Get column names."""
        return self._columns.copy()


class LogTransformer(FeatureTransformer):
    """Apply log1p transformation to specified columns.

    Uses log1p (log(1+x)) for zero-safe transformation.
    Handles negative values by taking absolute value and preserving sign.

    Example:
        >>> transformer = LogTransformer(columns=['price', 'population'])
        >>> features = transformer.fit_transform(df)
        >>> # Creates: log_price, log_population
    """

    def __init__(
        self,
        columns: list[str],
        prefix: str = "log_",
        name: str | None = None,
    ):
        """Initialize with columns to transform.

        Args:
            columns: Column names to log-transform.
            prefix: Prefix for output column names.
            name: Optional transformer name.
        """
        super().__init__(name or "LogTransformer")
        self._columns = columns
        self._prefix = prefix

    def fit(self, df: pd.DataFrame) -> "LogTransformer":
        """Verify columns exist.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        self._check_columns(df, self._columns)
        self._metadata.input_features = self._columns
        self._metadata.output_features = [
            f"{self._prefix}{col}" for col in self._columns
        ]
        self._metadata.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p transformation.

        Args:
            df: DataFrame to transform.

        Returns:
            DataFrame with log-transformed columns.
        """
        import numpy as np

        self._check_fitted()
        self._check_columns(df, self._columns)

        result = pd.DataFrame(index=df.index)
        for col in self._columns:
            out_col = f"{self._prefix}{col}"
            # Handle potential negative values
            values = df[col].fillna(0)
            result[out_col] = np.sign(values) * np.log1p(np.abs(values))

        return result

    def get_feature_names(self) -> list[str]:
        """Get output column names."""
        return [f"{self._prefix}{col}" for col in self._columns]


class StandardScaler(FeatureTransformer):
    """Standardize features by removing mean and scaling to unit variance.

    z = (x - mean) / std

    Example:
        >>> scaler = StandardScaler(columns=['income', 'distance'])
        >>> scaler.fit(train_df)
        >>> scaled_train = scaler.transform(train_df)
        >>> scaled_test = scaler.transform(test_df)
    """

    def __init__(
        self,
        columns: list[str],
        suffix: str = "_std",
        name: str | None = None,
    ):
        """Initialize with columns to standardize.

        Args:
            columns: Column names to standardize.
            suffix: Suffix for output column names.
            name: Optional transformer name.
        """
        super().__init__(name or "StandardScaler")
        self._columns = columns
        self._suffix = suffix
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "StandardScaler":
        """Learn mean and std from training data.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        self._check_columns(df, self._columns)

        for col in self._columns:
            self._means[col] = df[col].mean()
            self._stds[col] = df[col].std()
            # Avoid division by zero
            if self._stds[col] == 0:
                self._stds[col] = 1.0

        self._metadata.input_features = self._columns
        self._metadata.output_features = [
            f"{col}{self._suffix}" for col in self._columns
        ]
        self._metadata.parameters = {
            "means": self._means.copy(),
            "stds": self._stds.copy(),
        }
        self._metadata.is_fitted = True

        logger.debug(f"StandardScaler fitted on {len(self._columns)} columns")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standardization.

        Args:
            df: DataFrame to transform.

        Returns:
            DataFrame with standardized columns.
        """
        self._check_fitted()
        self._check_columns(df, self._columns)

        result = pd.DataFrame(index=df.index)
        for col in self._columns:
            out_col = f"{col}{self._suffix}"
            result[out_col] = (df[col] - self._means[col]) / self._stds[col]

        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse standardization.

        Args:
            df: Standardized DataFrame.

        Returns:
            DataFrame with original scale.
        """
        self._check_fitted()

        result = pd.DataFrame(index=df.index)
        for col in self._columns:
            std_col = f"{col}{self._suffix}"
            if std_col in df.columns:
                result[col] = df[std_col] * self._stds[col] + self._means[col]

        return result

    def get_feature_names(self) -> list[str]:
        """Get output column names."""
        return [f"{col}{self._suffix}" for col in self._columns]
