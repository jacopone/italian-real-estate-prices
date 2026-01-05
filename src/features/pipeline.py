"""Feature engineering pipeline.

Orchestrates the full feature engineering process, combining multiple
transformers into a cohesive pipeline that produces model-ready features.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.config import Config, FeatureConfig
from src.features.base import FeatureTransformer, LogTransformer, StandardScaler
from src.features.demographic import AgingFeatures, DemographicFeatures
from src.features.economic import EconomicIndicators, IncomeFeatures
from src.features.geographic import CoordinateFeatures, DistanceFeatures, LocationFlags
from src.features.tourism import STRFeatures, STRProxyFeatures, TourismFeatures


@dataclass
class FeaturePipelineResult:
    """Result of feature pipeline execution.

    Attributes:
        features: DataFrame with all computed features.
        feature_names: List of all feature column names.
        transformer_metadata: Metadata from each transformer.
        target_columns: Target variable columns if present.
    """

    features: pd.DataFrame
    feature_names: list[str]
    transformer_metadata: dict[str, Any] = field(default_factory=dict)
    target_columns: list[str] = field(default_factory=list)


class FeaturePipeline:
    """Orchestrate feature engineering from raw data to model-ready features.

    The pipeline combines multiple feature transformers and handles:
    - Fitting transformers on training data
    - Applying consistent transformations to train/test
    - Tracking feature names and metadata
    - Saving/loading fitted pipelines

    Example:
        >>> pipeline = FeaturePipeline(config.features)
        >>> pipeline.fit(train_df)
        >>> train_features = pipeline.transform(train_df)
        >>> test_features = pipeline.transform(test_df)
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        include_targets: bool = True,
    ):
        """Initialize feature pipeline.

        Args:
            config: Feature configuration.
            include_targets: Whether to include target columns in output.
        """
        self.config = config or FeatureConfig()
        self._include_targets = include_targets
        self._transformers: list[FeatureTransformer] = []
        self._is_fitted = False
        self._target_cols = ["prezzo_medio", "affitto_medio"]

    def _build_default_transformers(self) -> list[FeatureTransformer]:
        """Build default transformer list based on config.

        Returns:
            List of configured transformers.
        """
        transformers = []

        # Demographic features
        transformers.append(DemographicFeatures(
            include_log=True,
            name="demographics",
        ))

        # Income features
        transformers.append(IncomeFeatures(
            include_log=True,
            name="income",
        ))

        # Geographic features
        transformers.append(DistanceFeatures(
            include_log=True,
            name="distances",
        ))
        transformers.append(LocationFlags(name="location_flags"))
        transformers.append(CoordinateFeatures(
            standardize=False,
            name="coordinates",
        ))

        # Tourism features
        transformers.append(TourismFeatures(
            include_log=True,
            include_categories=False,
            name="tourism",
        ))

        # STR features (if available)
        if self.config.include_str_proxy:
            transformers.append(STRFeatures(
                include_log=True,
                name="str",
            ))
            transformers.append(STRProxyFeatures(name="str_proxy"))

        return transformers

    def add_transformer(self, transformer: FeatureTransformer) -> "FeaturePipeline":
        """Add a custom transformer to the pipeline.

        Args:
            transformer: Transformer to add.

        Returns:
            Self for chaining.
        """
        self._transformers.append(transformer)
        return self

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit all transformers on training data.

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        # Build default transformers if none added
        if not self._transformers:
            self._transformers = self._build_default_transformers()

        logger.info(f"Fitting pipeline with {len(self._transformers)} transformers")

        for transformer in self._transformers:
            try:
                transformer.fit(df)
                logger.debug(
                    f"  {transformer.name}: {len(transformer.get_feature_names())} features"
                )
            except Exception as e:
                logger.warning(f"  {transformer.name}: failed to fit - {e}")

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> FeaturePipelineResult:
        """Transform data using fitted transformers.

        Args:
            df: DataFrame to transform.

        Returns:
            FeaturePipelineResult with features and metadata.

        Raises:
            ValueError: If pipeline has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError("Pipeline has not been fitted. Call fit() first.")

        all_features = []
        all_names = []
        metadata = {}

        # Apply each transformer
        for transformer in self._transformers:
            if not transformer.is_fitted:
                continue

            try:
                features = transformer.transform(df)
                if not features.empty:
                    all_features.append(features)
                    all_names.extend(transformer.get_feature_names())
                    metadata[transformer.name] = transformer.metadata
            except Exception as e:
                logger.warning(f"{transformer.name}: transform failed - {e}")

        # Combine all features
        if all_features:
            combined = pd.concat(all_features, axis=1)
        else:
            combined = pd.DataFrame(index=df.index)

        # Add identifier columns
        id_cols = ["istat_code", "anno"]
        for col in id_cols:
            if col in df.columns:
                combined[col] = df[col].values

        # Add target columns if requested
        target_cols = []
        if self._include_targets:
            for col in self._target_cols:
                if col in df.columns:
                    combined[col] = df[col].values
                    target_cols.append(col)

        logger.info(
            f"Pipeline produced {len(all_names)} features from "
            f"{sum(1 for t in self._transformers if t.is_fitted)} transformers"
        )

        return FeaturePipelineResult(
            features=combined,
            feature_names=all_names,
            transformer_metadata=metadata,
            target_columns=target_cols,
        )

    def fit_transform(self, df: pd.DataFrame) -> FeaturePipelineResult:
        """Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform.

        Returns:
            FeaturePipelineResult.
        """
        return self.fit(df).transform(df)

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names.

        Returns:
            Combined list of feature names from all transformers.
        """
        if not self._is_fitted:
            return []

        names = []
        for transformer in self._transformers:
            if transformer.is_fitted:
                names.extend(transformer.get_feature_names())
        return names

    @property
    def is_fitted(self) -> bool:
        """Check if pipeline is fitted."""
        return self._is_fitted


class ModelReadyPipeline:
    """High-level pipeline that produces model-ready features.

    This is the main entry point for feature engineering, handling:
    - Loading data from multiple sources
    - Merging datasets
    - Running feature transformations
    - Producing train/test ready DataFrames

    Example:
        >>> pipeline = ModelReadyPipeline(config)
        >>> train_features, test_features = pipeline.prepare_for_training(
        ...     price_df, demographics_df, income_df, tourism_df
        ... )
    """

    def __init__(self, config: Config | None = None):
        """Initialize model-ready pipeline.

        Args:
            config: Full configuration object.
        """
        self.config = config or Config()
        self._feature_pipeline = FeaturePipeline(config=self.config.features)

    def merge_data_sources(
        self,
        price_df: pd.DataFrame,
        demographics_df: pd.DataFrame | None = None,
        income_df: pd.DataFrame | None = None,
        tourism_df: pd.DataFrame | None = None,
        airbnb_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Merge all data sources into unified DataFrame.

        Args:
            price_df: OMI price data (required).
            demographics_df: ISTAT demographics.
            income_df: IRPEF income data.
            tourism_df: Tourism statistics.
            airbnb_df: Airbnb/STR data.

        Returns:
            Merged DataFrame with all available data.
        """
        # Start with price data
        merged = price_df.copy()
        logger.info(f"Starting merge with {len(merged):,} price records")

        # Merge demographics
        if demographics_df is not None and not demographics_df.empty:
            merge_cols = ["istat_code"]
            demo_cols = [c for c in demographics_df.columns if c not in merged.columns or c in merge_cols]
            merged = merged.merge(
                demographics_df[demo_cols],
                on=merge_cols,
                how="left",
            )
            logger.info(f"  + demographics: {len(merged):,} records")

        # Merge income
        if income_df is not None and not income_df.empty:
            income_cols = ["istat_code", "avg_income", "income_change_pct"]
            income_cols = [c for c in income_cols if c in income_df.columns]
            if income_cols:
                merged = merged.merge(
                    income_df[income_cols].drop_duplicates("istat_code"),
                    on="istat_code",
                    how="left",
                )
                logger.info(f"  + income: {len(merged):,} records")

        # Merge tourism (at province level)
        if tourism_df is not None and not tourism_df.empty:
            if "prov_code" in tourism_df.columns:
                # Add prov_code to merged if not present
                if "prov_code" not in merged.columns and "istat_code" in merged.columns:
                    merged["prov_code"] = merged["istat_code"].str[:3]

                tourism_cols = ["prov_code", "tourism_intensity"]
                tourism_cols = [c for c in tourism_cols if c in tourism_df.columns]
                if "prov_code" in merged.columns:
                    merged = merged.merge(
                        tourism_df[tourism_cols].drop_duplicates("prov_code"),
                        on="prov_code",
                        how="left",
                    )
                    logger.info(f"  + tourism: {len(merged):,} records")

        # Merge Airbnb
        if airbnb_df is not None and not airbnb_df.empty:
            airbnb_cols = ["istat_code", "str_density", "airbnb_price_median", "str_premium"]
            airbnb_cols = [c for c in airbnb_cols if c in airbnb_df.columns]
            if airbnb_cols:
                merged = merged.merge(
                    airbnb_df[airbnb_cols].drop_duplicates("istat_code"),
                    on="istat_code",
                    how="left",
                )
                logger.info(f"  + airbnb: {len(merged):,} records")

        logger.info(f"Merged dataset: {len(merged):,} rows, {len(merged.columns)} columns")
        return merged

    def prepare_features(
        self,
        merged_df: pd.DataFrame,
        fit: bool = True,
    ) -> FeaturePipelineResult:
        """Run feature engineering pipeline.

        Args:
            merged_df: Merged data from all sources.
            fit: Whether to fit the pipeline (True for training).

        Returns:
            FeaturePipelineResult with model-ready features.
        """
        if fit:
            return self._feature_pipeline.fit_transform(merged_df)
        else:
            return self._feature_pipeline.transform(merged_df)

    def get_model_columns(self) -> list[str]:
        """Get list of feature columns for modeling.

        Returns:
            List of feature column names.
        """
        return self._feature_pipeline.get_feature_names()


def create_features(
    data_dir: Path,
    config: Config | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """Convenience function to create features from raw data.

    Args:
        data_dir: Root data directory.
        config: Optional configuration.
        year: Specific year to process.

    Returns:
        DataFrame with all features.
    """
    from src.data.processors import (
        AirbnbProcessor,
        IRPEFProcessor,
        ISTATProcessor,
        OMIProcessor,
        TourismProcessor,
    )

    config = config or Config()

    # Load data from each source
    logger.info("Loading data sources...")
    omi = OMIProcessor(config.omi).load_and_process(data_dir)
    istat = ISTATProcessor().load_and_process(data_dir)
    irpef = IRPEFProcessor().load_and_process(data_dir)
    tourism = TourismProcessor().load_and_process(data_dir)
    airbnb = AirbnbProcessor(config.airbnb).load_and_process(data_dir)

    # Filter to year if specified
    if year and "anno" in omi.columns:
        omi = omi[omi["anno"] == year]

    # Create pipeline and process
    pipeline = ModelReadyPipeline(config)
    merged = pipeline.merge_data_sources(
        price_df=omi,
        demographics_df=istat,
        income_df=irpef,
        tourism_df=tourism,
        airbnb_df=airbnb,
    )

    result = pipeline.prepare_features(merged, fit=True)
    return result.features
