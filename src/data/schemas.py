"""Pandera schemas for data validation.

This module defines schemas for all data sources in the pipeline.
Schemas serve as contracts between data loading and feature engineering,
ensuring data quality and catching issues early.

Usage:
    from src.data.schemas import OMIValoriSchema, EnhancedFeaturesSchema

    # Validate DataFrame
    validated_df = OMIValoriSchema.validate(raw_df)

    # Or use as decorator
    @pa.check_types
    def process_data(df: DataFrame[OMIValoriSchema]) -> DataFrame[EnhancedFeaturesSchema]:
        ...
"""

from typing import Optional

import pandera as pa
from pandera.typing import Series


# =============================================================================
# RAW DATA SCHEMAS
# =============================================================================


class OMIValoriSchema(pa.DataFrameModel):
    """Schema for OMI real estate quotations (valori.csv).

    Source: Agenzia delle Entrate - Osservatorio Mercato Immobiliare
    Contains price ranges (min/max) in EUR/sqm for purchase and rental.
    """

    # Geographic identifiers
    Area_territoriale: Series[str] = pa.Field(
        description="Macro-region (NORD-OVEST, NORD-EST, CENTRO, SUD, ISOLE)"
    )
    Regione: Series[str] = pa.Field(description="Region name")
    Prov: Series[str] = pa.Field(str_length=2, description="Province code (2 letters)")
    Comune_ISTAT: Series[str] = pa.Field(
        str_length={"min_value": 6, "max_value": 7},
        description="ISTAT municipality code",
    )
    Comune_cat: Series[str] = pa.Field(description="Cadastral municipality code")
    Comune_amm: Series[str] = pa.Field(description="Administrative municipality code")
    Comune_descrizione: Series[str] = pa.Field(description="Municipality name")

    # Zone classification
    Fascia: Series[str] = pa.Field(
        isin=["B", "C", "D", "E", "R", "S"],
        description="Zone type: B=central, C=semi-central, D=peripheral, E=suburban, R=rural",
    )
    Zona: Series[str] = pa.Field(description="Zone code")
    LinkZona: Series[str] = pa.Field(description="Full zone identifier")

    # Property type
    Cod_Tip: Series[int] = pa.Field(ge=0, description="Property type code")
    Descr_Tipologia: Series[str] = pa.Field(description="Property type description")

    # Condition
    Stato: Series[str] = pa.Field(description="Property condition")
    Stato_prev: Series[str] = pa.Field(nullable=True, description="Previous condition")

    # Purchase prices (EUR/sqm)
    Compr_min: Series[float] = pa.Field(ge=0, nullable=True, description="Min purchase price")
    Compr_max: Series[float] = pa.Field(ge=0, nullable=True, description="Max purchase price")
    Sup_NL_compr: Series[str] = pa.Field(nullable=True, description="Surface type for purchase")

    # Rental prices (EUR/sqm/month)
    Loc_min: Series[float] = pa.Field(ge=0, nullable=True, description="Min rental price")
    Loc_max: Series[float] = pa.Field(ge=0, nullable=True, description="Max rental price")
    Sup_NL_loc: Series[str] = pa.Field(nullable=True, description="Surface type for rental")

    class Config:
        """Schema configuration."""

        name = "OMIValori"
        strict = False  # Allow extra columns (file, Sez)
        coerce = True


class MunicipalityMetadataSchema(pa.DataFrameModel):
    """Schema for municipality metadata from opendatasicilia/comuni-italiani.

    Contains coordinates, administrative codes, and basic info for all Italian municipalities.
    """

    # Identifiers
    pro_com_t: Series[str] = pa.Field(
        str_length=6,
        description="ISTAT municipality code (6 digits, zero-padded)",
    )
    comune: Series[str] = pa.Field(description="Municipality name")

    # Geographic
    lat: Series[float] = pa.Field(ge=35.0, le=48.0, description="Latitude (WGS84)")
    long: Series[float] = pa.Field(ge=6.0, le=19.0, description="Longitude (WGS84)")

    # Administrative hierarchy
    den_prov: Series[str] = pa.Field(description="Province name")
    sigla: Series[str] = pa.Field(str_length=2, description="Province abbreviation")
    den_reg: Series[str] = pa.Field(description="Region name")
    cod_reg: Series[int] = pa.Field(ge=1, le=20, description="Region code")

    class Config:
        """Schema configuration."""

        name = "MunicipalityMetadata"
        strict = False  # Allow extra columns (cap, pec, mail, etc.)
        coerce = True


class IRPEFIncomeSchema(pa.DataFrameModel):
    """Schema for IRPEF income data from Ministry of Finance.

    Contains aggregated tax declaration data by municipality.
    """

    # Identifier - using flexible type since it can come as int or string
    Codice_comune: Series[str] = pa.Field(description="ISTAT municipality code")

    # Location
    Denominazione: Series[str] = pa.Field(description="Municipality name")

    # Income metrics
    Reddito_complessivo: Series[float] = pa.Field(
        ge=0, description="Total declared income (EUR)"
    )
    Numero_contribuenti: Series[int] = pa.Field(
        ge=0, description="Number of taxpayers"
    )

    class Config:
        """Schema configuration."""

        name = "IRPEFIncome"
        strict = False  # Many extra columns in source
        coerce = True


class TourismSchema(pa.DataFrameModel):
    """Schema for tourism statistics.

    Contains tourist arrivals and stays by municipality/province.
    """

    # Identifier
    prov_code: Series[str] = pa.Field(description="Province code")

    # Metrics
    anno: Series[int] = pa.Field(ge=2010, le=2030, description="Year")
    arrivals: Series[float] = pa.Field(ge=0, description="Tourist arrivals")
    nights: Series[float] = pa.Field(ge=0, nullable=True, description="Tourist nights")
    population: Series[float] = pa.Field(ge=0, description="Resident population")
    tourism_intensity: Series[float] = pa.Field(
        ge=0, description="Arrivals per 1000 residents"
    )

    class Config:
        """Schema configuration."""

        name = "Tourism"
        strict = False
        coerce = True


class AirbnbListingsSchema(pa.DataFrameModel):
    """Schema for InsideAirbnb listings data.

    Contains individual Airbnb listing details.
    """

    id: Series[int] = pa.Field(ge=0, description="Listing ID")
    name: Series[str] = pa.Field(nullable=True, description="Listing name")
    latitude: Series[float] = pa.Field(ge=35.0, le=48.0, description="Latitude")
    longitude: Series[float] = pa.Field(ge=6.0, le=19.0, description="Longitude")
    price: Series[float] = pa.Field(ge=0, description="Nightly price (EUR)")
    number_of_reviews: Series[int] = pa.Field(ge=0, description="Total reviews")
    room_type: Series[str] = pa.Field(description="Entire home/apt, Private room, etc.")

    class Config:
        """Schema configuration."""

        name = "AirbnbListings"
        strict = False  # Many extra columns in source
        coerce = True


# =============================================================================
# PROCESSED DATA SCHEMAS
# =============================================================================


class EnhancedFeaturesSchema(pa.DataFrameModel):
    """Schema for the main enhanced features dataset.

    This is the primary dataset used for model training, containing
    all features at the municipality-year level.
    """

    # Primary key
    istat_code: Series[str] = pa.Field(description="ISTAT municipality code")
    anno: Series[int] = pa.Field(ge=2010, le=2030, description="Year")

    # Target variables
    prezzo_medio: Series[float] = pa.Field(
        ge=100, le=30000, description="Average price EUR/sqm"
    )

    # Demographics
    popolazione: Series[float] = pa.Field(ge=0, description="Total population")
    pop_change_pct: Series[float] = pa.Field(
        ge=-50, le=100, nullable=True, description="Population change %"
    )

    # Economics
    avg_income: Series[float] = pa.Field(
        ge=0, nullable=True, description="Average declared income"
    )
    income_change_pct: Series[float] = pa.Field(
        nullable=True, description="Income change %"
    )

    # Geography
    lat: Series[float] = pa.Field(ge=35.0, le=48.0, description="Latitude")
    long: Series[float] = pa.Field(ge=6.0, le=19.0, description="Longitude")
    dist_major_city: Series[float] = pa.Field(ge=0, description="Km to nearest major city")
    dist_coast: Series[float] = pa.Field(ge=0, description="Km to coast")

    # Categorical flags
    coastal: Series[int] = pa.Field(isin=[0, 1], description="Is coastal municipality")
    northern: Series[int] = pa.Field(isin=[0, 1], description="Is in Northern Italy")
    urban: Series[int] = pa.Field(isin=[0, 1], description="Is urban area")

    # Tourism / STR (nullable - not available for all municipalities)
    tourism_intensity: Series[float] = pa.Field(
        ge=0, nullable=True, description="Tourist arrivals per 1000 residents"
    )
    str_density: Series[float] = pa.Field(
        ge=0, nullable=True, description="Short-term rental density"
    )

    class Config:
        """Schema configuration."""

        name = "EnhancedFeatures"
        strict = False  # Allow additional computed columns
        coerce = True


class ModelDataSchema(pa.DataFrameModel):
    """Schema for model-ready dataset.

    Contains log-transformed features ready for regression.
    """

    # Identifier
    istat_code: Series[str] = pa.Field(description="ISTAT municipality code")

    # Log-transformed targets
    log_price_mid: Series[float] = pa.Field(description="log(price) target")

    # Log-transformed features (nullable due to zeros)
    log_population: Series[float] = pa.Field(nullable=True)
    log_income: Series[float] = pa.Field(nullable=True)
    log_str_density: Series[float] = pa.Field(nullable=True)

    # Standardized features
    lat_std: Series[float] = pa.Field(nullable=True)
    lon_std: Series[float] = pa.Field(nullable=True)
    dist_major_city_std: Series[float] = pa.Field(nullable=True)

    class Config:
        """Schema configuration."""

        name = "ModelData"
        strict = False
        coerce = True


class ValuationResultsSchema(pa.DataFrameModel):
    """Schema for valuation/undervaluation results.

    Output of the valuation model showing predicted vs actual prices.
    """

    # Identifier
    istat_code: Series[str] = pa.Field(description="ISTAT municipality code")

    # Actual values
    price_actual: Series[float] = pa.Field(ge=0, description="Actual price EUR/sqm")
    rent_actual: Series[float] = pa.Field(
        ge=0, nullable=True, description="Actual rent EUR/sqm/month"
    )

    # Predictions
    price_predicted: Series[float] = pa.Field(ge=0, description="Model predicted price")
    rent_predicted: Series[float] = pa.Field(
        ge=0, nullable=True, description="Model predicted rent"
    )

    # Residuals / gaps
    price_gap_pct: Series[float] = pa.Field(
        description="Price gap % (negative = undervalued)"
    )
    rent_gap_pct: Series[float] = pa.Field(
        nullable=True, description="Rent gap %"
    )

    # Derived metrics
    gross_yield_pct: Series[float] = pa.Field(
        ge=0, nullable=True, description="Gross rental yield %"
    )
    valuation_score: Series[float] = pa.Field(
        nullable=True, description="Combined valuation score"
    )

    class Config:
        """Schema configuration."""

        name = "ValuationResults"
        strict = False
        coerce = True


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def validate_dataframe(
    df,
    schema: type[pa.DataFrameModel],
    raise_on_error: bool = True,
) -> tuple[bool, Optional[pa.errors.SchemaErrors]]:
    """Validate a DataFrame against a schema.

    Args:
        df: DataFrame to validate.
        schema: Pandera DataFrameModel class.
        raise_on_error: If True, raise on validation failure.

    Returns:
        Tuple of (is_valid, errors). errors is None if valid.

    Raises:
        SchemaErrors: If validation fails and raise_on_error=True.
    """
    try:
        schema.validate(df, lazy=True)
        return True, None
    except pa.errors.SchemaErrors as e:
        if raise_on_error:
            raise
        return False, e


def get_schema_documentation(schema: type[pa.DataFrameModel]) -> dict:
    """Extract documentation from a schema for data dictionary.

    Args:
        schema: Pandera DataFrameModel class.

    Returns:
        Dictionary with column names, types, and descriptions.
    """
    docs = {"name": schema.Config.name if hasattr(schema.Config, "name") else schema.__name__}
    columns = {}

    for name, field in schema.__fields__.items():
        col_info = {
            "dtype": str(field.annotation),
            "nullable": field.nullable if hasattr(field, "nullable") else False,
        }
        if hasattr(field, "description") and field.description:
            col_info["description"] = field.description
        columns[name] = col_info

    docs["columns"] = columns
    return docs
