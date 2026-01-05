"""Time series analysis for demographic-price relationships.

Implements:
- Cross-correlation analysis with lags
- Distributed Lag Models (DLM)
- Granger causality tests
- Panel data models with fixed effects
"""

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def cross_correlation(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 10,
    direction: Literal["both", "positive", "negative"] = "both",
) -> pd.DataFrame:
    """Compute cross-correlation between two time series at different lags.

    Args:
        x: First series (e.g., population change)
        y: Second series (e.g., price change)
        max_lag: Maximum lag to test
        direction: 'positive' (x leads y), 'negative' (y leads x), or 'both'

    Returns:
        DataFrame with lag, correlation, and p-value
    """
    results = []

    if direction in ("both", "negative"):
        for lag in range(-max_lag, 0):
            # Negative lag: y leads x
            x_aligned = x.iloc[-lag:]
            y_aligned = y.iloc[:lag]

            if len(x_aligned) > 10 and len(y_aligned) > 10:
                min_len = min(len(x_aligned), len(y_aligned))
                corr, pval = stats.pearsonr(
                    x_aligned.iloc[:min_len].dropna(),
                    y_aligned.iloc[:min_len].dropna()
                )
                results.append({"lag": lag, "correlation": corr, "p_value": pval})

    if direction in ("both", "positive"):
        for lag in range(0, max_lag + 1):
            # Positive lag: x leads y
            if lag == 0:
                x_aligned = x
                y_aligned = y
            else:
                x_aligned = x.iloc[:-lag]
                y_aligned = y.iloc[lag:]

            if len(x_aligned) > 10 and len(y_aligned) > 10:
                min_len = min(len(x_aligned), len(y_aligned))
                valid_x = x_aligned.iloc[:min_len].dropna()
                valid_y = y_aligned.iloc[:min_len].dropna()

                # Align indices
                common_idx = valid_x.index.intersection(valid_y.index)
                if len(common_idx) > 10:
                    corr, pval = stats.pearsonr(
                        valid_x.loc[common_idx],
                        valid_y.loc[common_idx]
                    )
                    results.append({"lag": lag, "correlation": corr, "p_value": pval})

    return pd.DataFrame(results).sort_values("lag")


def find_optimal_lag(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 10,
) -> dict:
    """Find the lag with maximum correlation.

    Args:
        x: Predictor series (e.g., population change)
        y: Response series (e.g., price change)
        max_lag: Maximum lag to test

    Returns:
        Dictionary with optimal_lag, correlation, p_value, all_results
    """
    cc = cross_correlation(x, y, max_lag=max_lag, direction="positive")

    if cc.empty:
        return {"optimal_lag": None, "correlation": None, "p_value": None}

    # Find lag with highest absolute correlation
    best_idx = cc["correlation"].abs().idxmax()
    best_row = cc.loc[best_idx]

    return {
        "optimal_lag": int(best_row["lag"]),
        "correlation": best_row["correlation"],
        "p_value": best_row["p_value"],
        "all_results": cc,
    }


def panel_cross_correlation(
    df: pd.DataFrame,
    entity_col: str,
    time_col: str,
    x_col: str,
    y_col: str,
    max_lag: int = 10,
) -> pd.DataFrame:
    """Compute cross-correlation across panel data.

    For each entity, computes correlation at each lag, then averages.

    Args:
        df: Panel DataFrame
        entity_col: Column identifying entities (e.g., 'municipality')
        time_col: Column identifying time periods
        x_col: Predictor column
        y_col: Response column
        max_lag: Maximum lag to test

    Returns:
        DataFrame with lag, mean_correlation, std_correlation, n_entities
    """
    entity_results = []

    for entity in df[entity_col].unique():
        entity_data = df[df[entity_col] == entity].sort_values(time_col)

        if len(entity_data) < max_lag + 5:
            continue

        x = entity_data[x_col]
        y = entity_data[y_col]

        for lag in range(0, max_lag + 1):
            if lag == 0:
                x_aligned = x.values
                y_aligned = y.values
            else:
                x_aligned = x.values[:-lag]
                y_aligned = y.values[lag:]

            if len(x_aligned) > 3:
                # Remove NaNs
                valid = ~(np.isnan(x_aligned) | np.isnan(y_aligned))
                if valid.sum() > 3:
                    corr = np.corrcoef(x_aligned[valid], y_aligned[valid])[0, 1]
                    if not np.isnan(corr):
                        entity_results.append({
                            "entity": entity,
                            "lag": lag,
                            "correlation": corr,
                        })

    if not entity_results:
        return pd.DataFrame()

    results_df = pd.DataFrame(entity_results)

    # Aggregate by lag
    summary = results_df.groupby("lag").agg({
        "correlation": ["mean", "std", "count"],
    }).round(4)
    summary.columns = ["mean_correlation", "std_correlation", "n_entities"]
    summary = summary.reset_index()

    # Add significance test (t-test against 0)
    for idx, row in summary.iterrows():
        lag_corrs = results_df[results_df["lag"] == row["lag"]]["correlation"]
        if len(lag_corrs) > 5:
            t_stat, p_val = stats.ttest_1samp(lag_corrs, 0)
            summary.loc[idx, "t_statistic"] = t_stat
            summary.loc[idx, "p_value"] = p_val

    return summary


def distributed_lag_regression(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    max_lag: int = 5,
    entity_col: str | None = None,
) -> dict:
    """Estimate a distributed lag model.

    Model: y_t = α + Σ(β_k * x_{t-k}) + ε_t

    Args:
        df: DataFrame with time series data
        y_col: Dependent variable column
        x_col: Independent variable column
        max_lag: Number of lags to include
        entity_col: If provided, include entity fixed effects

    Returns:
        Dictionary with coefficients, std_errors, cumulative_effect
    """
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        logger.error("sklearn required for regression")
        return {}

    df = df.copy().dropna(subset=[y_col, x_col])

    # Create lagged variables
    for lag in range(0, max_lag + 1):
        df[f"{x_col}_lag{lag}"] = df.groupby(entity_col)[x_col].shift(lag) if entity_col else df[x_col].shift(lag)

    # Drop rows with NaN lags
    lag_cols = [f"{x_col}_lag{lag}" for lag in range(0, max_lag + 1)]
    df = df.dropna(subset=lag_cols)

    if len(df) < max_lag + 10:
        logger.warning("Insufficient data for distributed lag regression")
        return {}

    # Prepare X and y
    X = df[lag_cols].values
    y = df[y_col].values

    # Add entity dummies if specified
    if entity_col and df[entity_col].nunique() < 100:
        entity_dummies = pd.get_dummies(df[entity_col], prefix="entity", drop_first=True)
        X = np.hstack([X, entity_dummies.values])

    # Fit regression
    model = LinearRegression()
    model.fit(X, y)

    # Extract coefficients for lags only
    lag_coefs = model.coef_[:max_lag + 1]

    # Bootstrap standard errors
    n_bootstrap = 100
    bootstrap_coefs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y), size=len(y), replace=True)
        model_boot = LinearRegression()
        model_boot.fit(X[idx], y[idx])
        bootstrap_coefs.append(model_boot.coef_[:max_lag + 1])

    std_errors = np.std(bootstrap_coefs, axis=0)

    return {
        "lag_coefficients": dict(zip(range(max_lag + 1), lag_coefs)),
        "std_errors": dict(zip(range(max_lag + 1), std_errors)),
        "cumulative_effect": sum(lag_coefs),
        "r_squared": model.score(X, y),
        "n_observations": len(y),
    }


def granger_causality_test(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 4,
) -> dict:
    """Test if x Granger-causes y.

    Null hypothesis: x does NOT Granger-cause y

    Args:
        x: Potential cause series
        y: Potential effect series
        max_lag: Number of lags for the test

    Returns:
        Dictionary with f_statistic, p_value, conclusion
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        logger.warning("statsmodels required for Granger causality test")
        return {"error": "statsmodels not installed"}

    # Prepare data
    data = pd.DataFrame({"y": y, "x": x}).dropna()

    if len(data) < max_lag * 3:
        return {"error": "Insufficient data"}

    try:
        results = grangercausalitytests(data[["y", "x"]], maxlag=max_lag, verbose=False)

        # Extract results for each lag
        lag_results = {}
        for lag in range(1, max_lag + 1):
            f_stat = results[lag][0]["ssr_ftest"][0]
            p_value = results[lag][0]["ssr_ftest"][1]
            lag_results[lag] = {"f_statistic": f_stat, "p_value": p_value}

        # Best lag (lowest p-value)
        best_lag = min(lag_results, key=lambda k: lag_results[k]["p_value"])
        best_result = lag_results[best_lag]

        return {
            "best_lag": best_lag,
            "f_statistic": best_result["f_statistic"],
            "p_value": best_result["p_value"],
            "rejects_null": best_result["p_value"] < 0.05,
            "conclusion": "x Granger-causes y" if best_result["p_value"] < 0.05 else "No Granger causality",
            "all_lags": lag_results,
        }

    except Exception as e:
        return {"error": str(e)}


def run_full_analysis(
    price_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    merge_col: str = "istat_code",
    price_col: str = "prezzo_medio",
    pop_col: str = "popolazione",
    time_col: str = "anno",
    max_lag: int = 10,
) -> dict:
    """Run complete time series analysis.

    Args:
        price_df: DataFrame with price data
        demo_df: DataFrame with demographic data
        merge_col: Column to merge on (municipality code)
        price_col: Price column
        pop_col: Population column
        time_col: Time period column
        max_lag: Maximum lag to test

    Returns:
        Dictionary with all analysis results
    """
    logger.info("Running full time series analysis...")

    # Merge datasets
    merged = price_df.merge(demo_df, on=[merge_col, time_col], how="inner")
    logger.info(f"Merged data: {len(merged):,} observations")

    # Calculate changes
    merged = merged.sort_values([merge_col, time_col])
    merged["price_change"] = merged.groupby(merge_col)[price_col].pct_change() * 100
    merged["pop_change"] = merged.groupby(merge_col)[pop_col].pct_change() * 100

    results = {}

    # 1. Panel cross-correlation
    logger.info("Computing panel cross-correlation...")
    results["cross_correlation"] = panel_cross_correlation(
        merged, merge_col, time_col, "pop_change", "price_change", max_lag
    )

    # 2. Find optimal lag
    logger.info("Finding optimal lag...")
    agg = merged.groupby(time_col).agg({
        "pop_change": "mean",
        "price_change": "mean",
    }).dropna()

    results["optimal_lag"] = find_optimal_lag(
        agg["pop_change"], agg["price_change"], max_lag
    )

    # 3. Distributed lag model
    logger.info("Estimating distributed lag model...")
    results["distributed_lag"] = distributed_lag_regression(
        merged, "price_change", "pop_change", max_lag=min(max_lag, 5), entity_col=merge_col
    )

    # 4. Granger causality
    logger.info("Testing Granger causality...")
    results["granger_causality"] = granger_causality_test(
        agg["pop_change"], agg["price_change"], max_lag=min(max_lag, 4)
    )

    logger.info("Analysis complete!")
    return results
