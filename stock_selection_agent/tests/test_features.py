"""
test_features.py

Tests for engineered stock features.

Run:
    python -m pytest tests/test_features.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_features():

    assert FEATURE_FILE.exists(), (
        f"Feature file not found: {FEATURE_FILE}"
    )

    df = pd.read_parquet(FEATURE_FILE)

    return df


# ============================================================
# TEST 1: FILE EXISTS
# ============================================================

def test_feature_file_exists():

    assert FEATURE_FILE.exists(), (
        f"Expected feature file does not exist: {FEATURE_FILE}"
    )


# ============================================================
# TEST 2: DATASET IS NOT EMPTY
# ============================================================

def test_feature_dataset_not_empty():

    df = load_features()

    assert len(df) > 0, (
        "features.parquet is empty."
    )


# ============================================================
# TEST 3: REQUIRED BASE COLUMNS EXIST
# ============================================================

def test_base_columns_exist():

    df = load_features()

    required_columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    assert not missing, (
        f"Missing base columns: {missing}"
    )


# ============================================================
# TEST 4: RETURN FEATURES EXIST
# ============================================================

def test_return_features_exist():

    df = load_features()

    expected = [
        "daily_return",
        "return_5d",
        "return_21d",
        "return_63d",
        "return_126d",
        "return_252d",
    ]

    missing = [
        col
        for col in expected
        if col not in df.columns
    ]

    assert not missing, (
        f"Missing return features: {missing}"
    )


# ============================================================
# TEST 5: PRICE-BEHAVIOR FEATURES EXIST
# ============================================================

def test_price_behavior_features_exist():

    df = load_features()

    expected = [
        "high_low_range",
        "open_close_return",
        "gap_return",
    ]

    missing = [
        col
        for col in expected
        if col not in df.columns
    ]

    assert not missing, (
        f"Missing price behavior features: {missing}"
    )


# ============================================================
# TEST 6: MOVING AVERAGE FEATURES EXIST
# ============================================================

def test_moving_average_features_exist():

    df = load_features()

    expected = [
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",
        "ema_12",
        "ema_26",
        "price_to_sma20",
        "price_to_sma50",
        "price_to_sma200",
    ]

    missing = [
        col
        for col in expected
        if col not in df.columns
    ]

    assert not missing, (
        f"Missing moving-average features: {missing}"
    )


# ============================================================
# TEST 7: RSI FEATURE EXISTS
# ============================================================

def test_rsi_feature_exists():

    df = load_features()

    assert "rsi_14" in df.columns, (
        "Missing RSI feature: rsi_14"
    )


# ============================================================
# TEST 8: RSI VALID RANGE
# ============================================================

def test_rsi_valid_range():

    df = load_features()

    rsi = df["rsi_14"].dropna()

    assert not rsi.empty, (
        "RSI column contains no valid observations."
    )

    invalid = (
        (rsi < 0)
        |
        (rsi > 100)
    ).sum()

    assert invalid == 0, (
        f"Found {invalid} RSI values outside 0-100."
    )


# ============================================================
# TEST 9: MACD FEATURES EXIST
# ============================================================

def test_macd_features_exist():

    df = load_features()

    expected = [
        "macd",
        "macd_signal",
        "macd_histogram",
    ]

    missing = [
        col
        for col in expected
        if col not in df.columns
    ]

    assert not missing, (
        f"Missing MACD features: {missing}"
    )


# ============================================================
# TEST 10: VOLATILITY FEATURES EXIST
# ============================================================

def test_volatility_features_exist():

    df = load_features()

    expected = [
        "volatility_20d",
        "volatility_60d",
        "volatility_252d",
    ]

    missing = [
        col
        for col in expected
        if col not in df.columns
    ]

    assert not missing, (
        f"Missing volatility features: {missing}"
    )


# ============================================================
# TEST 11: VOLATILITY IS NON-NEGATIVE
# ============================================================

def test_volatility_non_negative():

    df = load_features()

    columns = [
        "volatility_20d",
        "volatility_60d",
        "volatility_252d",
    ]

    for column in columns:

        values = (
            df[column]
            .dropna()
        )

        invalid = (
            values < 0
        ).sum()

        assert invalid == 0, (
            f"Negative volatility values found in {column}: "
            f"{invalid}"
        )


# ============================================================
# TEST 12: SHARPE FEATURE EXISTS
# ============================================================

def test_sharpe_feature_exists():

    df = load_features()

    assert "sharpe_252d" in df.columns, (
        "Missing Sharpe feature: sharpe_252d"
    )


# ============================================================
# TEST 13: ROLLING HIGH/LOW FEATURES EXIST
# ============================================================

def test_rolling_high_low_features_exist():

    df = load_features()

    expected = [
    "rolling_52w_high",
    "rolling_52w_low",
    "drawdown",
    "distance_52w_high",
    "distance_52w_low",
    "max_drawdown_252d",
    ]

    missing = [
        col
        for col in expected
        if col not in df.columns
    ]

    assert not missing, (
        f"Missing rolling high/low features: {missing}"
    )


# ============================================================
# TEST 14: ROLLING HIGH >= ROLLING LOW
# ============================================================

def test_rolling_high_greater_than_low():

    df = load_features()

    subset = df[
        [
            "rolling_52w_high",
            "rolling_52w_low",
        ]
    ].dropna()

    invalid = (
        subset["rolling_52w_high"]
        <
        subset["rolling_52w_low"]
    ).sum()

    assert invalid == 0, (
        f"Found {invalid} rows where "
        "rolling_52w_high < rolling_52w_low."
    )


# ============================================================
# TEST 15: VOLUME FEATURES EXIST
# ============================================================

def test_volume_features_exist():

    df = load_features()

    expected = [
        "volume_change",
        "avg_volume_20d",
        "volume_ratio",
    ]

    missing = [
        col
        for col in expected
        if col not in df.columns
    ]

    assert not missing, (
        f"Missing volume features: {missing}"
    )


# ============================================================
# TEST 16: NO INFINITE VALUES
# ============================================================

def test_no_infinite_values():

    df = load_features()

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    infinite_count = np.isinf(
        numeric_df.to_numpy()
    ).sum()

    assert infinite_count == 0, (
        f"Found {infinite_count} infinite feature values."
    )


# ============================================================
# TEST 17: EXPECTED WARM-UP NAN VALUES EXIST
# ============================================================

def test_expected_warmup_nans_exist():

    df = load_features()

    expected_nan_columns = [
        "daily_return",
        "return_5d",
        "return_21d",
        "return_63d",
        "return_126d",
        "return_252d",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",
        "rsi_14",
        "volatility_20d",
        "volatility_60d",
        "volatility_252d",
        "sharpe_252d",
        "rolling_52w_high",
        "rolling_52w_low",
    ]

    for column in expected_nan_columns:

        assert df[column].isna().sum() > 0, (
            f"Expected warm-up NaNs were not found in {column}."
        )


# ============================================================
# TEST 18: NOT ALL VALUES ARE NAN
# ============================================================

def test_feature_columns_not_all_nan():

    df = load_features()

    feature_columns = [
        "daily_return",
        "return_21d",
        "return_63d",
        "sma_20",
        "sma_200",
        "rsi_14",
        "macd",
        "volatility_60d",
        "sharpe_252d",
        "rolling_52w_high",
        "max_drawdown_252d",
        "volume_ratio",
    ]

    for column in feature_columns:

        valid_count = (
            df[column]
            .notna()
            .sum()
        )

        assert valid_count > 0, (
            f"{column} contains only NaN values."
        )


# ============================================================
# TEST 19: ROW COUNT IS CONSISTENT
# ============================================================

def test_feature_row_count_reasonable():

    df = load_features()

    assert len(df) > 1_000_000, (
        f"Unexpectedly low number of rows: {len(df)}"
    )


# ============================================================
# TEST 20: SYMBOL COUNT IS REASONABLE
# ============================================================

def test_feature_symbol_count():

    df = load_features()

    symbol_count = (
        df["symbol"]
        .nunique()
    )

    assert symbol_count >= 400, (
        f"Too few symbols found: {symbol_count}"
    )

    assert symbol_count <= 500, (
        f"Unexpected symbol count: {symbol_count}"
    )


# ============================================================
# TEST 21: DATE RANGE
# ============================================================

def test_feature_date_range():

    df = load_features()

    dates = pd.to_datetime(
        df["date"]
    )

    assert dates.min() <= pd.Timestamp(
        "2015-01-01"
    )

    assert dates.max() >= pd.Timestamp(
        "2025-12-01"
    )


# ============================================================
# TEST 22: DAILY RETURN IS PLAUSIBLE
# ============================================================

def test_daily_return_plausible():

    df = load_features()

    returns = (
        df["daily_return"]
        .dropna()
    )

    assert not returns.empty

    # Very wide safety bounds.
    # This is intended to catch catastrophic calculation errors,
    # not genuine extreme market movements.
    invalid = (
        (returns <= -1.0)
        |
        (returns > 20.0)
    ).sum()

    assert invalid == 0, (
        f"Found {invalid} implausible daily returns."
    )