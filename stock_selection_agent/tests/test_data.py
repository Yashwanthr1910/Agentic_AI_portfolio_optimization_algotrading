"""
test_data.py

Tests for the cleaned NIFTY-500 dataset.

Run:
    pytest tests/test_data.py -v
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty500_daily.parquet"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_test_data():

    assert DATA_FILE.exists(), (
        f"Cleaned dataset not found: {DATA_FILE}"
    )

    df = pd.read_parquet(DATA_FILE)

    return df


# ============================================================
# TEST 1: FILE EXISTS
# ============================================================

def test_cleaned_data_file_exists():

    assert DATA_FILE.exists(), (
        f"Expected file does not exist: {DATA_FILE}"
    )


# ============================================================
# TEST 2: DATASET IS NOT EMPTY
# ============================================================

def test_dataset_not_empty():

    df = load_test_data()

    assert len(df) > 0, (
        "Cleaned dataset is empty."
    )


# ============================================================
# TEST 3: REQUIRED COLUMNS EXIST
# ============================================================

def test_required_columns_exist():

    df = load_test_data()

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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    assert not missing_columns, (
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# TEST 4: SYMBOLS EXIST
# ============================================================

def test_symbols_exist():

    df = load_test_data()

    assert df["symbol"].notna().all(), (
        "Missing symbol values found."
    )

    assert df["symbol"].nunique() > 0, (
        "No stocks found in dataset."
    )


# ============================================================
# TEST 5: DATE COLUMN VALID
# ============================================================

def test_date_column_valid():

    df = load_test_data()

    dates = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    assert dates.notna().all(), (
        "Invalid dates detected."
    )


# ============================================================
# TEST 6: NO DUPLICATE SYMBOL-DATE ROWS
# ============================================================

def test_no_duplicate_symbol_date_rows():

    df = load_test_data()

    duplicate_count = (
        df.duplicated(
            subset=[
                "symbol",
                "date",
            ]
        ).sum()
    )

    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicate "
        "symbol-date records."
    )


# ============================================================
# TEST 7: PRICE VALUES ARE POSITIVE
# ============================================================

def test_price_values_positive():

    df = load_test_data()

    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
    ]

    for column in price_columns:

        assert (
            df[column] > 0
        ).all(), (
            f"Non-positive values found in {column}."
        )


# ============================================================
# TEST 8: VOLUME IS NON-NEGATIVE
# ============================================================

def test_volume_non_negative():

    df = load_test_data()

    assert (
        df["volume"] >= 0
    ).all(), (
        "Negative volume values found."
    )


# ============================================================
# TEST 9: HIGH >= LOW
# ============================================================

def test_high_greater_than_or_equal_to_low():

    df = load_test_data()

    invalid_rows = (
        df["high"]
        <
        df["low"]
    ).sum()

    assert invalid_rows == 0, (
        f"Found {invalid_rows} rows where high < low."
    )


# ============================================================
# TEST 10: HIGH >= OPEN AND CLOSE
# ============================================================

def test_high_consistency():

    df = load_test_data()

    invalid_open = (
        df["high"]
        <
        df["open"]
    ).sum()

    invalid_close = (
        df["high"]
        <
        df["close"]
    ).sum()

    assert invalid_open == 0, (
        f"Found {invalid_open} rows where high < open."
    )

    assert invalid_close == 0, (
        f"Found {invalid_close} rows where high < close."
    )


# ============================================================
# TEST 11: LOW <= OPEN AND CLOSE
# ============================================================

def test_low_consistency():

    df = load_test_data()

    invalid_open = (
        df["low"]
        >
        df["open"]
    ).sum()

    invalid_close = (
        df["low"]
        >
        df["close"]
    ).sum()

    assert invalid_open == 0, (
        f"Found {invalid_open} rows where low > open."
    )

    assert invalid_close == 0, (
        f"Found {invalid_close} rows where low > close."
    )


# ============================================================
# TEST 12: EXPECTED DATE RANGE
# ============================================================

def test_date_range():

    df = load_test_data()

    dates = pd.to_datetime(
        df["date"]
    )

    min_date = dates.min()

    max_date = dates.max()

    assert min_date <= pd.Timestamp(
        "2015-01-01"
    ), (
        f"Unexpected start date: {min_date}"
    )

    assert max_date >= pd.Timestamp(
        "2025-12-01"
    ), (
        f"Unexpected end date: {max_date}"
    )


# ============================================================
# TEST 13: REASONABLE NUMBER OF STOCKS
# ============================================================

def test_reasonable_stock_count():

    df = load_test_data()

    stock_count = (
        df["symbol"]
        .nunique()
    )

    assert stock_count >= 400, (
        f"Too few stocks found: {stock_count}"
    )

    assert stock_count <= 500, (
        f"Unexpectedly high stock count: {stock_count}"
    )


# ============================================================
# TEST 14: DATA SORTING
# ============================================================

def test_each_symbol_dates_are_ordered():

    df = load_test_data().copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    for symbol, group in df.groupby(
        "symbol"
    ):

        dates = group[
            "date"
        ].reset_index(
            drop=True
        )

        assert dates.is_monotonic_increasing, (
            f"Dates are not sorted for symbol: {symbol}"
        )