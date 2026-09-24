"""
data_cleaner.py

Cleans individual NIFTY 500 historical stock CSV files downloaded
using yfinance and combines them into a single processed dataset.

Input:
    data/raw/daily/*.csv

Output:
    data/processed/nifty500_daily.parquet
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "daily"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DATA_DIR / "nifty500_daily.parquet"


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MIN_REQUIRED_ROWS = 100

REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


# ---------------------------------------------------------
# COLUMN STANDARDIZATION
# ---------------------------------------------------------

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to a consistent lowercase format.

    Example:
        'Adj Close' -> 'adj_close'
        'Date'      -> 'date'
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "_", regex=False)
    )

    return df


# ---------------------------------------------------------
# REMOVE UNWANTED INDEX COLUMNS
# ---------------------------------------------------------

def remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns such as 'Unnamed: 0' that may appear
    after saving pandas DataFrames to CSV.
    """

    unnamed_columns = [
        col for col in df.columns
        if col.lower().startswith("unnamed")
    ]

    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    return df


# ---------------------------------------------------------
# FLATTEN POSSIBLE YFINANCE COLUMN NAMES
# ---------------------------------------------------------

def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles common column variations produced by yfinance.
    """

    df = df.copy()

    rename_map = {
        "adj_close": "adj_close",
        "adjclose": "adj_close",
        "adjusted_close": "adj_close",
    }

    df = df.rename(columns=rename_map)

    return df


# ---------------------------------------------------------
# DATE CLEANING
# ---------------------------------------------------------

def clean_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the date column into datetime format and
    removes invalid dates.
    """

    df = df.copy()

    if "date" not in df.columns:
        raise ValueError("Missing required 'date' column.")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True
    )

    # Remove timezone to simplify later backtesting
    df["date"] = df["date"].dt.tz_localize(None)

    # Remove invalid dates
    df = df.dropna(subset=["date"])

    return df


# ---------------------------------------------------------
# NUMERIC CLEANING
# ---------------------------------------------------------

def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts OHLCV columns into numeric values.
    """

    df = df.copy()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ---------------------------------------------------------
# REQUIRED COLUMN VALIDATION
# ---------------------------------------------------------

def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Ensures essential columns are present.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


# ---------------------------------------------------------
# REMOVE DUPLICATE DATES
# ---------------------------------------------------------

def remove_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate trading dates.

    Keeps the latest occurrence.
    """

    df = df.copy()

    df = df.drop_duplicates(
        subset=["date"],
        keep="last"
    )

    return df


# ---------------------------------------------------------
# REMOVE INVALID PRICE DATA
# ---------------------------------------------------------

def remove_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows containing impossible OHLC price values.
    """

    df = df.copy()

    price_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    # Remove missing OHLC prices
    df = df.dropna(subset=price_columns)

    # Prices cannot be zero or negative
    valid_price_mask = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    )

    df = df.loc[valid_price_mask]

    return df


# ---------------------------------------------------------
# VALIDATE OHLC RELATIONSHIPS
# ---------------------------------------------------------

def validate_ohlc_relationships(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Removes rows where OHLC relationships are logically invalid.

    High should be >= Open, Close and Low.
    Low should be <= Open, Close and High.
    """

    df = df.copy()

    valid_ohlc = (
        (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    )

    df = df.loc[valid_ohlc]

    return df


# ---------------------------------------------------------
# CLEAN VOLUME
# ---------------------------------------------------------

def clean_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans volume values.

    Negative volume values are invalid.
    Zero volume is retained because it may occasionally
    occur for illiquid securities.
    """

    df = df.copy()

    df = df.dropna(subset=["volume"])

    df = df.loc[df["volume"] >= 0]

    df["volume"] = df["volume"].astype("int64")

    return df


# ---------------------------------------------------------
# CLEAN ADJUSTED CLOSE
# ---------------------------------------------------------

def clean_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles Adjusted Close.

    Newer versions of yfinance may auto-adjust prices and
    may not always return Adj Close.
    """

    df = df.copy()

    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # If adj_close contains missing/invalid values,
    # fall back to normal close.
    invalid_mask = (
        df["adj_close"].isna()
        | (df["adj_close"] <= 0)
    )

    df.loc[invalid_mask, "adj_close"] = df.loc[
        invalid_mask,
        "close"
    ]

    return df


# ---------------------------------------------------------
# REMOVE INFINITE VALUES
# ---------------------------------------------------------

def remove_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts positive/negative infinity into NaN.
    """

    df = df.copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ---------------------------------------------------------
# CLEAN ONE STOCK
# ---------------------------------------------------------

def clean_stock_data(
    df: pd.DataFrame,
    symbol: str
) -> pd.DataFrame:
    """
    Cleans historical data for a single stock.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock price data.

    symbol : str
        NSE stock symbol.

    Returns
    -------
    pd.DataFrame
        Cleaned stock data.
    """

    df = standardize_column_names(df)

    df = remove_unnamed_columns(df)

    df = normalize_yfinance_columns(df)

    validate_required_columns(df)

    df = clean_date_column(df)

    df = convert_numeric_columns(df)

    df = remove_infinite_values(df)

    df = remove_duplicate_dates(df)

    df = remove_invalid_prices(df)

    df = validate_ohlc_relationships(df)

    df = clean_volume(df)

    df = clean_adjusted_close(df)

    # Add NSE symbol
    df["symbol"] = symbol.upper()

    # Sort chronologically
    df = df.sort_values("date")

    df = df.reset_index(drop=True)

    # Final column order
    final_columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    df = df[final_columns]

    return df


# ---------------------------------------------------------
# LOAD SINGLE CSV
# ---------------------------------------------------------

def load_and_clean_stock(
    file_path: Path
) -> pd.DataFrame | None:
    """
    Loads and cleans one stock CSV file.
    """

    symbol = file_path.stem.upper()

    try:

        raw_df = pd.read_csv(file_path)

        if raw_df.empty:
            logger.warning(
                "%s skipped: file is empty.",
                symbol
            )
            return None

        original_rows = len(raw_df)

        clean_df = clean_stock_data(
            raw_df,
            symbol
        )

        cleaned_rows = len(clean_df)

        removed_rows = original_rows - cleaned_rows

        if cleaned_rows < MIN_REQUIRED_ROWS:
            logger.warning(
                "%s skipped: only %d valid rows.",
                symbol,
                cleaned_rows
            )
            return None

        logger.info(
            "%s | raw=%d | clean=%d | removed=%d",
            symbol,
            original_rows,
            cleaned_rows,
            removed_rows
        )

        return clean_df

    except Exception as error:

        logger.error(
            "Failed to clean %s: %s",
            symbol,
            error
        )

        return None


# ---------------------------------------------------------
# CLEAN ALL STOCK FILES
# ---------------------------------------------------------

def clean_all_stocks() -> pd.DataFrame:
    """
    Cleans every stock CSV inside data/raw/daily/
    and combines them.
    """

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: "
            f"{RAW_DATA_DIR}"
        )

    csv_files = sorted(
        RAW_DATA_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found inside "
            f"{RAW_DATA_DIR}"
        )

    logger.info(
        "Found %d stock files.",
        len(csv_files)
    )

    cleaned_datasets = []

    failed_symbols = []

    for file_path in csv_files:

        cleaned_df = load_and_clean_stock(
            file_path
        )

        if cleaned_df is not None:
            cleaned_datasets.append(
                cleaned_df
            )
        else:
            failed_symbols.append(
                file_path.stem.upper()
            )

    if not cleaned_datasets:
        raise RuntimeError(
            "No valid stock datasets were produced."
        )

    combined_df = pd.concat(
        cleaned_datasets,
        ignore_index=True
    )

    # Remove accidental duplicate symbol/date rows
    combined_df = combined_df.drop_duplicates(
        subset=["symbol", "date"],
        keep="last"
    )

    combined_df = combined_df.sort_values(
        ["symbol", "date"]
    )

    combined_df = combined_df.reset_index(
        drop=True
    )

    logger.info(
        "Successfully cleaned %d stocks.",
        combined_df["symbol"].nunique()
    )

    logger.info(
        "Total observations: %d",
        len(combined_df)
    )

    if failed_symbols:
        logger.warning(
            "Skipped %d stocks: %s",
            len(failed_symbols),
            ", ".join(failed_symbols)
        )

    return combined_df


# ---------------------------------------------------------
# DATA QUALITY REPORT
# ---------------------------------------------------------

def generate_quality_report(
    df: pd.DataFrame
) -> None:
    """
    Prints useful data-quality statistics.
    """

    logger.info(
        "========== DATA QUALITY REPORT =========="
    )

    logger.info(
        "Stocks: %d",
        df["symbol"].nunique()
    )

    logger.info(
        "Rows: %d",
        len(df)
    )

    logger.info(
        "Start date: %s",
        df["date"].min().date()
    )

    logger.info(
        "End date: %s",
        df["date"].max().date()
    )

    logger.info(
        "Duplicate symbol/date rows: %d",
        df.duplicated(
            subset=["symbol", "date"]
        ).sum()
    )

    logger.info(
        "Missing values:\n%s",
        df.isna().sum()
    )

    observations_per_stock = (
        df.groupby("symbol")
        .size()
    )

    logger.info(
        "Minimum rows per stock: %d",
        observations_per_stock.min()
    )

    logger.info(
        "Maximum rows per stock: %d",
        observations_per_stock.max()
    )

    logger.info(
        "Median rows per stock: %.0f",
        observations_per_stock.median()
    )

    logger.info(
        "========================================="
    )


# ---------------------------------------------------------
# SAVE PROCESSED DATA
# ---------------------------------------------------------

def save_processed_data(
    df: pd.DataFrame
) -> None:
    """
    Saves cleaned dataset to Parquet.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    logger.info(
        "Processed dataset saved to: %s",
        OUTPUT_FILE
    )


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():
    """
    Runs the complete NIFTY 500 cleaning pipeline.
    """

    logger.info(
        "Starting NIFTY 500 data cleaning..."
    )

    combined_df = clean_all_stocks()

    generate_quality_report(
        combined_df
    )

    save_processed_data(
        combined_df
    )

    logger.info(
        "Data cleaning completed successfully."
    )


if __name__ == "__main__":
    main()