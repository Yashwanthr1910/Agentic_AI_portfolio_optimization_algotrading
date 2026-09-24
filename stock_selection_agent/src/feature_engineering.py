"""
feature_engineering.py

Feature engineering pipeline for NIFTY-500 stock data.

V1:
    - Price features
    - Return features
    - Risk features
    - Volume features
    - Trend/technical indicators

Input:
    data/processed/nifty500_daily.parquet

Output:
    data/processed/features.parquet
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty500_daily.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

TRADING_DAYS = 252

RISK_FREE_RATE = 0.0


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    logger.info(
        "Loading cleaned dataset..."
    )

    df = pd.read_parquet(INPUT_FILE)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        ["symbol", "date"]
    )

    df = df.reset_index(drop=True)

    logger.info(
        "Loaded %d rows and %d stocks.",
        len(df),
        df["symbol"].nunique()
    )

    return df


# ============================================================
# RETURN FEATURES
# ============================================================

def add_return_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating return features..."
    )

    grouped = df.groupby("symbol")["adj_close"]

    df["daily_return"] = (
        grouped.pct_change()
    )

    periods = {
        "return_5d": 5,
        "return_21d": 21,
        "return_63d": 63,
        "return_126d": 126,
        "return_252d": 252,
    }

    for feature, period in periods.items():

        df[feature] = (
            grouped.pct_change(period)
        )

    return df


# ============================================================
# PRICE BEHAVIOUR FEATURES
# ============================================================

def add_price_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating price behaviour features..."
    )

    df["high_low_range"] = (
        (df["high"] - df["low"])
        / df["close"]
    )

    df["open_close_return"] = (
        (df["close"] - df["open"])
        / df["open"]
    )

    previous_close = (
        df.groupby("symbol")["close"]
        .shift(1)
    )

    df["gap_return"] = (
        (df["open"] - previous_close)
        / previous_close
    )

    return df


# ============================================================
# TREND FEATURES
# ============================================================

def add_trend_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating trend features..."
    )

    group = df.groupby("symbol")["adj_close"]

    # Simple Moving Averages

    for window in [20, 50, 100, 200]:

        df[f"sma_{window}"] = (
            group.transform(
                lambda x:
                x.rolling(window).mean()
            )
        )

    # Exponential Moving Averages

    df["ema_12"] = (
        group.transform(
            lambda x:
            x.ewm(
                span=12,
                adjust=False
            ).mean()
        )
    )

    df["ema_26"] = (
        group.transform(
            lambda x:
            x.ewm(
                span=26,
                adjust=False
            ).mean()
        )
    )

    # Price relative to moving averages

    df["price_to_sma20"] = (
        df["adj_close"]
        / df["sma_20"]
        - 1
    )

    df["price_to_sma50"] = (
        df["adj_close"]
        / df["sma_50"]
        - 1
    )

    df["price_to_sma200"] = (
        df["adj_close"]
        / df["sma_200"]
        - 1
    )

    return df


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    series: pd.Series,
    period: int = 14
) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


def add_rsi(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating RSI..."
    )

    df["rsi_14"] = (
        df.groupby(
            "symbol",
            group_keys=False
        )["adj_close"]
        .apply(calculate_rsi)
    )

    return df


# ============================================================
# MACD
# ============================================================

def add_macd(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating MACD..."
    )

    df["macd"] = (
        df["ema_12"]
        - df["ema_26"]
    )

    df["macd_signal"] = (
        df.groupby("symbol")["macd"]
        .transform(
            lambda x:
            x.ewm(
                span=9,
                adjust=False
            ).mean()
        )
    )

    df["macd_histogram"] = (
        df["macd"]
        - df["macd_signal"]
    )

    return df


# ============================================================
# VOLATILITY
# ============================================================

def add_volatility_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating volatility..."
    )

    grouped = (
        df.groupby("symbol")["daily_return"]
    )

    for window in [20, 60, 252]:

        df[f"volatility_{window}d"] = (
            grouped.transform(
                lambda x:
                x.rolling(window).std()
                * np.sqrt(TRADING_DAYS)
            )
        )

    return df


# ============================================================
# SHARPE RATIO
# ============================================================

def add_sharpe_ratio(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating rolling Sharpe ratio..."
    )

    grouped = (
        df.groupby("symbol")["daily_return"]
    )

    rolling_mean = (
        grouped.transform(
            lambda x:
            x.rolling(252).mean()
        )
    )

    rolling_std = (
        grouped.transform(
            lambda x:
            x.rolling(252).std()
        )
    )

    df["sharpe_252d"] = (
        (
            rolling_mean
            - RISK_FREE_RATE / TRADING_DAYS
        )
        / rolling_std
        * np.sqrt(TRADING_DAYS)
    )

    return df


# ============================================================
# DRAWDOWN FEATURES
# ============================================================

def add_drawdown_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating drawdown features..."
    )

    df["rolling_52w_high"] = (
        df.groupby("symbol")["adj_close"]
        .transform(
            lambda x:
            x.rolling(252).max()
        )
    )

    df["rolling_52w_low"] = (
        df.groupby("symbol")["adj_close"]
        .transform(
            lambda x:
            x.rolling(252).min()
        )
    )

    df["drawdown"] = (
        df["adj_close"]
        / df["rolling_52w_high"]
        - 1
    )

    df["distance_52w_high"] = (
        df["adj_close"]
        / df["rolling_52w_high"]
        - 1
    )

    df["distance_52w_low"] = (
        df["adj_close"]
        / df["rolling_52w_low"]
        - 1
    )

    df["max_drawdown_252d"] = (
        df.groupby("symbol")["drawdown"]
        .transform(
            lambda x:
            x.rolling(252).min()
        )
    )

    return df


# ============================================================
# VOLUME FEATURES
# ============================================================

def add_volume_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating volume features..."
    )

    grouped = (
        df.groupby("symbol")["volume"]
    )

    df["volume_change"] = (
        grouped.pct_change()
    )

    df["avg_volume_20d"] = (
        grouped.transform(
            lambda x:
            x.rolling(20).mean()
        )
    )

    df["volume_ratio"] = (
        df["volume"]
        / df["avg_volume_20d"]
    )

    return df


# ============================================================
# CLEAN GENERATED FEATURES
# ============================================================

def clean_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Cleaning generated features..."
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ============================================================
# FEATURE QUALITY REPORT
# ============================================================

def quality_report(
    df: pd.DataFrame
):

    logger.info(
        "========== FEATURE REPORT =========="
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
        "Columns: %d",
        len(df.columns)
    )

    logger.info(
        "Date range: %s -> %s",
        df["date"].min().date(),
        df["date"].max().date()
    )

    logger.info(
        "Missing values:\n%s",
        df.isna().sum()
    )

    logger.info(
        "===================================="
    )


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(
    df: pd.DataFrame
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    logger.info(
        "Feature dataset saved to: %s",
        OUTPUT_FILE
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    logger.info(
        "Starting feature engineering..."
    )

    df = load_data()

    df = add_return_features(df)

    df = add_price_features(df)

    df = add_trend_features(df)

    df = add_rsi(df)

    df = add_macd(df)

    df = add_volatility_features(df)

    df = add_sharpe_ratio(df)

    df = add_drawdown_features(df)

    df = add_volume_features(df)

    df = clean_features(df)

    quality_report(df)

    save_features(df)

    logger.info(
        "Feature engineering completed."
    )


if __name__ == "__main__":
    main()