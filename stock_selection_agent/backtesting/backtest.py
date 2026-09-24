"""
backtest.py

Stock Selection Agent - Portfolio Backtesting Module

Purpose:
    Backtest the portfolio produced by portfolio.py using
    actual historical adjusted-close prices AFTER the
    stock-selection date.

Inputs:
    outputs/portfolio.csv
    data/processed/nifty500_daily.parquet

Outputs:
    outputs/backtest_results.csv
    outputs/backtest_results.parquet
    outputs/performance.csv

Pipeline:

    portfolio.csv
         +
    nifty500_daily.parquet
         ↓
      backtest.py
         ↓
    daily stock returns
         ↓
    portfolio returns
         ↓
    portfolio value
         ↓
    performance.py
         ↓
    performance.csv
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd

from performance import evaluate_performance


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PORTFOLIO_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "portfolio.csv"
)

PRICE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nifty500_daily.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

BACKTEST_CSV = (
    OUTPUT_DIR
    / "backtest_results.csv"
)

BACKTEST_PARQUET = (
    OUTPUT_DIR
    / "backtest_results.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_CAPITAL = 1_000_000.0

# No transaction cost for the initial baseline.
#
# Later we can add realistic brokerage/slippage.
TRANSACTION_COST_RATE = 0.0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD PORTFOLIO
# ============================================================

def load_portfolio() -> pd.DataFrame:

    logger.info(
        "Loading portfolio..."
    )

    if not PORTFOLIO_FILE.exists():

        raise FileNotFoundError(
            "portfolio.csv not found.\n"
            f"Expected location: {PORTFOLIO_FILE}\n\n"
            "Run backtesting/portfolio.py first."
        )

    portfolio = pd.read_csv(
        PORTFOLIO_FILE
    )

    if portfolio.empty:

        raise ValueError(
            "portfolio.csv is empty."
        )

    if "date" in portfolio.columns:

        portfolio["date"] = pd.to_datetime(
            portfolio["date"]
        )

    required_columns = [
        "symbol",
        "portfolio_weight"
    ]

    missing = [
        column
        for column in required_columns
        if column not in portfolio.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required portfolio columns: {missing}"
        )

    logger.info(
        "Loaded portfolio containing %d stocks.",
        len(portfolio)
    )

    return portfolio


# ============================================================
# GET SELECTION DATE
# ============================================================

def get_selection_date(
    portfolio: pd.DataFrame
) -> pd.Timestamp:

    if "date" not in portfolio.columns:

        raise ValueError(
            "Portfolio does not contain a 'date' column."
        )

    dates = (
        portfolio["date"]
        .dropna()
        .unique()
    )

    if len(dates) == 0:

        raise ValueError(
            "No portfolio selection date found."
        )

    if len(dates) > 1:

        raise ValueError(
            "Current backtester expects one portfolio "
            "selection date, but multiple dates were found."
        )

    selection_date = pd.Timestamp(
        dates[0]
    )

    logger.info(
        "Portfolio selection date: %s",
        selection_date.date()
    )

    return selection_date


# ============================================================
# LOAD PRICE DATA
# ============================================================

def load_price_data(
    symbols: list[str],
    selection_date: pd.Timestamp
) -> pd.DataFrame:

    logger.info(
        "Loading historical price data..."
    )

    if not PRICE_FILE.exists():

        raise FileNotFoundError(
            "Price parquet file not found.\n"
            f"Expected location: {PRICE_FILE}"
        )

    prices = pd.read_parquet(
        PRICE_FILE
    )

    required_columns = [
        "date",
        "symbol",
        "adj_close"
    ]

    missing = [
        column
        for column in required_columns
        if column not in prices.columns
    ]

    if missing:

        raise ValueError(
            f"Price dataset is missing columns: {missing}"
        )

    prices["date"] = pd.to_datetime(
        prices["date"]
    )

    # --------------------------------------------------------
    # Keep only selected portfolio stocks.
    # --------------------------------------------------------

    prices = prices[
        prices["symbol"].isin(
            symbols
        )
    ].copy()

    if prices.empty:

        raise ValueError(
            "No price data found for selected portfolio stocks."
        )

    # --------------------------------------------------------
    # Keep selection date + all future observations.
    #
    # We need selection-date prices so that the first
    # post-selection daily return can be calculated.
    # --------------------------------------------------------

    prices = prices[
        prices["date"] >= selection_date
    ].copy()

    prices = prices.sort_values(
        [
            "date",
            "symbol"
        ]
    )

    if prices.empty:

        raise ValueError(
            "No price data exists on or after the "
            "portfolio selection date."
        )

    available_symbols = set(
        prices["symbol"].unique()
    )

    missing_symbols = [
        symbol
        for symbol in symbols
        if symbol not in available_symbols
    ]

    if missing_symbols:

        logger.warning(
            "No post-selection price data found for: %s",
            missing_symbols
        )

    logger.info(
        "Loaded %d price observations for %d stocks.",
        len(prices),
        prices["symbol"].nunique()
    )

    logger.info(
        "Price date range: %s -> %s",
        prices["date"].min().date(),
        prices["date"].max().date()
    )

    return prices


# ============================================================
# CREATE PRICE MATRIX
# ============================================================

def create_price_matrix(
    prices: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Creating adjusted-close price matrix..."
    )

    price_matrix = (
        prices
        .pivot(
            index="date",
            columns="symbol",
            values="adj_close"
        )
        .sort_index()
    )

    # --------------------------------------------------------
    # Ensure numeric values
    # --------------------------------------------------------

    price_matrix = price_matrix.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove duplicate index if somehow present
    # --------------------------------------------------------

    price_matrix = price_matrix[
        ~price_matrix.index.duplicated(
            keep="last"
        )
    ]

    logger.info(
        "Price matrix shape: %s",
        price_matrix.shape
    )

    return price_matrix


# ============================================================
# CALCULATE STOCK RETURNS
# ============================================================

def calculate_stock_returns(
    price_matrix: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating daily stock returns..."
    )

    # Explicit fill_method=None avoids artificially
    # forward-filling missing prices.
    stock_returns = (
        price_matrix
        .pct_change(
            fill_method=None
        )
    )

    stock_returns = stock_returns.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    return stock_returns


# ============================================================
# PREPARE PORTFOLIO WEIGHTS
# ============================================================

def prepare_weights(
    portfolio: pd.DataFrame,
    available_symbols: list[str]
) -> pd.Series:

    logger.info(
        "Preparing portfolio weights..."
    )

    weights = (
        portfolio
        .set_index(
            "symbol"
        )[
            "portfolio_weight"
        ]
    )

    weights = pd.to_numeric(
        weights,
        errors="coerce"
    )

    weights = weights.dropna()

    # --------------------------------------------------------
    # Keep only symbols actually available in price data.
    # --------------------------------------------------------

    weights = weights[
        weights.index.isin(
            available_symbols
        )
    ]

    if weights.empty:

        raise ValueError(
            "No valid portfolio weights remain after "
            "matching against price data."
        )

    total_weight = (
        weights.sum()
    )

    if total_weight <= 0:

        raise ValueError(
            "Portfolio weights must sum to a positive value."
        )

    # --------------------------------------------------------
    # Normalize weights exactly to 1.
    # --------------------------------------------------------

    weights = (
        weights
        / total_weight
    )

    logger.info(
        "Portfolio weight sum: %.6f",
        weights.sum()
    )

    return weights


# ============================================================
# CALCULATE DAILY PORTFOLIO RETURN
# ============================================================

def calculate_portfolio_returns(
    stock_returns: pd.DataFrame,
    weights: pd.Series
) -> pd.DataFrame:

    logger.info(
        "Calculating portfolio daily returns..."
    )

    # --------------------------------------------------------
    # Align return matrix to portfolio stocks.
    # --------------------------------------------------------

    common_symbols = [
        symbol
        for symbol in weights.index
        if symbol in stock_returns.columns
    ]

    if not common_symbols:

        raise ValueError(
            "No common symbols between portfolio and "
            "stock return data."
        )

    stock_returns = stock_returns[
        common_symbols
    ].copy()

    weights = weights[
        common_symbols
    ].copy()

    # --------------------------------------------------------
    # Calculate portfolio return day-by-day.
    #
    # If a stock is missing a return on a particular day,
    # temporarily normalize weights over stocks having
    # valid observations that day.
    # --------------------------------------------------------

    portfolio_returns = []

    active_stock_counts = []

    for date, row in stock_returns.iterrows():

        valid_returns = (
            row.dropna()
        )

        if valid_returns.empty:

            portfolio_returns.append(
                np.nan
            )

            active_stock_counts.append(
                0
            )

            continue

        daily_weights = weights[
            valid_returns.index
        ].copy()

        weight_sum = (
            daily_weights.sum()
        )

        if weight_sum <= 0:

            portfolio_returns.append(
                np.nan
            )

            active_stock_counts.append(
                0
            )

            continue

        daily_weights = (
            daily_weights
            / weight_sum
        )

        portfolio_return = (
            valid_returns
            * daily_weights
        ).sum()

        portfolio_returns.append(
            portfolio_return
        )

        active_stock_counts.append(
            len(valid_returns)
        )

    result = pd.DataFrame(
        {
            "date": stock_returns.index,
            "portfolio_return": portfolio_returns,
            "active_stocks": active_stock_counts
        }
    )

    return result


# ============================================================
# REMOVE SELECTION-DATE RETURN
# ============================================================

def keep_post_selection_returns(
    backtest: pd.DataFrame,
    selection_date: pd.Timestamp
) -> pd.DataFrame:

    """
    A stock is selected using information available on the
    selection date.

    Therefore performance begins AFTER that date.
    """

    backtest = backtest[
        backtest["date"] > selection_date
    ].copy()

    backtest = backtest.dropna(
        subset=[
            "portfolio_return"
        ]
    )

    backtest = backtest.reset_index(
        drop=True
    )

    if backtest.empty:

        raise ValueError(
            "No valid trading days exist after the "
            "selection date."
        )

    return backtest


# ============================================================
# TRANSACTION COST
# ============================================================

def apply_initial_transaction_cost(
    backtest: pd.DataFrame
) -> pd.DataFrame:

    backtest = backtest.copy()

    backtest[
        "transaction_cost"
    ] = 0.0

    if (
        TRANSACTION_COST_RATE > 0
        and not backtest.empty
    ):

        # Full initial investment incurs the configured
        # transaction cost once.
        backtest.loc[
            backtest.index[0],
            "transaction_cost"
        ] = TRANSACTION_COST_RATE

        backtest.loc[
            backtest.index[0],
            "portfolio_return"
        ] -= TRANSACTION_COST_RATE

    return backtest


# ============================================================
# PORTFOLIO VALUE
# ============================================================

def calculate_portfolio_value(
    backtest: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating portfolio value..."
    )

    backtest = backtest.copy()

    backtest[
        "portfolio_growth"
    ] = (
        1
        + backtest[
            "portfolio_return"
        ]
    ).cumprod()

    backtest[
        "portfolio_value"
    ] = (
        INITIAL_CAPITAL
        * backtest[
            "portfolio_growth"
        ]
    )

    backtest[
        "cumulative_return"
    ] = (
        backtest[
            "portfolio_growth"
        ]
        - 1
    )

    # --------------------------------------------------------
    # Running peak
    # --------------------------------------------------------

    backtest[
        "running_peak"
    ] = (
        backtest[
            "portfolio_value"
        ]
        .cummax()
    )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    backtest[
        "drawdown"
    ] = (
        backtest[
            "portfolio_value"
        ]
        /
        backtest[
            "running_peak"
        ]
        - 1
    )

    return backtest


# ============================================================
# ADD INDIVIDUAL STOCK RETURNS
# ============================================================

def add_stock_returns_to_output(
    backtest: pd.DataFrame,
    stock_returns: pd.DataFrame
) -> pd.DataFrame:

    stock_return_output = (
        stock_returns
        .copy()
        .reset_index()
    )

    # --------------------------------------------------------
    # Prefix stock return columns
    # --------------------------------------------------------

    rename_mapping = {
        column: f"return_{column}"
        for column in stock_return_output.columns
        if column != "date"
    }

    stock_return_output = (
        stock_return_output.rename(
            columns=rename_mapping
        )
    )

    backtest = backtest.merge(
        stock_return_output,
        on="date",
        how="left"
    )

    return backtest


# ============================================================
# SAVE BACKTEST
# ============================================================

def save_backtest(
    backtest: pd.DataFrame
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    backtest.to_csv(
        BACKTEST_CSV,
        index=False
    )

    backtest.to_parquet(
        BACKTEST_PARQUET,
        index=False
    )

    logger.info(
        "Backtest CSV saved to:"
    )

    logger.info(
        "%s",
        BACKTEST_CSV
    )

    logger.info(
        "Backtest Parquet saved to:"
    )

    logger.info(
        "%s",
        BACKTEST_PARQUET
    )


# ============================================================
# DISPLAY BACKTEST SUMMARY
# ============================================================

def display_backtest_summary(
    backtest: pd.DataFrame,
    selection_date: pd.Timestamp,
    symbols: list[str]
) -> None:

    logger.info(
        ""
    )

    logger.info(
        "=============================================================="
    )

    logger.info(
        "                      BACKTEST SUMMARY"
    )

    logger.info(
        "=============================================================="
    )

    logger.info(
        "Selection date      : %s",
        selection_date.date()
    )

    logger.info(
        "Backtest start      : %s",
        backtest[
            "date"
        ].min().date()
    )

    logger.info(
        "Backtest end        : %s",
        backtest[
            "date"
        ].max().date()
    )

    logger.info(
        "Trading days        : %d",
        len(backtest)
    )

    logger.info(
        "Portfolio stocks    : %d",
        len(symbols)
    )

    logger.info(
        "Initial capital     : %.2f",
        INITIAL_CAPITAL
    )

    logger.info(
        "Final value         : %.2f",
        backtest[
            "portfolio_value"
        ].iloc[-1]
    )

    logger.info(
        "Total return        : %.2f%%",
        backtest[
            "cumulative_return"
        ].iloc[-1]
        * 100
    )

    logger.info(
        "Maximum drawdown    : %.2f%%",
        backtest[
            "drawdown"
        ].min()
        * 100
    )

    logger.info(
        "--------------------------------------------------------------"
    )

    logger.info(
        "Selected symbols:"
    )

    logger.info(
        "%s",
        ", ".join(
            symbols
        )
    )

    logger.info(
        "=============================================================="
    )


# ============================================================
# RUN BACKTEST
# ============================================================

def run_backtest() -> pd.DataFrame:

    logger.info(
        "=============================================================="
    )

    logger.info(
        "STARTING PORTFOLIO BACKTEST"
    )

    logger.info(
        "=============================================================="
    )

    # --------------------------------------------------------
    # Step 1: Load portfolio
    # --------------------------------------------------------

    portfolio = (
        load_portfolio()
    )

    # --------------------------------------------------------
    # Step 2: Selection date
    # --------------------------------------------------------

    selection_date = (
        get_selection_date(
            portfolio
        )
    )

    # --------------------------------------------------------
    # Step 3: Symbols
    # --------------------------------------------------------

    symbols = (
        portfolio[
            "symbol"
        ]
        .astype(str)
        .tolist()
    )

    logger.info(
        "Portfolio symbols: %s",
        ", ".join(
            symbols
        )
    )

    # --------------------------------------------------------
    # Step 4: Load prices
    # --------------------------------------------------------

    prices = (
        load_price_data(
            symbols,
            selection_date
        )
    )

    # --------------------------------------------------------
    # Step 5: Price matrix
    # --------------------------------------------------------

    price_matrix = (
        create_price_matrix(
            prices
        )
    )

    # --------------------------------------------------------
    # Step 6: Stock daily returns
    # --------------------------------------------------------

    stock_returns = (
        calculate_stock_returns(
            price_matrix
        )
    )

    # --------------------------------------------------------
    # Step 7: Portfolio weights
    # --------------------------------------------------------

    weights = (
        prepare_weights(
            portfolio,
            list(
                price_matrix.columns
            )
        )
    )

    # --------------------------------------------------------
    # Step 8: Portfolio daily return
    # --------------------------------------------------------

    backtest = (
        calculate_portfolio_returns(
            stock_returns,
            weights
        )
    )

    # --------------------------------------------------------
    # Step 9:
    # Keep ONLY dates after stock selection.
    # --------------------------------------------------------

    backtest = (
        keep_post_selection_returns(
            backtest,
            selection_date
        )
    )

    # --------------------------------------------------------
    # Step 10: Transaction cost
    # --------------------------------------------------------

    backtest = (
        apply_initial_transaction_cost(
            backtest
        )
    )

    # --------------------------------------------------------
    # Step 11: Portfolio value
    # --------------------------------------------------------

    backtest = (
        calculate_portfolio_value(
            backtest
        )
    )

    # --------------------------------------------------------
    # Step 12:
    # Add individual stock returns for inspection.
    # --------------------------------------------------------

    backtest = (
        add_stock_returns_to_output(
            backtest,
            stock_returns
        )
    )

    # --------------------------------------------------------
    # Step 13: Save detailed backtest
    # --------------------------------------------------------

    save_backtest(
        backtest
    )

    # --------------------------------------------------------
    # Step 14: Display summary
    # --------------------------------------------------------

    display_backtest_summary(
        backtest,
        selection_date,
        symbols
    )

    # --------------------------------------------------------
    # Step 15:
    # Calculate performance metrics using performance.py
    # --------------------------------------------------------

    logger.info(
        ""
    )

    logger.info(
        "Calculating performance metrics..."
    )

    evaluate_performance(
        backtest[
            "portfolio_return"
        ]
    )

    logger.info(
        ""
    )

    logger.info(
        "Backtest completed successfully."
    )

    return backtest


# ============================================================
# MAIN
# ============================================================

def main():

    run_backtest()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()