"""
portfolio.py

Portfolio Construction Module

Purpose:
    Build a portfolio from the final ranked stocks produced
    by stock_scorer.py.

Input:
    outputs/rankings.csv

Output:
    outputs/portfolio.csv
    outputs/portfolio.parquet

Current method:
    Equal-weight allocation across selected stocks.

Example:
    If 10 stocks are selected:

        weight per stock = 1 / 10 = 10%

Pipeline:

    rankings.csv
        ↓
    portfolio.py
        ↓
    portfolio.csv
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "rankings.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

PORTFOLIO_CSV = (
    OUTPUT_DIR
    / "portfolio.csv"
)

PORTFOLIO_PARQUET = (
    OUTPUT_DIR
    / "portfolio.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Number of stocks to include in final portfolio.
#
# Since your current stock-selection stage selects 10 stocks,
# this will currently include all 10.
TOP_N_STOCKS = 10

# Initial portfolio capital.
#
# Change this later if required.
INITIAL_CAPITAL = 1_000_000.0

# Portfolio weighting method
WEIGHTING_METHOD = "equal"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD RANKINGS
# ============================================================

def load_rankings() -> pd.DataFrame:

    logger.info(
        "Loading stock rankings..."
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "rankings.csv not found.\n"
            f"Expected location: {INPUT_FILE}\n\n"
            "Run stock_scorer.py first."
        )

    rankings = pd.read_csv(
        INPUT_FILE
    )

    if rankings.empty:

        raise ValueError(
            "rankings.csv is empty."
        )

    if "date" in rankings.columns:

        rankings["date"] = pd.to_datetime(
            rankings["date"]
        )

    logger.info(
        "Loaded %d ranked stocks.",
        len(rankings)
    )

    return rankings


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_rankings(
    rankings: pd.DataFrame
) -> None:

    logger.info(
        "Validating ranking data..."
    )

    required_columns = [
        "symbol",
        "final_rank",
        "final_stock_score"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in rankings.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required ranking columns: "
            f"{missing_columns}"
        )

    if rankings["symbol"].isna().any():

        raise ValueError(
            "Missing stock symbols detected."
        )

    duplicate_count = (
        rankings["symbol"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        raise ValueError(
            f"Found {duplicate_count} duplicate symbols."
        )

    logger.info(
        "Ranking validation passed."
    )


# ============================================================
# SELECT PORTFOLIO STOCKS
# ============================================================

def select_portfolio_stocks(
    rankings: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Selecting top %d stocks for portfolio...",
        TOP_N_STOCKS
    )

    portfolio = (
        rankings
        .sort_values(
            "final_rank"
        )
        .head(
            TOP_N_STOCKS
        )
        .copy()
    )

    portfolio = portfolio.reset_index(
        drop=True
    )

    if portfolio.empty:

        raise ValueError(
            "No stocks available for portfolio construction."
        )

    logger.info(
        "%d stocks selected for portfolio.",
        len(portfolio)
    )

    return portfolio


# ============================================================
# EQUAL-WEIGHT PORTFOLIO
# ============================================================

def assign_equal_weights(
    portfolio: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Applying equal-weight allocation..."
    )

    portfolio = portfolio.copy()

    number_of_stocks = len(
        portfolio
    )

    weight = (
        1.0
        /
        number_of_stocks
    )

    portfolio[
        "portfolio_weight"
    ] = weight

    portfolio[
        "portfolio_weight_percent"
    ] = (
        weight
        * 100
    )

    return portfolio


# ============================================================
# CAPITAL ALLOCATION
# ============================================================

def allocate_capital(
    portfolio: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Allocating initial capital..."
    )

    portfolio = portfolio.copy()

    portfolio[
        "allocated_capital"
    ] = (
        INITIAL_CAPITAL
        *
        portfolio[
            "portfolio_weight"
        ]
    )

    return portfolio


# ============================================================
# PORTFOLIO CONTRIBUTION SCORES
# ============================================================

def calculate_portfolio_metrics(
    portfolio: pd.DataFrame
) -> pd.DataFrame:

    """
    Add useful portfolio-level contribution columns.

    These are NOT realized backtest returns yet.

    They simply show weighted values of the features
    available at portfolio construction time.
    """

    portfolio = portfolio.copy()

    if "return_21d" in portfolio.columns:

        portfolio[
            "weighted_return_21d"
        ] = (
            portfolio[
                "portfolio_weight"
            ]
            *
            portfolio[
                "return_21d"
            ]
        )

    if "return_63d" in portfolio.columns:

        portfolio[
            "weighted_return_63d"
        ] = (
            portfolio[
                "portfolio_weight"
            ]
            *
            portfolio[
                "return_63d"
            ]
        )

    if "volatility_60d" in portfolio.columns:

        portfolio[
            "weighted_volatility_60d"
        ] = (
            portfolio[
                "portfolio_weight"
            ]
            *
            portfolio[
                "volatility_60d"
            ]
        )

    if "sharpe_252d" in portfolio.columns:

        portfolio[
            "weighted_sharpe_252d"
        ] = (
            portfolio[
                "portfolio_weight"
            ]
            *
            portfolio[
                "sharpe_252d"
            ]
        )

    return portfolio


# ============================================================
# PORTFOLIO SUMMARY
# ============================================================

def display_portfolio_summary(
    portfolio: pd.DataFrame
) -> None:

    logger.info(
        ""
    )

    logger.info(
        "=============================================================="
    )

    logger.info(
        "                     PORTFOLIO SUMMARY"
    )

    logger.info(
        "=============================================================="
    )

    logger.info(
        "Initial Capital : %.2f",
        INITIAL_CAPITAL
    )

    logger.info(
        "Stocks          : %d",
        len(portfolio)
    )

    logger.info(
        "Weighting       : %s",
        WEIGHTING_METHOD.upper()
    )

    logger.info(
        "--------------------------------------------------------------"
    )

    display_columns = [
        "final_rank",
        "symbol",
        "final_stock_score",
        "portfolio_weight_percent",
        "allocated_capital"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in portfolio.columns
    ]

    logger.info(
        "\n%s",
        portfolio[
            display_columns
        ].to_string(
            index=False
        )
    )

    logger.info(
        "--------------------------------------------------------------"
    )

    total_weight = (
        portfolio[
            "portfolio_weight"
        ]
        .sum()
    )

    total_capital = (
        portfolio[
            "allocated_capital"
        ]
        .sum()
    )

    logger.info(
        "Total portfolio weight : %.4f",
        total_weight
    )

    logger.info(
        "Total capital allocated : %.2f",
        total_capital
    )

    # --------------------------------------------------------
    # Current feature averages
    # --------------------------------------------------------

    if "weighted_return_21d" in portfolio.columns:

        logger.info(
            "Weighted 21-day return : %.4f",
            portfolio[
                "weighted_return_21d"
            ].sum()
        )

    if "weighted_return_63d" in portfolio.columns:

        logger.info(
            "Weighted 63-day return : %.4f",
            portfolio[
                "weighted_return_63d"
            ].sum()
        )

    if "weighted_volatility_60d" in portfolio.columns:

        logger.info(
            "Weighted 60d volatility: %.4f",
            portfolio[
                "weighted_volatility_60d"
            ].sum()
        )

    if "weighted_sharpe_252d" in portfolio.columns:

        logger.info(
            "Weighted Sharpe        : %.4f",
            portfolio[
                "weighted_sharpe_252d"
            ].sum()
        )

    logger.info(
        "=============================================================="
    )


# ============================================================
# PREPARE OUTPUT COLUMNS
# ============================================================

def prepare_output(
    portfolio: pd.DataFrame
) -> pd.DataFrame:

    preferred_columns = [
        "final_rank",
        "selection_rank",
        "bgsto_rank",
        "date",
        "symbol",

        "portfolio_weight",
        "portfolio_weight_percent",
        "allocated_capital",

        "final_stock_score",
        "score_category",

        "buy_probability",
        "buy_confidence_percent",
        "confidence_category",

        "bgsto_stock_score",
        "bgsto_fitness",

        "return_21d",
        "return_63d",
        "volatility_60d",
        "sharpe_252d",
        "rsi_14",

        "weighted_return_21d",
        "weighted_return_63d",
        "weighted_volatility_60d",
        "weighted_sharpe_252d"
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in portfolio.columns
    ]

    remaining_columns = [
        column
        for column in portfolio.columns
        if column not in available_columns
    ]

    return portfolio[
        available_columns
        + remaining_columns
    ]


# ============================================================
# SAVE PORTFOLIO
# ============================================================

def save_portfolio(
    portfolio: pd.DataFrame
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    portfolio.to_csv(
        PORTFOLIO_CSV,
        index=False
    )

    portfolio.to_parquet(
        PORTFOLIO_PARQUET,
        index=False
    )

    logger.info(
        "Portfolio CSV saved to:"
    )

    logger.info(
        "%s",
        PORTFOLIO_CSV
    )

    logger.info(
        "Portfolio Parquet saved to:"
    )

    logger.info(
        "%s",
        PORTFOLIO_PARQUET
    )


# ============================================================
# BUILD PORTFOLIO
# ============================================================

def build_portfolio() -> pd.DataFrame:

    logger.info(
        "=============================================================="
    )

    logger.info(
        "STARTING PORTFOLIO CONSTRUCTION"
    )

    logger.info(
        "=============================================================="
    )

    # --------------------------------------------------------
    # Step 1: Load rankings
    # --------------------------------------------------------

    rankings = (
        load_rankings()
    )

    # --------------------------------------------------------
    # Step 2: Validate
    # --------------------------------------------------------

    validate_rankings(
        rankings
    )

    # --------------------------------------------------------
    # Step 3: Select top stocks
    # --------------------------------------------------------

    portfolio = (
        select_portfolio_stocks(
            rankings
        )
    )

    # --------------------------------------------------------
    # Step 4: Assign weights
    # --------------------------------------------------------

    if WEIGHTING_METHOD == "equal":

        portfolio = (
            assign_equal_weights(
                portfolio
            )
        )

    else:

        raise ValueError(
            f"Unsupported weighting method: "
            f"{WEIGHTING_METHOD}"
        )

    # --------------------------------------------------------
    # Step 5: Allocate capital
    # --------------------------------------------------------

    portfolio = (
        allocate_capital(
            portfolio
        )
    )

    # --------------------------------------------------------
    # Step 6: Add supporting metrics
    # --------------------------------------------------------

    portfolio = (
        calculate_portfolio_metrics(
            portfolio
        )
    )

    # --------------------------------------------------------
    # Step 7: Prepare final columns
    # --------------------------------------------------------

    portfolio = (
        prepare_output(
            portfolio
        )
    )

    # --------------------------------------------------------
    # Step 8: Save
    # --------------------------------------------------------

    save_portfolio(
        portfolio
    )

    # --------------------------------------------------------
    # Step 9: Display report
    # --------------------------------------------------------

    display_portfolio_summary(
        portfolio
    )

    logger.info(
        "Portfolio construction completed successfully."
    )

    return portfolio


# ============================================================
# MAIN
# ============================================================

def main():

    build_portfolio()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()