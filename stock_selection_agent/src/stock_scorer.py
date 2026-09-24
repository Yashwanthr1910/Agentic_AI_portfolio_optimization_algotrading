"""
stock_scorer.py

Stock Scoring and Ranking Module

Purpose:
    Rank the stocks already selected by the
    Decision Tree + BGSTO stock-selection pipeline.

Input:
    data/outputs/selected_stocks.csv

Output:
    outputs/rankings.csv
    outputs/rankings.parquet

Important:
    This module DOES NOT select new stocks.

    Decision Tree + BGSTO already performed the actual
    stock selection.

    stock_scorer.py only:
        - converts BGSTO score to a 0-100 score
        - assigns final ranking
        - assigns confidence category
        - prepares rankings.csv

Pipeline:

    Decision Tree
          ↓
        BGSTO
          ↓
    selected_stocks.csv
          ↓
    stock_scorer.py
          ↓
      rankings.csv
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
    / "data"
    / "outputs"
    / "selected_stocks.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

RANKINGS_CSV = (
    OUTPUT_DIR
    / "rankings.csv"
)

RANKINGS_PARQUET = (
    OUTPUT_DIR
    / "rankings.parquet"
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
# LOAD SELECTED STOCKS
# ============================================================

def load_selected_stocks() -> pd.DataFrame:

    logger.info(
        "Loading selected stocks..."
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "selected_stocks.csv not found.\n"
            f"Expected location: {INPUT_FILE}\n\n"
            "Run stock_selector.py first."
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    if df.empty:

        raise ValueError(
            "selected_stocks.csv is empty."
        )

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"]
        )

    logger.info(
        "Loaded %d selected stocks.",
        len(df)
    )

    return df


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(
    df: pd.DataFrame
) -> None:

    logger.info(
        "Validating selected-stock data..."
    )

    required_columns = [
        "symbol",
        "buy_probability",
        "bgsto_stock_score"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Check duplicate symbols
    # --------------------------------------------------------

    duplicate_symbols = (
        df["symbol"]
        .duplicated()
        .sum()
    )

    if duplicate_symbols > 0:

        logger.warning(
            "%d duplicate symbols found.",
            duplicate_symbols
        )

    # --------------------------------------------------------
    # Check missing BGSTO scores
    # --------------------------------------------------------

    missing_scores = (
        df["bgsto_stock_score"]
        .isna()
        .sum()
    )

    if missing_scores > 0:

        raise ValueError(
            f"{missing_scores} stocks have missing "
            "BGSTO stock scores."
        )

    logger.info(
        "Input validation passed."
    )


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def min_max_score(
    series: pd.Series
) -> pd.Series:

    """
    Normalize a numeric series to a 0-100 score.
    """

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):

        return pd.Series(
            np.nan,
            index=series.index
        )

    if np.isclose(
        maximum,
        minimum
    ):

        return pd.Series(
            100.0,
            index=series.index
        )

    normalized = (
        (
            series - minimum
        )
        /
        (
            maximum - minimum
        )
    )

    return (
        normalized * 100
    )


# ============================================================
# CONFIDENCE CATEGORY
# ============================================================

def confidence_category(
    probability: float
) -> str:

    """
    Convert Decision Tree BUY probability into
    a human-readable confidence category.
    """

    if pd.isna(probability):

        return "Unknown"

    if probability >= 0.90:

        return "Very High"

    if probability >= 0.80:

        return "High"

    if probability >= 0.70:

        return "Moderate"

    if probability >= 0.60:

        return "Low"

    return "Very Low"


# ============================================================
# SCORE CATEGORY
# ============================================================

def score_category(
    score: float
) -> str:

    """
    Convert final score into a ranking category.
    """

    if pd.isna(score):

        return "Unknown"

    if score >= 80:

        return "Excellent"

    if score >= 60:

        return "Strong"

    if score >= 40:

        return "Good"

    if score >= 20:

        return "Moderate"

    return "Weak"


# ============================================================
# CALCULATE STOCK SCORES
# ============================================================

def calculate_scores(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Calculating final stock scores..."
    )

    scored = df.copy()

    # ========================================================
    # PRIMARY SCORE
    #
    # BGSTO has already incorporated the underlying
    # Decision Tree probability, return, Sharpe ratio,
    # volatility and optimizer fitness.
    #
    # Therefore we do NOT recombine those variables again
    # with arbitrary weights here.
    # ========================================================

    scored[
        "final_stock_score"
    ] = min_max_score(
        scored[
            "bgsto_stock_score"
        ]
    )

    # --------------------------------------------------------
    # Decision Tree confidence as percentage
    # --------------------------------------------------------

    scored[
        "buy_confidence_percent"
    ] = (
        scored[
            "buy_probability"
        ]
        * 100
    )

    # --------------------------------------------------------
    # Confidence label
    # --------------------------------------------------------

    scored[
        "confidence_category"
    ] = (
        scored[
            "buy_probability"
        ]
        .apply(
            confidence_category
        )
    )

    # --------------------------------------------------------
    # Score label
    # --------------------------------------------------------

    scored[
        "score_category"
    ] = (
        scored[
            "final_stock_score"
        ]
        .apply(
            score_category
        )
    )

    return scored


# ============================================================
# CREATE FINAL RANKING
# ============================================================

def create_ranking(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Creating final stock ranking..."
    )

    ranked = (
        df
        .sort_values(
            [
                "final_stock_score",
                "buy_probability"
            ],
            ascending=[
                False,
                False
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Create final ranking
    # --------------------------------------------------------

    ranked[
        "final_rank"
    ] = np.arange(
        1,
        len(ranked) + 1
    )

    return ranked


# ============================================================
# PREPARE OUTPUT COLUMNS
# ============================================================

def prepare_output(
    df: pd.DataFrame
) -> pd.DataFrame:

    preferred_columns = [
        "final_rank",
        "selection_rank",
        "bgsto_rank",
        "date",
        "symbol",
        "selected",

        # Final score
        "final_stock_score",
        "score_category",

        # Decision Tree
        "buy_probability",
        "buy_confidence_percent",
        "confidence_category",

        # BGSTO
        "bgsto_stock_score",
        "bgsto_fitness",

        # Supporting features
        "return_21d",
        "return_63d",
        "volatility_60d",
        "sharpe_252d",
        "rsi_14"
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    remaining_columns = [
        column
        for column in df.columns
        if column not in existing_columns
    ]

    return df[
        existing_columns
        + remaining_columns
    ]


# ============================================================
# DISPLAY REPORT
# ============================================================

def display_report(
    df: pd.DataFrame
) -> None:

    logger.info(
        ""
    )

    logger.info(
        "=============================================================="
    )

    logger.info(
        "                    STOCK RANKING REPORT"
    )

    logger.info(
        "=============================================================="
    )

    if "date" in df.columns:

        selection_date = (
            df["date"]
            .iloc[0]
        )

        logger.info(
            "Selection date : %s",
            (
                selection_date.date()
                if hasattr(
                    selection_date,
                    "date"
                )
                else selection_date
            )
        )

    logger.info(
        "Stocks ranked   : %d",
        len(df)
    )

    logger.info(
        "--------------------------------------------------------------"
    )

    display_columns = [
        "final_rank",
        "symbol",
        "final_stock_score",
        "score_category",
        "buy_probability",
        "bgsto_stock_score",
        "return_63d",
        "sharpe_252d"
    ]

    display_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    logger.info(
        "\n%s",
        df[
            display_columns
        ].to_string(
            index=False
        )
    )

    logger.info(
        "--------------------------------------------------------------"
    )

    logger.info(
        "Average final score : %.2f",
        df[
            "final_stock_score"
        ].mean()
    )

    logger.info(
        "Highest score       : %.2f",
        df[
            "final_stock_score"
        ].max()
    )

    logger.info(
        "Lowest score        : %.2f",
        df[
            "final_stock_score"
        ].min()
    )

    logger.info(
        "Average BUY prob.    : %.4f",
        df[
            "buy_probability"
        ].mean()
    )

    logger.info(
        "--------------------------------------------------------------"
    )

    top_stock = (
        df.iloc[0]
    )

    logger.info(
        "Top-ranked stock     : %s",
        top_stock[
            "symbol"
        ]
    )

    logger.info(
        "Top stock score      : %.2f",
        top_stock[
            "final_stock_score"
        ]
    )

    logger.info(
        "=============================================================="
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_rankings(
    df: pd.DataFrame
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    df.to_csv(
        RANKINGS_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Parquet
    # --------------------------------------------------------

    df.to_parquet(
        RANKINGS_PARQUET,
        index=False
    )

    logger.info(
        "Rankings CSV saved to:"
    )

    logger.info(
        "%s",
        RANKINGS_CSV
    )

    logger.info(
        "Rankings Parquet saved to:"
    )

    logger.info(
        "%s",
        RANKINGS_PARQUET
    )


# ============================================================
# STOCK SCORER
# ============================================================

def score_stocks() -> pd.DataFrame:

    logger.info(
        "=============================================================="
    )

    logger.info(
        "STARTING STOCK SCORER"
    )

    logger.info(
        "=============================================================="
    )

    # --------------------------------------------------------
    # Step 1
    # Load selected stocks
    # --------------------------------------------------------

    selected = (
        load_selected_stocks()
    )

    # --------------------------------------------------------
    # Step 2
    # Validate
    # --------------------------------------------------------

    validate_input(
        selected
    )

    # --------------------------------------------------------
    # Step 3
    # Calculate scores
    # --------------------------------------------------------

    scored = (
        calculate_scores(
            selected
        )
    )

    # --------------------------------------------------------
    # Step 4
    # Rank
    # --------------------------------------------------------

    ranked = (
        create_ranking(
            scored
        )
    )

    # --------------------------------------------------------
    # Step 5
    # Prepare output
    # --------------------------------------------------------

    ranked = (
        prepare_output(
            ranked
        )
    )

    # --------------------------------------------------------
    # Step 6
    # Save
    # --------------------------------------------------------

    save_rankings(
        ranked
    )

    # --------------------------------------------------------
    # Step 7
    # Report
    # --------------------------------------------------------

    display_report(
        ranked
    )

    logger.info(
        "Stock scoring completed successfully."
    )

    return ranked


# ============================================================
# MAIN
# ============================================================

def main():

    score_stocks()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()