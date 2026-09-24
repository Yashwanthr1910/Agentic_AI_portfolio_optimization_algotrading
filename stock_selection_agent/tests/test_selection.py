"""
test_selection.py

Tests for the stock-selection pipeline.

Run:
    python -m pytest tests/test_selection.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DECISION_TREE_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "decision_tree_predictions.parquet"
)

BGSTO_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "bgsto_selected_stocks.csv"
)

SELECTED_STOCKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "selected_stocks.csv"
)

RANKINGS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "rankings.csv"
)

PORTFOLIO_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "portfolio.csv"
)


# ============================================================
# LOAD HELPERS
# ============================================================

def load_decision_tree_predictions():

    assert DECISION_TREE_FILE.exists(), (
        f"Decision Tree prediction file not found: "
        f"{DECISION_TREE_FILE}"
    )

    return pd.read_parquet(
        DECISION_TREE_FILE
    )


def load_bgsto():

    assert BGSTO_FILE.exists(), (
        f"BGSTO output file not found: {BGSTO_FILE}"
    )

    return pd.read_csv(
        BGSTO_FILE
    )


def load_selected_stocks():

    assert SELECTED_STOCKS_FILE.exists(), (
        f"Selected stocks file not found: "
        f"{SELECTED_STOCKS_FILE}"
    )

    return pd.read_csv(
        SELECTED_STOCKS_FILE
    )


def load_rankings():

    assert RANKINGS_FILE.exists(), (
        f"Rankings file not found: {RANKINGS_FILE}"
    )

    return pd.read_csv(
        RANKINGS_FILE
    )


def load_portfolio():

    assert PORTFOLIO_FILE.exists(), (
        f"Portfolio file not found: {PORTFOLIO_FILE}"
    )

    return pd.read_csv(
        PORTFOLIO_FILE
    )


# ============================================================
# TEST 1: DECISION TREE OUTPUT EXISTS
# ============================================================

def test_decision_tree_output_exists():

    assert DECISION_TREE_FILE.exists()


# ============================================================
# TEST 2: DECISION TREE OUTPUT NOT EMPTY
# ============================================================

def test_decision_tree_output_not_empty():

    df = load_decision_tree_predictions()

    assert len(df) > 0, (
        "Decision Tree output is empty."
    )


# ============================================================
# TEST 3: DECISION TREE REQUIRED COLUMNS
# ============================================================

def test_decision_tree_required_columns():

    df = load_decision_tree_predictions()

    expected = [
        "date",
        "symbol",
        "predicted_target",
        "buy_probability",
        "dataset_split",
    ]

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    assert not missing, (
        f"Missing Decision Tree columns: {missing}"
    )


# ============================================================
# TEST 4: BUY PROBABILITY RANGE
# ============================================================

def test_buy_probability_range():

    df = load_decision_tree_predictions()

    probabilities = (
        df["buy_probability"]
        .dropna()
    )

    assert not probabilities.empty, (
        "No valid buy_probability values found."
    )

    invalid = (
        (probabilities < 0)
        |
        (probabilities > 1)
    ).sum()

    assert invalid == 0, (
        f"Found {invalid} buy probabilities "
        "outside the range [0, 1]."
    )


# ============================================================
# TEST 5: PREDICTED TARGET IS BINARY
# ============================================================

def test_predicted_target_binary():

    df = load_decision_tree_predictions()

    unique_values = set(
        df[
            "predicted_target"
        ]
        .dropna()
        .unique()
    )

    assert unique_values.issubset(
        {
            0,
            1,
        }
    ), (
        f"Unexpected predicted_target values: "
        f"{unique_values}"
    )


# ============================================================
# TEST 6: DATASET SPLITS VALID
# ============================================================

def test_dataset_splits_valid():

    df = load_decision_tree_predictions()

    valid_splits = {
        "train",
        "backtest",
        "validation",
    }

    actual_splits = set(
        df[
            "dataset_split"
        ]
        .dropna()
        .unique()
    )

    assert actual_splits.issubset(
        valid_splits
    ), (
        f"Unexpected dataset splits: "
        f"{actual_splits}"
    )


# ============================================================
# TEST 7: BGSTO OUTPUT EXISTS
# ============================================================

def test_bgsto_output_exists():

    assert BGSTO_FILE.exists()


# ============================================================
# TEST 8: BGSTO OUTPUT NOT EMPTY
# ============================================================

def test_bgsto_output_not_empty():

    df = load_bgsto()

    assert len(df) > 0, (
        "BGSTO output is empty."
    )


# ============================================================
# TEST 9: BGSTO REQUIRED COLUMNS
# ============================================================

def test_bgsto_required_columns():

    df = load_bgsto()

    expected = [
        "bgsto_rank",
        "date",
        "symbol",
        "buy_probability",
        "bgsto_stock_score",
        "bgsto_fitness",
    ]

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    assert not missing, (
        f"Missing BGSTO columns: {missing}"
    )


# ============================================================
# TEST 10: BGSTO STOCK COUNT
# ============================================================

def test_bgsto_stock_count():

    df = load_bgsto()

    stock_count = (
        df["symbol"]
        .nunique()
    )

    assert stock_count >= 10, (
        f"BGSTO selected too few stocks: "
        f"{stock_count}"
    )

    assert stock_count <= 20, (
        f"BGSTO selected too many stocks: "
        f"{stock_count}"
    )


# ============================================================
# TEST 11: BGSTO SYMBOLS UNIQUE
# ============================================================

def test_bgsto_symbols_unique():

    df = load_bgsto()

    duplicate_count = (
        df["symbol"]
        .duplicated()
        .sum()
    )

    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicate "
        "symbols in BGSTO output."
    )


# ============================================================
# TEST 12: BGSTO SCORES FINITE
# ============================================================

def test_bgsto_scores_are_finite():

    df = load_bgsto()

    numeric_columns = [
        "buy_probability",
        "bgsto_stock_score",
        "bgsto_fitness",
    ]

    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        assert values.notna().all(), (
            f"Invalid numeric values found in {column}."
        )

        assert np.isfinite(
            values
        ).all(), (
            f"Infinite values found in {column}."
        )


# ============================================================
# TEST 13: SELECTED STOCKS OUTPUT EXISTS
# ============================================================

def test_selected_stocks_output_exists():

    assert SELECTED_STOCKS_FILE.exists()


# ============================================================
# TEST 14: SELECTED STOCKS REQUIRED COLUMNS
# ============================================================

def test_selected_stocks_required_columns():

    df = load_selected_stocks()

    expected = [
        "selection_rank",
        "symbol",
        "selected",
        "buy_probability",
        "bgsto_stock_score",
    ]

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    assert not missing, (
        f"Missing selected-stock columns: {missing}"
    )


# ============================================================
# TEST 15: ALL FINAL STOCKS MARKED SELECTED
# ============================================================

def test_all_final_stocks_marked_selected():

    df = load_selected_stocks()

    assert (
        df["selected"] == 1
    ).all(), (
        "Some final stocks are not marked selected=1."
    )


# ============================================================
# TEST 16: SELECTION RANK UNIQUE
# ============================================================

def test_selection_rank_unique():

    df = load_selected_stocks()

    assert df[
        "selection_rank"
    ].is_unique, (
        "Duplicate selection_rank values found."
    )


# ============================================================
# TEST 17: SELECTION RANK STARTS AT ONE
# ============================================================

def test_selection_rank_starts_at_one():

    df = load_selected_stocks()

    minimum_rank = (
        df[
            "selection_rank"
        ]
        .min()
    )

    assert minimum_rank == 1, (
        f"Expected minimum selection rank 1, "
        f"found {minimum_rank}."
    )


# ============================================================
# TEST 18: RANKINGS OUTPUT EXISTS
# ============================================================

def test_rankings_output_exists():

    assert RANKINGS_FILE.exists()


# ============================================================
# TEST 19: RANKINGS REQUIRED COLUMNS
# ============================================================

def test_rankings_required_columns():

    df = load_rankings()

    expected = [
        "final_rank",
        "symbol",
        "final_stock_score",
        "buy_probability",
    ]

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    assert not missing, (
        f"Missing ranking columns: {missing}"
    )


# ============================================================
# TEST 20: FINAL SCORE RANGE
# ============================================================

def test_final_stock_score_range():

    df = load_rankings()

    scores = (
        df[
            "final_stock_score"
        ]
        .dropna()
    )

    assert not scores.empty

    invalid = (
        (scores < 0)
        |
        (scores > 100)
    ).sum()

    assert invalid == 0, (
        f"Found {invalid} final scores "
        "outside the range 0-100."
    )


# ============================================================
# TEST 21: FINAL RANK UNIQUE
# ============================================================

def test_final_rank_unique():

    df = load_rankings()

    assert df[
        "final_rank"
    ].is_unique, (
        "Duplicate final_rank values found."
    )


# ============================================================
# TEST 22: FINAL RANK ORDER
# ============================================================

def test_final_rank_order():

    df = load_rankings().sort_values(
        "final_rank"
    )

    expected_ranks = list(
        range(
            1,
            len(df) + 1
        )
    )

    actual_ranks = (
        df[
            "final_rank"
        ]
        .astype(int)
        .tolist()
    )

    assert actual_ranks == expected_ranks, (
        "Final rankings are not consecutive."
    )


# ============================================================
# TEST 23: SAME SYMBOLS IN SELECTION AND RANKINGS
# ============================================================

def test_selected_and_ranked_symbols_match():

    selected = load_selected_stocks()

    rankings = load_rankings()

    selected_symbols = set(
        selected["symbol"]
    )

    ranked_symbols = set(
        rankings["symbol"]
    )

    assert selected_symbols == ranked_symbols, (
        "Selected stock symbols and ranking "
        "symbols do not match."
    )


# ============================================================
# TEST 24: PORTFOLIO OUTPUT EXISTS
# ============================================================

def test_portfolio_output_exists():

    assert PORTFOLIO_FILE.exists()


# ============================================================
# TEST 25: PORTFOLIO REQUIRED COLUMNS
# ============================================================

def test_portfolio_required_columns():

    df = load_portfolio()

    expected = [
        "symbol",
        "portfolio_weight",
        "allocated_capital",
    ]

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    assert not missing, (
        f"Missing portfolio columns: {missing}"
    )


# ============================================================
# TEST 26: PORTFOLIO NOT EMPTY
# ============================================================

def test_portfolio_not_empty():

    df = load_portfolio()

    assert len(df) > 0, (
        "Portfolio is empty."
    )


# ============================================================
# TEST 27: PORTFOLIO MAXIMUM SIZE
# ============================================================

def test_portfolio_size():

    df = load_portfolio()

    assert len(df) <= 10, (
        f"Portfolio contains more than 10 stocks: "
        f"{len(df)}"
    )

    assert len(df) > 0


# ============================================================
# TEST 28: PORTFOLIO SYMBOLS UNIQUE
# ============================================================

def test_portfolio_symbols_unique():

    df = load_portfolio()

    assert df[
        "symbol"
    ].is_unique, (
        "Duplicate portfolio symbols found."
    )


# ============================================================
# TEST 29: PORTFOLIO WEIGHTS POSITIVE
# ============================================================

def test_portfolio_weights_positive():

    df = load_portfolio()

    weights = pd.to_numeric(
        df[
            "portfolio_weight"
        ],
        errors="coerce"
    )

    assert weights.notna().all(), (
        "Invalid portfolio weights found."
    )

    assert (
        weights > 0
    ).all(), (
        "Portfolio contains zero or negative weights."
    )


# ============================================================
# TEST 30: PORTFOLIO WEIGHTS SUM TO ONE
# ============================================================

def test_portfolio_weights_sum_to_one():

    df = load_portfolio()

    weights = (
        pd.to_numeric(
            df[
                "portfolio_weight"
            ],
            errors="coerce"
        )
    )

    total_weight = (
        weights.sum()
    )

    assert np.isclose(
        total_weight,
        1.0,
        atol=1e-8
    ), (
        f"Portfolio weights sum to "
        f"{total_weight}, expected 1.0."
    )


# ============================================================
# TEST 31: ALLOCATED CAPITAL POSITIVE
# ============================================================

def test_allocated_capital_positive():

    df = load_portfolio()

    allocated = pd.to_numeric(
        df[
            "allocated_capital"
        ],
        errors="coerce"
    )

    assert allocated.notna().all()

    assert (
        allocated > 0
    ).all(), (
        "Non-positive allocated capital found."
    )


# ============================================================
# TEST 32: PORTFOLIO IS SUBSET OF RANKINGS
# ============================================================

def test_portfolio_symbols_in_rankings():

    portfolio = load_portfolio()

    rankings = load_rankings()

    portfolio_symbols = set(
        portfolio[
            "symbol"
        ]
    )

    ranking_symbols = set(
        rankings[
            "symbol"
        ]
    )

    assert portfolio_symbols.issubset(
        ranking_symbols
    ), (
        "Portfolio contains symbols "
        "not present in rankings."
    )


# ============================================================
# TEST 33: PORTFOLIO SHOULD USE TOP RANKED STOCKS
# ============================================================

def test_portfolio_uses_top_ranked_stocks():

    portfolio = load_portfolio()

    rankings = (
        load_rankings()
        .sort_values(
            "final_rank"
        )
    )

    n = len(
        portfolio
    )

    expected_symbols = set(
        rankings
        .head(n)[
            "symbol"
        ]
    )

    actual_symbols = set(
        portfolio[
            "symbol"
        ]
    )

    assert actual_symbols == expected_symbols, (
        "Portfolio does not contain the "
        "top-ranked stocks."
    )