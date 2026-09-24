"""
Tests for Agent 5 - Portfolio Rebalancing Agent
===============================================================================

Validates the final stages of:

    Agent 5 - Portfolio Rebalancing Agent

Components covered
------------------
1. Rebalancing environment
2. Final portfolio construction
3. Portfolio output persistence
4. Portfolio evaluation
5. Agent-4 -> Agent-5 integration
6. Final Agent-5 summary


Current validated pipeline
--------------------------
Agent-4 actions:

    BUY  = 1
    HOLD = 7
    SELL = 2

Final portfolio:

    BHARATFORG   ~21.240505%
    HBLENGINE    ~19.171430%
    CGCL         ~14.827836%
    CREDITACC    ~12.106820%
    360ONE       ~10.464289%
    MRPL          ~9.830546%
    IDEA          ~6.957305%
    HINDCOPPER    ~5.401269%
    GVT&D          0.000000%
    HONASA         0.000000%


IMPORTANT
---------
These tests validate the implemented Agent-5 pipeline.

They do not retrain Agent 4 and do not rerun the full five-agent system.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATH SETUP
# =============================================================================

TEST_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

AGENT_ROOT = (
    TEST_DIR
    .parent
)

PROJECT_ROOT = (
    AGENT_ROOT
    .parent
)

if str(AGENT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(AGENT_ROOT),
    )

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from config import config as cfg


# =============================================================================
# LOCAL PATHS
# =============================================================================

REBALANCING_TRANSITION_CSV = (
    cfg.PROCESSED_DATA_DIR
    / "rebalancing_transition.csv"
)

REBALANCING_ENVIRONMENT_SUMMARY = (
    cfg.REPORT_DIR
    / "rebalancing_environment_summary.json"
)


# =============================================================================
# HELPERS
# =============================================================================

def load_final_portfolio() -> pd.DataFrame:
    """
    Load final Agent-5 portfolio.
    """

    if cfg.REBALANCED_PORTFOLIO_PARQUET.exists():

        dataframe = pd.read_parquet(
            cfg.REBALANCED_PORTFOLIO_PARQUET
        )

    elif cfg.REBALANCED_PORTFOLIO_CSV.exists():

        dataframe = pd.read_csv(
            cfg.REBALANCED_PORTFOLIO_CSV,
            low_memory=False,
        )

    else:

        raise FileNotFoundError(
            "Final Agent-5 portfolio not found."
        )

    return dataframe


def load_final_allocation() -> pd.DataFrame:
    """
    Load clean final allocation table.
    """

    if not cfg.FINAL_PORTFOLIO_ALLOCATION_CSV.exists():

        raise FileNotFoundError(
            "Final portfolio allocation file not found."
        )

    return pd.read_csv(
        cfg.FINAL_PORTFOLIO_ALLOCATION_CSV,
        low_memory=False,
    )


def load_transition() -> pd.DataFrame:
    """
    Load rebalancing environment transition.
    """

    if not REBALANCING_TRANSITION_CSV.exists():

        raise FileNotFoundError(
            "Rebalancing transition file not found."
        )

    return pd.read_csv(
        REBALANCING_TRANSITION_CSV,
        low_memory=False,
    )


def load_json(
    path: Path,
) -> dict:
    """
    Load JSON report.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Report not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# =============================================================================
# OUTPUT FILE TESTS
# =============================================================================

def test_final_output_files_exist():
    """
    Final Agent-5 portfolio files must exist.
    """

    assert cfg.REBALANCED_PORTFOLIO_CSV.exists()

    assert cfg.REBALANCED_PORTFOLIO_PARQUET.exists()

    assert cfg.FINAL_PORTFOLIO_ALLOCATION_CSV.exists()

    assert cfg.AGENT5_SUMMARY.exists()


def test_rebalancing_environment_outputs_exist():
    """
    Rebalancing transition and environment report must exist.
    """

    assert REBALANCING_TRANSITION_CSV.exists()

    assert REBALANCING_ENVIRONMENT_SUMMARY.exists()


def test_evaluation_outputs_exist():
    """
    Evaluation report must exist.
    """

    assert cfg.EVALUATION_SUMMARY.exists()


def test_integration_validation_output_exists():
    """
    Agent-4 -> Agent-5 validation report must exist.
    """

    assert cfg.AGENT4_AGENT5_VALIDATION.exists()


# =============================================================================
# FINAL PORTFOLIO STRUCTURE
# =============================================================================

def test_final_portfolio_not_empty():
    """
    Final portfolio must contain stocks.
    """

    dataframe = load_final_portfolio()

    assert not dataframe.empty


def test_final_portfolio_has_ten_stocks():
    """
    Agent 5 must preserve all ten Agent-4 stocks.
    """

    dataframe = load_final_portfolio()

    assert len(
        dataframe
    ) == 10

    assert dataframe[
        "symbol"
    ].nunique() == 10


def test_final_portfolio_required_columns():
    """
    Core final portfolio columns must exist.
    """

    dataframe = load_final_portfolio()

    required_columns = {
        "symbol",
        "trade_action",
        "current_weight",
        "target_weight",
        "weight_change",
        "rebalance_direction",
        "final_weight",
        "final_allocation_percent",
        "portfolio_status",
        "final_decision",
    }

    assert required_columns.issubset(
        dataframe.columns
    )


# =============================================================================
# FINAL WEIGHT TESTS
# =============================================================================

def test_final_weights_are_finite():
    """
    Final weights must not contain NaN or infinity.
    """

    dataframe = load_final_portfolio()

    weights = pd.to_numeric(
        dataframe[
            "final_weight"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    assert np.isfinite(
        weights
    ).all()


def test_final_weights_are_nonnegative():
    """
    Short selling is disabled.
    """

    dataframe = load_final_portfolio()

    weights = dataframe[
        "final_weight"
    ].to_numpy(
        dtype=float
    )

    assert (
        weights
        >=
        -cfg.FLOAT_TOLERANCE
    ).all()


def test_final_weights_do_not_exceed_maximum():
    """
    Final weights must respect configured maximum.
    """

    dataframe = load_final_portfolio()

    weights = dataframe[
        "final_weight"
    ].to_numpy(
        dtype=float
    )

    assert (
        weights
        <=
        (
            cfg.MAX_WEIGHT
            +
            cfg.FLOAT_TOLERANCE
        )
    ).all()


def test_final_weights_sum_to_one():
    """
    Final Agent-5 portfolio must be fully invested.
    """

    dataframe = load_final_portfolio()

    total_weight = float(
        dataframe[
            "final_weight"
        ]
        .sum()
    )

    assert np.isclose(
        total_weight,
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    )


def test_final_weights_match_target_weights():
    """
    Final portfolio must equal validated RL/MPT target state.
    """

    dataframe = load_final_portfolio()

    assert np.allclose(
        dataframe[
            "final_weight"
        ]
        .to_numpy(
            dtype=float
        ),
        dataframe[
            "target_weight"
        ]
        .to_numpy(
            dtype=float
        ),
        atol=cfg.FLOAT_TOLERANCE,
    )


# =============================================================================
# AGENT-4 ACTION TESTS
# =============================================================================

def test_agent4_action_distribution_preserved():
    """
    Agent-4 action distribution must remain:

        BUY  = 1
        HOLD = 7
        SELL = 2
    """

    dataframe = load_final_portfolio()

    actions = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .value_counts()
        .to_dict()
    )

    assert actions.get(
        "BUY",
        0,
    ) == 1

    assert actions.get(
        "HOLD",
        0,
    ) == 7

    assert actions.get(
        "SELL",
        0,
    ) == 2


def test_sell_stocks_have_zero_final_weight():
    """
    Agent-4 SELL stocks must exit final portfolio.
    """

    dataframe = load_final_portfolio()

    sell_mask = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .eq(
            cfg.ACTION_SELL
        )
    )

    assert int(
        sell_mask.sum()
    ) == 2

    assert np.allclose(
        dataframe.loc[
            sell_mask,
            "final_weight",
        ]
        .to_numpy(
            dtype=float
        ),
        cfg.SELL_TARGET_WEIGHT,
        atol=cfg.FLOAT_TOLERANCE,
    )


def test_sell_stocks_have_exit_status():
    """
    SELL stocks must be marked EXIT.
    """

    dataframe = load_final_portfolio()

    sell_mask = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .eq(
            cfg.ACTION_SELL
        )
    )

    assert (
        dataframe.loc[
            sell_mask,
            "portfolio_status",
        ]
        .astype(str)
        .str.upper()
        .eq(
            "EXIT"
        )
    ).all()


def test_sell_stocks_have_exit_decision():
    """
    Agent-4 SELL stocks must produce EXIT_POSITION.
    """

    dataframe = load_final_portfolio()

    sell_mask = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .eq(
            cfg.ACTION_SELL
        )
    )

    assert (
        dataframe.loc[
            sell_mask,
            "final_decision",
        ]
        .astype(str)
        .str.upper()
        .eq(
            "EXIT_POSITION"
        )
    ).all()


# =============================================================================
# CURRENT KNOWN ACTION SYMBOLS
# =============================================================================

def test_current_buy_symbol_is_hblengine():
    """
    Current validated Agent-4 BUY stock is HBLENGINE.
    """

    dataframe = load_final_portfolio()

    buy_symbols = set(
        dataframe.loc[
            dataframe[
                "trade_action"
            ]
            .astype(str)
            .str.upper()
            .eq(
                cfg.ACTION_BUY
            ),
            "symbol",
        ]
        .astype(str)
    )

    assert buy_symbols == {
        "HBLENGINE",
    }


def test_current_sell_symbols_are_correct():
    """
    Current validated Agent-4 SELL stocks:

        HONASA
        GVT&D
    """

    dataframe = load_final_portfolio()

    sell_symbols = set(
        dataframe.loc[
            dataframe[
                "trade_action"
            ]
            .astype(str)
            .str.upper()
            .eq(
                cfg.ACTION_SELL
            ),
            "symbol",
        ]
        .astype(str)
    )

    assert sell_symbols == {
        "HONASA",
        "GVT&D",
    }


# =============================================================================
# ACTIVE PORTFOLIO
# =============================================================================

def test_eight_active_assets_remain():
    """
    Two SELL exits leave eight active assets.
    """

    dataframe = load_final_portfolio()

    active_mask = (
        dataframe[
            "final_weight"
        ]
        .to_numpy(
            dtype=float
        )
        >
        cfg.FLOAT_TOLERANCE
    )

    assert int(
        active_mask.sum()
    ) == 8


def test_active_assets_are_not_agent4_sell():
    """
    No SELL stock may remain active.
    """

    dataframe = load_final_portfolio()

    active = dataframe.loc[
        dataframe[
            "final_weight"
        ]
        >
        cfg.FLOAT_TOLERANCE
    ]

    assert (
        active[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        !=
        cfg.ACTION_SELL
    ).all()


# =============================================================================
# CURRENT VALIDATED FINAL ALLOCATION
# =============================================================================

def test_current_final_allocation_matches_pipeline():
    """
    Validate current known final Agent-5 weights.

    A small tolerance is used because CSV/parquet serialization can create
    insignificant floating-point differences.
    """

    dataframe = (
        load_final_portfolio()
        .set_index(
            "symbol"
        )
    )

    expected = {
        "BHARATFORG":
            0.21240505,

        "HBLENGINE":
            0.19171430,

        "CGCL":
            0.14827836,

        "CREDITACC":
            0.12106820,

        "360ONE":
            0.10464289,

        "MRPL":
            0.09830546,

        "IDEA":
            0.06957305,

        "HINDCOPPER":
            0.05401269,

        "GVT&D":
            0.0,

        "HONASA":
            0.0,
    }

    assert set(
        dataframe.index
    ) == set(
        expected.keys()
    )

    for symbol, expected_weight in expected.items():

        actual_weight = float(
            dataframe.loc[
                symbol,
                "final_weight",
            ]
        )

        assert np.isclose(
            actual_weight,
            expected_weight,
            atol=1e-6,
        )


# =============================================================================
# REBALANCING TRANSITION
# =============================================================================

def test_transition_has_ten_stocks():
    """
    Rebalancing environment must preserve complete universe.
    """

    dataframe = load_transition()

    assert len(
        dataframe
    ) == 10

    assert dataframe[
        "symbol"
    ].nunique() == 10


def test_transition_weight_identity():
    """
    weight_change must equal:

        target_weight - current_weight
    """

    dataframe = load_transition()

    expected = (
        dataframe[
            "target_weight"
        ].to_numpy(
            dtype=float
        )
        -
        dataframe[
            "current_weight"
        ].to_numpy(
            dtype=float
        )
    )

    actual = dataframe[
        "weight_change"
    ].to_numpy(
        dtype=float
    )

    assert np.allclose(
        actual,
        expected,
        atol=cfg.FLOAT_TOLERANCE,
    )


def test_transition_current_weights_sum_to_one():
    """
    Provisional portfolio must be normalized.
    """

    dataframe = load_transition()

    assert np.isclose(
        dataframe[
            "current_weight"
        ]
        .sum(),
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    )


def test_transition_target_weights_sum_to_one():
    """
    Target portfolio must be normalized.
    """

    dataframe = load_transition()

    assert np.isclose(
        dataframe[
            "target_weight"
        ]
        .sum(),
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    )


def test_rebalance_directions_are_valid():
    """
    Rebalancing directions must be BUY/SELL/HOLD.
    """

    dataframe = load_transition()

    assert dataframe[
        "rebalance_direction"
    ].isin(
        [
            "BUY",
            "SELL",
            "HOLD",
        ]
    ).all()


# =============================================================================
# FINAL ALLOCATION TABLE
# =============================================================================

def test_final_allocation_table_not_empty():
    """
    Clean final allocation CSV must contain results.
    """

    dataframe = (
        load_final_allocation()
    )

    assert not dataframe.empty


def test_final_allocation_table_has_ten_stocks():
    """
    Presentation allocation should preserve full stock universe.
    """

    dataframe = (
        load_final_allocation()
    )

    assert len(
        dataframe
    ) == 10

    assert dataframe[
        "symbol"
    ].nunique() == 10


# =============================================================================
# AGENT-5 SUMMARY
# =============================================================================

def test_agent5_summary_status_complete():
    """
    Final Agent-5 report must show COMPLETE.
    """

    report = load_json(
        cfg.AGENT5_SUMMARY
    )

    assert (
        str(
            report.get(
                "status",
                "",
            )
        )
        .upper()
        ==
        "COMPLETE"
    )


def test_agent5_summary_reports_eight_active_assets():
    """
    Final report should show 8 active and 2 exited stocks.
    """

    report = load_json(
        cfg.AGENT5_SUMMARY
    )

    assert int(
        report[
            "active_assets"
        ]
    ) == 8

    assert int(
        report[
            "exited_assets"
        ]
    ) == 2


def test_agent5_summary_weight_sum_is_one():
    """
    Agent-5 report must record fully invested portfolio.
    """

    report = load_json(
        cfg.AGENT5_SUMMARY
    )

    assert np.isclose(
        float(
            report[
                "final_weight_sum"
            ]
        ),
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    )


def test_agent5_summary_validation_all_passed():
    """
    Every final portfolio validation check must pass.
    """

    report = load_json(
        cfg.AGENT5_SUMMARY
    )

    validation = report[
        "validation"
    ]

    assert validation

    assert all(
        bool(
            value
        )
        for value
        in validation.values()
    )


# =============================================================================
# ENVIRONMENT REPORT
# =============================================================================

def test_environment_summary_status_complete():
    """
    Rebalancing environment must be complete.
    """

    report = load_json(
        REBALANCING_ENVIRONMENT_SUMMARY
    )

    assert (
        str(
            report.get(
                "status",
                "",
            )
        )
        .upper()
        ==
        "COMPLETE"
    )


def test_environment_turnover_is_valid():
    """
    Rebalancing turnover must be finite and non-negative.
    """

    report = load_json(
        REBALANCING_ENVIRONMENT_SUMMARY
    )

    turnover = float(
        report[
            "turnover"
        ]
    )

    assert np.isfinite(
        turnover
    )

    assert turnover >= 0


def test_environment_validation_all_passed():
    """
    Every environment validation check must pass.
    """

    report = load_json(
        REBALANCING_ENVIRONMENT_SUMMARY
    )

    validation = report[
        "validation"
    ]

    assert all(
        bool(
            value
        )
        for value
        in validation.values()
    )


# =============================================================================
# EVALUATION REPORT
# =============================================================================

def test_evaluation_summary_status_complete():
    """
    Portfolio evaluator must have completed.
    """

    report = load_json(
        cfg.EVALUATION_SUMMARY
    )

    assert (
        str(
            report.get(
                "status",
                "",
            )
        )
        .upper()
        ==
        "COMPLETE"
    )


def test_evaluation_has_105_observations():
    """
    Current Agent-5 diagnostic window contains 105 aligned returns.
    """

    report = load_json(
        cfg.EVALUATION_SUMMARY
    )

    assert int(
        report[
            "return_observations"
        ]
    ) == 105


def test_evaluation_reports_eight_active_assets():
    """
    Evaluation must recognize eight active portfolio assets.
    """

    report = load_json(
        cfg.EVALUATION_SUMMARY
    )

    assert int(
        report[
            "active_assets"
        ]
    ) == 8


def test_evaluation_validation_all_passed():
    """
    All evaluator checks must pass.
    """

    report = load_json(
        cfg.EVALUATION_SUMMARY
    )

    validation = report[
        "validation"
    ]

    assert validation

    assert all(
        bool(
            value
        )
        for value
        in validation.values()
    )


# =============================================================================
# AGENT-4 -> AGENT-5 INTEGRATION
# =============================================================================

def test_integration_report_status_pass():
    """
    Agent-4 -> Agent-5 integration must be PASS.
    """

    report = load_json(
        cfg.AGENT4_AGENT5_VALIDATION
    )

    assert (
        str(
            report.get(
                "status",
                "",
            )
        )
        .upper()
        ==
        "PASS"
    )


def test_integration_preserves_ten_stocks():
    """
    Ten stocks must pass from Agent 4 to Agent 5.
    """

    report = load_json(
        cfg.AGENT4_AGENT5_VALIDATION
    )

    assert int(
        report[
            "agent4_stocks"
        ]
    ) == 10

    assert int(
        report[
            "agent5_stocks"
        ]
    ) == 10


def test_integration_validation_all_passed():
    """
    Every Agent-4 -> Agent-5 integration check must pass.
    """

    report = load_json(
        cfg.AGENT4_AGENT5_VALIDATION
    )

    validation = report[
        "validation"
    ]

    assert validation

    assert all(
        bool(
            value
        )
        for value
        in validation.values()
    )


# =============================================================================
# REBALANCE DATE
# =============================================================================

def test_integration_rebalance_date_matches_config():
    """
    Final Agent-5 rebalance date must match configuration.
    """

    report = load_json(
        cfg.AGENT4_AGENT5_VALIDATION
    )

    assert (
        str(
            report[
                "rebalance_date"
            ]
        )
        ==
        str(
            cfg.FINAL_REBALANCE_DATE
        )
    )