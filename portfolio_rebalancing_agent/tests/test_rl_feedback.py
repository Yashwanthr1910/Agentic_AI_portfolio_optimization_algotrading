"""
Tests for Agent 5 - RL Feedback
===============================================================================

Validates:

    portfolio_rebalancing_agent/src/rl_feedback.py

The tests confirm:

1. RL feedback output files exist
2. MPT solution loads correctly
3. All ten stocks are preserved
4. Agent-4 BUY / HOLD / SELL distribution remains 1 / 7 / 2
5. RL-adjusted weights are finite
6. RL-adjusted weights are non-negative
7. RL-adjusted weights sum to one
8. SELL stocks receive exactly zero allocation
9. BUY/HOLD stocks remain eligible
10. RL feedback does not introduce arbitrary BUY/HOLD multipliers
11. Final target weights equal RL-adjusted weights
12. MPT and RL allocations remain consistent
13. RL feedback report is COMPLETE
14. All RL validation checks passed
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

from src.rl_feedback import (
    load_mpt_solution,
    load_mpt_report,
    apply_rl_feedback,
    validate_rl_feedback,
)


# =============================================================================
# HELPERS
# =============================================================================

def load_rl_adjusted_weights() -> pd.DataFrame:
    """
    Load saved RL-feedback output.
    """

    if not cfg.RL_ADJUSTED_WEIGHTS_CSV.exists():

        raise FileNotFoundError(
            f"RL-adjusted weights not found:\n"
            f"{cfg.RL_ADJUSTED_WEIGHTS_CSV}"
        )

    dataframe = pd.read_csv(
        cfg.RL_ADJUSTED_WEIGHTS_CSV,
        low_memory=False,
    )

    return dataframe


def load_rl_report() -> dict:
    """
    Load RL feedback summary.
    """

    if not cfg.RL_FEEDBACK_SUMMARY.exists():

        raise FileNotFoundError(
            f"RL feedback summary not found:\n"
            f"{cfg.RL_FEEDBACK_SUMMARY}"
        )

    with open(
        cfg.RL_FEEDBACK_SUMMARY,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# =============================================================================
# FILE TESTS
# =============================================================================

def test_rl_feedback_output_files_exist():
    """
    RL feedback output and report must exist.
    """

    assert cfg.RL_ADJUSTED_WEIGHTS_CSV.exists()

    assert cfg.RL_FEEDBACK_SUMMARY.exists()


# =============================================================================
# MPT INPUT TESTS
# =============================================================================

def test_mpt_solution_loads_successfully():
    """
    RL feedback must be able to load MPT solution.
    """

    dataframe = load_mpt_solution()

    assert not dataframe.empty


def test_mpt_solution_has_ten_stocks():
    """
    All ten Agent-4 stocks must remain in the MPT solution.
    """

    dataframe = load_mpt_solution()

    assert len(
        dataframe
    ) == 10

    assert dataframe[
        "symbol"
    ].nunique() == 10


def test_mpt_report_is_complete():
    """
    RL feedback should consume only a completed MPT stage.
    """

    report = load_mpt_report()

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


# =============================================================================
# ACTION DISTRIBUTION
# =============================================================================

def test_rl_feedback_action_distribution():
    """
    Current Agent-4 action distribution:

        BUY  = 1
        HOLD = 7
        SELL = 2
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

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


# =============================================================================
# STRUCTURE
# =============================================================================

def test_rl_feedback_preserves_ten_symbols():
    """
    RL feedback must preserve all ten stocks.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    assert len(
        dataframe
    ) == 10

    assert dataframe[
        "symbol"
    ].nunique() == 10


def test_rl_feedback_required_columns_exist():
    """
    Required RL feedback fields must exist.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    required_columns = {
        "symbol",
        "trade_action",
        "mpt_weight",
        "rl_action_mask",
        "rl_raw_weight",
        "rl_adjusted_weight",
        "rl_feedback",
        "final_target_weight",
    }

    assert required_columns.issubset(
        dataframe.columns
    )


# =============================================================================
# WEIGHT VALIDATION
# =============================================================================

def test_rl_adjusted_weights_are_finite():
    """
    RL-adjusted portfolio weights must be valid.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    weights = pd.to_numeric(
        dataframe[
            "rl_adjusted_weight"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    assert np.isfinite(
        weights
    ).all()


def test_rl_adjusted_weights_nonnegative():
    """
    Long-only portfolio.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    weights = dataframe[
        "rl_adjusted_weight"
    ].to_numpy(
        dtype=float
    )

    assert (
        weights
        >=
        -cfg.FLOAT_TOLERANCE
    ).all()


def test_rl_adjusted_weights_below_maximum():
    """
    RL weights must respect configured maximum.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    weights = dataframe[
        "rl_adjusted_weight"
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


def test_rl_adjusted_weights_sum_to_one():
    """
    Portfolio must remain fully invested.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    weight_sum = float(
        dataframe[
            "rl_adjusted_weight"
        ]
        .sum()
    )

    assert np.isclose(
        weight_sum,
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    )


# =============================================================================
# SELL RULE
# =============================================================================

def test_sell_stocks_have_zero_rl_weight():
    """
    Agent-4 SELL stocks must have zero RL-adjusted allocation.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

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
            "rl_adjusted_weight",
        ]
        .to_numpy(
            dtype=float
        ),
        cfg.SELL_TARGET_WEIGHT,
        atol=cfg.FLOAT_TOLERANCE,
    )


def test_sell_stocks_have_zero_action_mask():
    """
    SELL assets must be masked out.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

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

    masks = dataframe.loc[
        sell_mask,
        "rl_action_mask",
    ].to_numpy(
        dtype=float
    )

    assert np.allclose(
        masks,
        0.0,
        atol=cfg.FLOAT_TOLERANCE,
    )


# =============================================================================
# BUY / HOLD RULE
# =============================================================================

def test_buy_hold_stocks_have_active_action_mask():
    """
    BUY and HOLD assets remain active for portfolio allocation.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    eligible_mask = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .isin(
            cfg.MPT_ELIGIBLE_ACTIONS
        )
    )

    masks = dataframe.loc[
        eligible_mask,
        "rl_action_mask",
    ].to_numpy(
        dtype=float
    )

    assert np.allclose(
        masks,
        1.0,
        atol=cfg.FLOAT_TOLERANCE,
    )


def test_buy_hold_count_is_eight():
    """
    Eight stocks should remain BUY/HOLD eligible.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    eligible_mask = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .isin(
            cfg.MPT_ELIGIBLE_ACTIONS
        )
    )

    assert int(
        eligible_mask.sum()
    ) == 8


# =============================================================================
# NO ARBITRARY MULTIPLIERS
# =============================================================================

def test_rl_does_not_modify_buy_hold_mpt_weights():
    """
    Because MPT already applies SELL=0, the RL feedback layer should not
    arbitrarily change BUY/HOLD allocations.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    eligible_mask = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .isin(
            cfg.MPT_ELIGIBLE_ACTIONS
        )
    )

    mpt_weights = (
        dataframe.loc[
            eligible_mask,
            "mpt_weight",
        ]
        .to_numpy(
            dtype=float
        )
    )

    rl_weights = (
        dataframe.loc[
            eligible_mask,
            "rl_adjusted_weight",
        ]
        .to_numpy(
            dtype=float
        )
    )

    assert np.allclose(
        mpt_weights,
        rl_weights,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
        rtol=1e-6,
    )


# =============================================================================
# FINAL TARGET
# =============================================================================

def test_final_target_matches_rl_adjusted_weights():
    """
    final_target_weight should equal RL-adjusted portfolio weight.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    assert np.allclose(
        dataframe[
            "final_target_weight"
        ]
        .to_numpy(
            dtype=float
        ),
        dataframe[
            "rl_adjusted_weight"
        ]
        .to_numpy(
            dtype=float
        ),
        atol=cfg.FLOAT_TOLERANCE,
    )


def test_final_target_weights_sum_to_one():
    """
    Final RL target allocation must remain fully invested.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    assert np.isclose(
        dataframe[
            "final_target_weight"
        ]
        .sum(),
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    )


# =============================================================================
# DIRECT FUNCTION TESTS
# =============================================================================

def test_apply_rl_feedback_runs_successfully():
    """
    Reapply RL feedback to saved MPT solution.
    """

    mpt_solution = (
        load_mpt_solution()
    )

    result = apply_rl_feedback(
        mpt_solution
    )

    assert not result.empty

    assert len(
        result
    ) == 10


def test_apply_rl_feedback_validation_passes():
    """
    Directly validate newly generated feedback dataframe.
    """

    mpt_solution = (
        load_mpt_solution()
    )

    result = apply_rl_feedback(
        mpt_solution
    )

    checks = validate_rl_feedback(
        result
    )

    assert checks

    assert all(
        bool(
            value
        )
        for value
        in checks.values()
    )


# =============================================================================
# Q-VALUE DIAGNOSTIC
# =============================================================================

def test_q_margin_column_exists():
    """
    RL output should preserve Q-value diagnostic margin.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    assert "dqn_q_margin" in dataframe.columns


def test_q_margin_is_finite_when_q_values_available():
    """
    With current Agent-4 output, Q-values are present and margins should
    therefore be finite.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

    q_columns_present = all(
        column in dataframe.columns
        for column in [
            "q_hold",
            "q_buy",
            "q_sell",
        ]
    )

    if q_columns_present:

        margins = dataframe[
            "dqn_q_margin"
        ].to_numpy(
            dtype=float
        )

        assert np.isfinite(
            margins
        ).all()


# =============================================================================
# REPORT TESTS
# =============================================================================

def test_rl_feedback_report_status_complete():
    """
    RL feedback summary must show COMPLETE.
    """

    report = load_rl_report()

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


def test_rl_feedback_report_action_counts():
    """
    Saved report must preserve Agent-4 action distribution.
    """

    report = load_rl_report()

    actions = report[
        "actions"
    ]

    assert int(
        actions[
            "BUY"
        ]
    ) == 1

    assert int(
        actions[
            "HOLD"
        ]
    ) == 7

    assert int(
        actions[
            "SELL"
        ]
    ) == 2


def test_rl_feedback_report_weight_sum():
    """
    Saved RL-adjusted weight sum must equal one.
    """

    report = load_rl_report()

    weight_sum = float(
        report[
            "weight_summary"
        ][
            "rl_adjusted_weight_sum"
        ]
    )

    assert np.isclose(
        weight_sum,
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    )


def test_rl_feedback_report_validation_all_passed():
    """
    Every stored RL feedback validation check must be True.
    """

    report = load_rl_report()

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
# CURRENT KNOWN SELL SYMBOLS
# =============================================================================

def test_current_sell_symbols_match_validated_pipeline():
    """
    Current Agent-4 final inference produced SELL for:

        HONASA
        GVT&D
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

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
# CURRENT KNOWN BUY SYMBOL
# =============================================================================

def test_current_buy_symbol_matches_validated_pipeline():
    """
    Current Agent-4 final inference produced BUY for HBLENGINE.
    """

    dataframe = (
        load_rl_adjusted_weights()
    )

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