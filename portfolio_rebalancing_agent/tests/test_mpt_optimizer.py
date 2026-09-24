"""
Tests for Agent 5 - Markowitz MPT Optimizer
===============================================================================

Validates:

    portfolio_rebalancing_agent/src/mpt_optimizer.py

The tests confirm:

1. MPT output files exist
2. Optimizer receives eight eligible assets
3. Expected returns are finite
4. Target return equals average eligible-asset return
5. Covariance matrix is correctly aligned
6. MPT weights are finite
7. MPT weights are non-negative
8. MPT weights sum to one
9. Agent-4 SELL stocks receive zero weight
10. BUY/HOLD stocks remain MPT eligible
11. Portfolio expected return satisfies target
12. Optimized variance is valid
13. Optimized variance does not exceed equal-weight variance
14. Optimized volatility is finite
15. MPT solution contains all ten Agent-4 stocks
16. Optimization report is complete
17. Solver reports success
18. Optimization validation checks all passed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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

from src.mpt_optimizer import (
    prepare_optimizer_inputs,
    optimize_minimum_variance,
    load_covariance_matrix,
)

from src.portfolio_metrics import (
    portfolio_expected_return,
    portfolio_variance,
    portfolio_volatility,
)


# =============================================================================
# HELPERS
# =============================================================================

def load_mpt_solution() -> pd.DataFrame:
    """
    Load saved MPT solution.
    """

    if not cfg.MPT_SOLUTION_CSV.exists():

        raise FileNotFoundError(
            f"MPT solution not found:\n"
            f"{cfg.MPT_SOLUTION_CSV}"
        )

    dataframe = pd.read_csv(
        cfg.MPT_SOLUTION_CSV,
        low_memory=False,
    )

    return dataframe


def load_mpt_report() -> dict:
    """
    Load saved optimization summary.
    """

    if not cfg.MPT_OPTIMIZATION_SUMMARY.exists():

        raise FileNotFoundError(
            f"MPT report not found:\n"
            f"{cfg.MPT_OPTIMIZATION_SUMMARY}"
        )

    with open(
        cfg.MPT_OPTIMIZATION_SUMMARY,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# =============================================================================
# FILE EXISTENCE
# =============================================================================

def test_mpt_output_files_exist():
    """
    Core optimizer outputs must exist.
    """

    assert cfg.MPT_SOLUTION_CSV.exists()

    assert cfg.MPT_OPTIMIZATION_SUMMARY.exists()


# =============================================================================
# OPTIMIZER INPUTS
# =============================================================================

def test_optimizer_inputs_prepare_successfully():
    """
    Optimizer input preparation should complete without error.
    """

    inputs = prepare_optimizer_inputs()

    assert inputs is not None

    assert "eligible_symbols" in inputs

    assert "expected_returns" in inputs

    assert "covariance" in inputs

    assert "target_return" in inputs


def test_optimizer_has_eight_eligible_assets():
    """
    Current validated Agent-4 decisions contain:

        BUY  = 1
        HOLD = 7
        SELL = 2

    Therefore MPT has eight eligible assets.
    """

    inputs = prepare_optimizer_inputs()

    assert len(
        inputs[
            "eligible_symbols"
        ]
    ) == 8


def test_expected_returns_are_finite():
    """
    MPT expected-return vector must be numerically valid.
    """

    inputs = prepare_optimizer_inputs()

    expected_returns = (
        inputs[
            "expected_returns"
        ]
        .to_numpy(
            dtype=float
        )
    )

    assert np.isfinite(
        expected_returns
    ).all()


def test_target_return_equals_average_asset_mean():
    """
    Paper-aligned implementation:

        target return =
        average expected return across eligible assets
    """

    inputs = prepare_optimizer_inputs()

    expected_target = float(
        inputs[
            "expected_returns"
        ]
        .mean()
    )

    actual_target = float(
        inputs[
            "target_return"
        ]
    )

    assert np.isclose(
        actual_target,
        expected_target,
        atol=cfg.TARGET_RETURN_TOLERANCE,
    )


# =============================================================================
# COVARIANCE
# =============================================================================

def test_covariance_matches_eligible_assets():
    """
    Covariance matrix must contain exactly the eligible MPT assets.
    """

    inputs = prepare_optimizer_inputs()

    eligible_symbols = (
        inputs[
            "eligible_symbols"
        ]
    )

    covariance = (
        inputs[
            "covariance"
        ]
    )

    assert list(
        covariance.index
    ) == eligible_symbols

    assert list(
        covariance.columns
    ) == eligible_symbols


def test_covariance_loader_returns_symmetric_matrix():
    """
    Saved covariance matrix must remain symmetric.
    """

    covariance = (
        load_covariance_matrix()
    )

    values = covariance.to_numpy(
        dtype=float
    )

    assert np.allclose(
        values,
        values.T,
        atol=1e-10,
    )


# =============================================================================
# SAVED MPT SOLUTION
# =============================================================================

def test_mpt_solution_has_ten_stocks():
    """
    MPT output must preserve all ten Agent-4 stocks.

    SELL stocks remain in the output with zero allocation.
    """

    solution = load_mpt_solution()

    assert len(
        solution
    ) == 10

    assert solution[
        "symbol"
    ].nunique() == 10


def test_mpt_solution_required_columns():
    """
    Core MPT fields must exist.
    """

    solution = load_mpt_solution()

    required_columns = {
        "symbol",
        "trade_action",
        "mpt_expected_return",
        "mpt_weight",
        "mpt_allocation_percent",
        "mpt_allocation_rank",
    }

    assert required_columns.issubset(
        solution.columns
    )


def test_mpt_weights_are_finite():
    """
    MPT portfolio weights must contain no NaN/inf.
    """

    solution = load_mpt_solution()

    weights = pd.to_numeric(
        solution[
            "mpt_weight"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    assert np.isfinite(
        weights
    ).all()


def test_mpt_weights_are_nonnegative():
    """
    Short selling is disabled.
    """

    solution = load_mpt_solution()

    weights = solution[
        "mpt_weight"
    ].to_numpy(
        dtype=float
    )

    assert (
        weights
        >=
        -cfg.FLOAT_TOLERANCE
    ).all()


def test_mpt_weights_do_not_exceed_maximum():
    """
    Portfolio weights must respect configured upper bound.
    """

    solution = load_mpt_solution()

    weights = solution[
        "mpt_weight"
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


def test_mpt_weights_sum_to_one():
    """
    Final MPT allocation must be fully invested.
    """

    solution = load_mpt_solution()

    weight_sum = float(
        solution[
            "mpt_weight"
        ]
        .sum()
    )

    assert np.isclose(
        weight_sum,
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    )


# =============================================================================
# AGENT-4 ACTION CONSTRAINTS
# =============================================================================

def test_agent4_sell_stocks_have_zero_mpt_weight():
    """
    Agent-4 SELL decisions must receive zero MPT allocation.
    """

    solution = load_mpt_solution()

    sell_rows = (
        solution[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .eq(
            cfg.ACTION_SELL
        )
    )

    assert int(
        sell_rows.sum()
    ) == 2

    assert np.allclose(
        solution.loc[
            sell_rows,
            "mpt_weight",
        ].to_numpy(
            dtype=float
        ),
        cfg.SELL_TARGET_WEIGHT,
        atol=cfg.FLOAT_TOLERANCE,
    )


def test_buy_hold_assets_are_the_eight_eligible_assets():
    """
    BUY/HOLD stock count must equal eight.
    """

    solution = load_mpt_solution()

    eligible_rows = (
        solution[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .isin(
            cfg.MPT_ELIGIBLE_ACTIONS
        )
    )

    assert int(
        eligible_rows.sum()
    ) == 8


# =============================================================================
# DIRECT OPTIMIZER TEST
# =============================================================================

def test_optimizer_runs_successfully():
    """
    Run the lightweight SLSQP optimizer directly using the already
    prepared 105-day inputs.
    """

    inputs = prepare_optimizer_inputs()

    result = optimize_minimum_variance(
        expected_returns=inputs[
            "expected_returns"
        ],
        covariance=inputs[
            "covariance"
        ],
        target_return=inputs[
            "target_return"
        ],
    )

    assert result is not None

    assert result[
        "optimizer_result"
    ].success


def test_optimized_weights_sum_to_one():
    """
    Direct optimizer output must remain fully invested.
    """

    inputs = prepare_optimizer_inputs()

    result = optimize_minimum_variance(
        expected_returns=inputs[
            "expected_returns"
        ],
        covariance=inputs[
            "covariance"
        ],
        target_return=inputs[
            "target_return"
        ],
    )

    weights = result[
        "weights"
    ]

    assert np.isclose(
        weights.sum(),
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    )


def test_optimized_return_matches_target():
    """
    Equality constraint:

        w.T μ = target return
    """

    inputs = prepare_optimizer_inputs()

    result = optimize_minimum_variance(
        expected_returns=inputs[
            "expected_returns"
        ],
        covariance=inputs[
            "covariance"
        ],
        target_return=inputs[
            "target_return"
        ],
    )

    actual_return = (
        portfolio_expected_return(
            weights=result[
                "weights"
            ],
            expected_returns=inputs[
                "expected_returns"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    assert np.isclose(
        actual_return,
        inputs[
            "target_return"
        ],
        atol=cfg.TARGET_RETURN_TOLERANCE,
    )


# =============================================================================
# VARIANCE TESTS
# =============================================================================

def test_optimized_variance_is_nonnegative():
    """
    Portfolio variance must be non-negative.
    """

    inputs = prepare_optimizer_inputs()

    result = optimize_minimum_variance(
        expected_returns=inputs[
            "expected_returns"
        ],
        covariance=inputs[
            "covariance"
        ],
        target_return=inputs[
            "target_return"
        ],
    )

    variance = portfolio_variance(
        weights=result[
            "weights"
        ],
        covariance_matrix=inputs[
            "covariance"
        ].to_numpy(
            dtype=float
        ),
    )

    assert variance >= 0


def test_optimized_variance_not_above_equal_weight():
    """
    Minimum-variance optimization should not produce variance greater
    than the feasible equal-weight starting portfolio.
    """

    inputs = prepare_optimizer_inputs()

    result = optimize_minimum_variance(
        expected_returns=inputs[
            "expected_returns"
        ],
        covariance=inputs[
            "covariance"
        ],
        target_return=inputs[
            "target_return"
        ],
    )

    optimized_variance = portfolio_variance(
        weights=result[
            "weights"
        ],
        covariance_matrix=inputs[
            "covariance"
        ].to_numpy(
            dtype=float
        ),
    )

    number_of_assets = len(
        inputs[
            "eligible_symbols"
        ]
    )

    equal_weights = np.full(
        number_of_assets,
        1.0
        /
        number_of_assets,
        dtype=float,
    )

    equal_variance = portfolio_variance(
        weights=equal_weights,
        covariance_matrix=inputs[
            "covariance"
        ].to_numpy(
            dtype=float
        ),
    )

    assert optimized_variance <= (
        equal_variance
        +
        1e-8
    )


def test_optimized_volatility_is_finite():
    """
    Optimized portfolio volatility must be valid.
    """

    inputs = prepare_optimizer_inputs()

    result = optimize_minimum_variance(
        expected_returns=inputs[
            "expected_returns"
        ],
        covariance=inputs[
            "covariance"
        ],
        target_return=inputs[
            "target_return"
        ],
    )

    volatility = portfolio_volatility(
        weights=result[
            "weights"
        ],
        covariance_matrix=inputs[
            "covariance"
        ].to_numpy(
            dtype=float
        ),
    )

    assert np.isfinite(
        volatility
    )

    assert volatility >= 0


# =============================================================================
# REPORT TESTS
# =============================================================================

def test_mpt_report_status_complete():
    """
    Saved MPT report must show COMPLETE.
    """

    report = load_mpt_report()

    assert (
        str(
            report.get(
                "status"
            )
        )
        .upper()
        ==
        "COMPLETE"
    )


def test_mpt_report_solver_success():
    """
    SLSQP must report successful optimization.
    """

    report = load_mpt_report()

    assert report[
        "solver"
    ][
        "success"
    ] is True


def test_mpt_report_uses_slsqp():
    """
    Current Agent-5 configuration uses SLSQP.
    """

    report = load_mpt_report()

    assert (
        str(
            report[
                "optimizer"
            ]
        )
        .upper()
        ==
        str(
            cfg.OPTIMIZER_METHOD
        )
        .upper()
    )


def test_mpt_report_target_mode():
    """
    Target-return mode must match configured paper-aligned mode.
    """

    report = load_mpt_report()

    assert (
        report[
            "target_return_mode"
        ]
        ==
        cfg.TARGET_RETURN_MODE
    )


def test_mpt_report_validation_all_passed():
    """
    Every saved MPT validation check must be True.
    """

    report = load_mpt_report()

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


def test_mpt_report_contains_eight_eligible_assets():
    """
    Saved optimization summary should contain eight optimized assets.
    """

    report = load_mpt_report()

    assert int(
        report[
            "eligible_assets"
        ]
    ) == 8


def test_saved_target_return_matches_current_inputs():
    """
    Saved report target should match currently prepared target.
    """

    report = load_mpt_report()

    inputs = prepare_optimizer_inputs()

    assert np.isclose(
        float(
            report[
                "target_return"
            ]
        ),
        float(
            inputs[
                "target_return"
            ]
        ),
        atol=cfg.TARGET_RETURN_TOLERANCE,
    )