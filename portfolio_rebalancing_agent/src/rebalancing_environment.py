"""
Agent 5 - Portfolio Rebalancing Environment
===============================================================================

PROJECT
-------
Agentic AI Portfolio Optimization and Algorithmic Trading

AGENT
-----
Agent 5 - Portfolio Rebalancing Agent

PURPOSE
-------
Model the transition from the current/provisional portfolio allocation
to the final Agent-5 RL + Markowitz MPT target allocation.

PIPELINE
--------
Agent 3 provisional risk allocation
        +
Agent 4 DQN decisions
        +
Agent 5 MPT weights
        +
Agent 5 RL feedback
        ↓
Current Portfolio State
        ↓
Weight Changes
        ↓
Rebalancing BUY / SELL / HOLD
        ↓
Target Portfolio State
        ↓
Turnover
        ↓
Portfolio Metrics


IMPORTANT TERMINOLOGY
---------------------
Two different action concepts are used.

1. trade_action

   This is Agent-4 DQN output:

       BUY
       SELL
       HOLD

2. rebalance_direction

   This represents how portfolio weight must change to move from
   the provisional portfolio to the final MPT + RL target.

Example:

    Agent-4 trade_action = HOLD

but:

    current weight = 15%
    target weight  = 10%

Then:

    rebalance_direction = SELL

This does NOT change the original Agent-4 decision.

It only means the portfolio allocation needs to be reduced.


CURRENT PORTFOLIO
-----------------
Preferred source:

    Agent-3 risk_adjusted_weight

Fallback:

    Equal weights

Agent-3 risk-adjusted weights are treated as the provisional portfolio
state before Agent-5 optimization.


TARGET PORTFOLIO
----------------
Target allocation comes from:

    rl_feedback.py

using:

    final_target_weight

or:

    rl_adjusted_weight


DQN SELL RULE
-------------
Agent-4 SELL stocks must have:

    target_weight = 0


TURNOVER
--------
Turnover is calculated as:

    Σ |target_weight - current_weight|


IMPORTANT
---------
The current and target portfolio metrics calculated here use the same
historical 105-day estimation period.

They are diagnostic/in-sample statistics.

They are NOT a future post-2025-12-01 backtest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# IMPORT PATH
# =============================================================================

AGENT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(AGENT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(AGENT_ROOT),
    )


from config import config as cfg

from src.portfolio_metrics import (
    load_returns_matrix,
    calculate_portfolio_metrics,
    portfolio_turnover,
)


# =============================================================================
# OUTPUT PATHS
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
# LOAD RL-ADJUSTED PORTFOLIO
# =============================================================================

def load_rl_adjusted_portfolio() -> pd.DataFrame:
    """
    Load Agent-5 RL-adjusted MPT portfolio.

    Input:
        rl_adjusted_weights.csv
    """

    path = cfg.RL_ADJUSTED_WEIGHTS_CSV

    if not path.exists():

        raise FileNotFoundError(
            "RL-adjusted portfolio was not found.\n\n"
            f"Expected:\n{path}\n\n"
            "Run rl_feedback.py first."
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    if dataframe.empty:

        raise ValueError(
            "RL-adjusted portfolio is empty."
        )

    required_columns = {
        "symbol",
        "trade_action",
        "mpt_weight",
        "rl_adjusted_weight",
    }

    missing_columns = (
        required_columns
        -
        set(dataframe.columns)
    )

    if missing_columns:

        raise ValueError(
            "RL-adjusted portfolio is missing required columns:\n"
            +
            "\n".join(
                sorted(missing_columns)
            )
        )

    dataframe = dataframe.copy()

    # -------------------------------------------------------------------------
    # SYMBOL
    # -------------------------------------------------------------------------

    dataframe[
        "symbol"
    ] = (
        dataframe[
            "symbol"
        ]
        .astype(str)
        .str.strip()
    )

    # -------------------------------------------------------------------------
    # AGENT-4 ACTION
    # -------------------------------------------------------------------------

    dataframe[
        "trade_action"
    ] = (
        dataframe[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # -------------------------------------------------------------------------
    # NUMERIC COLUMNS
    # -------------------------------------------------------------------------

    numeric_columns = [
        column
        for column in [
            "mpt_weight",
            "rl_adjusted_weight",
            "final_target_weight",
            "risk_adjusted_weight",
            "q_hold",
            "q_buy",
            "q_sell",
            "selected_q_value",
            "agent3_score",
            "agent3_rank",
            "top30_probability",
        ]
        if column in dataframe.columns
    ]

    for column in numeric_columns:

        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # UNIQUE SYMBOLS
    # -------------------------------------------------------------------------

    if dataframe[
        "symbol"
    ].duplicated().any():

        duplicates = (
            dataframe.loc[
                dataframe[
                    "symbol"
                ].duplicated(
                    keep=False
                ),
                "symbol",
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate symbols in RL-adjusted portfolio:\n"
            +
            "\n".join(
                sorted(
                    set(duplicates)
                )
            )
        )

    # -------------------------------------------------------------------------
    # ACTION VALIDATION
    # -------------------------------------------------------------------------

    if not dataframe[
        "trade_action"
    ].isin(
        cfg.VALID_ACTIONS
    ).all():

        invalid_actions = (
            dataframe.loc[
                ~dataframe[
                    "trade_action"
                ].isin(
                    cfg.VALID_ACTIONS
                ),
                "trade_action",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Invalid Agent-4 actions found:\n"
            f"{invalid_actions}"
        )

    # -------------------------------------------------------------------------
    # REQUIRED WEIGHTS
    # -------------------------------------------------------------------------

    if dataframe[
        "mpt_weight"
    ].isna().any():

        raise ValueError(
            "MPT weights contain missing values."
        )

    if dataframe[
        "rl_adjusted_weight"
    ].isna().any():

        raise ValueError(
            "RL-adjusted weights contain missing values."
        )

    return dataframe


# =============================================================================
# DETERMINE CURRENT PORTFOLIO WEIGHTS
# =============================================================================

def determine_current_weights(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, str]:
    """
    Determine current/provisional portfolio allocation.

    Preferred source:
        Agent-3 risk_adjusted_weight

    Fallback:
        equal-weight portfolio

    IMPORTANT FIX
    -------------
    np.array(..., copy=True) is used deliberately.

    Pandas/Parquet can expose a read-only NumPy view.

    The previous implementation attempted to modify such an array and
    produced:

        ValueError: assignment destination is read-only
    """

    number_of_assets = len(
        dataframe
    )

    if number_of_assets == 0:

        raise ValueError(
            "Cannot create current portfolio from zero assets."
        )

    # =========================================================================
    # PREFERRED SOURCE - AGENT 3
    # =========================================================================

    if "risk_adjusted_weight" in dataframe.columns:

        candidate = np.array(
            pd.to_numeric(
                dataframe[
                    "risk_adjusted_weight"
                ],
                errors="coerce",
            ),
            dtype=float,
            copy=True,
        ).reshape(
            -1
        )

        # ---------------------------------------------------------------------
        # VALIDATE RAW AGENT-3 WEIGHTS
        # ---------------------------------------------------------------------

        candidate_is_valid = bool(
            candidate.size
            ==
            number_of_assets

            and

            np.isfinite(
                candidate
            ).all()

            and

            np.all(
                candidate
                >=
                -cfg.FLOAT_TOLERANCE
            )

            and

            float(
                candidate.sum()
            )
            >
            cfg.FLOAT_TOLERANCE
        )

        if candidate_is_valid:

            # -----------------------------------------------------------------
            # REMOVE VERY SMALL FLOATING-POINT VALUES
            # -----------------------------------------------------------------

            candidate[
                np.abs(
                    candidate
                )
                <
                1e-12
            ] = 0.0

            # -----------------------------------------------------------------
            # SHORT SELLING DISABLED
            # -----------------------------------------------------------------

            candidate = np.clip(
                candidate,
                0.0,
                None,
            )

            candidate_sum = float(
                candidate.sum()
            )

            if candidate_sum <= cfg.FLOAT_TOLERANCE:

                raise ValueError(
                    "Agent-3 risk-adjusted weights have "
                    "zero usable total allocation."
                )

            # -----------------------------------------------------------------
            # NORMALIZE TO 100%
            # -----------------------------------------------------------------

            candidate = (
                candidate
                /
                candidate_sum
                *
                cfg.WEIGHT_SUM_TARGET
            )

            # Ensure writable array after arithmetic.
            candidate = np.array(
                candidate,
                dtype=float,
                copy=True,
            )

            # -----------------------------------------------------------------
            # FINAL NUMERICAL CLEANUP
            # -----------------------------------------------------------------

            candidate[
                np.abs(
                    candidate
                )
                <
                1e-12
            ] = 0.0

            final_sum = float(
                candidate.sum()
            )

            if final_sum <= cfg.FLOAT_TOLERANCE:

                raise ValueError(
                    "Current portfolio became empty after normalization."
                )

            candidate = (
                candidate
                /
                final_sum
                *
                cfg.WEIGHT_SUM_TARGET
            )

            candidate = np.array(
                candidate,
                dtype=float,
                copy=True,
            )

            # -----------------------------------------------------------------
            # FINAL VALIDATION
            # -----------------------------------------------------------------

            if not np.isfinite(
                candidate
            ).all():

                raise ValueError(
                    "Current portfolio weights contain invalid values."
                )

            if np.any(
                candidate
                <
                -cfg.FLOAT_TOLERANCE
            ):

                raise ValueError(
                    "Current portfolio contains negative weights."
                )

            if not np.isclose(
                candidate.sum(),
                cfg.WEIGHT_SUM_TARGET,
                atol=cfg.WEIGHT_SUM_TOLERANCE,
            ):

                raise ValueError(
                    "Current portfolio weights failed normalization."
                )

            return (
                candidate,
                "agent3_risk_adjusted_weight",
            )

    # =========================================================================
    # FALLBACK - EQUAL WEIGHT
    # =========================================================================

    equal_weights = np.full(
        shape=number_of_assets,
        fill_value=(
            cfg.WEIGHT_SUM_TARGET
            /
            number_of_assets
        ),
        dtype=float,
    )

    if not np.isclose(
        equal_weights.sum(),
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    ):

        raise ValueError(
            "Equal-weight fallback failed normalization."
        )

    return (
        equal_weights,
        "equal_weight_fallback",
    )


# =============================================================================
# DETERMINE TARGET WEIGHTS
# =============================================================================

def determine_target_weights(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    """
    Determine final Agent-5 target portfolio weights.

    Preferred:
        final_target_weight

    Fallback:
        rl_adjusted_weight

    SELL stocks are forced to zero.

    IMPORTANT
    ---------
    A writable NumPy copy is explicitly created to avoid the same
    pandas read-only-array issue that can occur with to_numpy().
    """

    if "final_target_weight" in dataframe.columns:

        source_series = pd.to_numeric(
            dataframe[
                "final_target_weight"
            ],
            errors="coerce",
        )

    else:

        source_series = pd.to_numeric(
            dataframe[
                "rl_adjusted_weight"
            ],
            errors="coerce",
        )

    target = np.array(
        source_series,
        dtype=float,
        copy=True,
    ).reshape(
        -1
    )

    if target.size != len(
        dataframe
    ):

        raise ValueError(
            "Target-weight vector length mismatch."
        )

    # -------------------------------------------------------------------------
    # FINITE VALUES
    # -------------------------------------------------------------------------

    if not np.isfinite(
        target
    ).all():

        raise ValueError(
            "Target weights contain NaN or infinity."
        )

    # -------------------------------------------------------------------------
    # LONG-ONLY PORTFOLIO
    # -------------------------------------------------------------------------

    if np.any(
        target
        <
        (
            cfg.MIN_WEIGHT
            -
            cfg.FLOAT_TOLERANCE
        )
    ):

        raise ValueError(
            "Target portfolio contains negative weights."
        )

    # -------------------------------------------------------------------------
    # SMALL VALUES -> EXACT ZERO
    # -------------------------------------------------------------------------

    target[
        np.abs(
            target
        )
        <
        1e-12
    ] = 0.0

    # =========================================================================
    # DQN SELL -> ZERO ALLOCATION
    # =========================================================================

    sell_mask = (
        dataframe[
            "trade_action"
        ]
        .eq(
            cfg.ACTION_SELL
        )
        .to_numpy()
    )

    target[
        sell_mask
    ] = (
        cfg.SELL_TARGET_WEIGHT
    )

    # -------------------------------------------------------------------------
    # NORMALIZE
    # -------------------------------------------------------------------------

    target_sum = float(
        target.sum()
    )

    if target_sum <= cfg.FLOAT_TOLERANCE:

        raise ValueError(
            "Target portfolio has zero total allocation."
        )

    target = (
        target
        /
        target_sum
        *
        cfg.WEIGHT_SUM_TARGET
    )

    target = np.array(
        target,
        dtype=float,
        copy=True,
    )

    # -------------------------------------------------------------------------
    # RE-ENFORCE EXACT SELL ZERO
    # -------------------------------------------------------------------------

    target[
        sell_mask
    ] = (
        cfg.SELL_TARGET_WEIGHT
    )

    eligible_mask = (
        ~sell_mask
    )

    eligible_sum = float(
        target[
            eligible_mask
        ].sum()
    )

    if eligible_sum <= cfg.FLOAT_TOLERANCE:

        raise ValueError(
            "No BUY/HOLD target allocation remains."
        )

    target[
        eligible_mask
    ] = (
        target[
            eligible_mask
        ]
        /
        eligible_sum
        *
        cfg.WEIGHT_SUM_TARGET
    )

    # -------------------------------------------------------------------------
    # FINAL VALIDATION
    # -------------------------------------------------------------------------

    if not np.isclose(
        target.sum(),
        cfg.WEIGHT_SUM_TARGET,
        atol=cfg.WEIGHT_SUM_TOLERANCE,
    ):

        raise ValueError(
            "Target portfolio weights do not sum to one."
        )

    if sell_mask.any():

        if not np.allclose(
            target[
                sell_mask
            ],
            cfg.SELL_TARGET_WEIGHT,
            atol=cfg.FLOAT_TOLERANCE,
        ):

            raise ValueError(
                "Agent-4 SELL stocks were not assigned "
                "zero target weight."
            )

    return target


# =============================================================================
# BUILD REBALANCING TRANSITION
# =============================================================================

def build_rebalancing_transition(
    dataframe: pd.DataFrame,
    current_weights: np.ndarray,
    target_weights: np.ndarray,
) -> pd.DataFrame:
    """
    Construct complete current -> target portfolio transition.
    """

    result = dataframe.copy()

    if len(
        result
    ) != len(
        current_weights
    ):

        raise ValueError(
            "Current-weight vector length does not match "
            "portfolio rows."
        )

    if len(
        result
    ) != len(
        target_weights
    ):

        raise ValueError(
            "Target-weight vector length does not match "
            "portfolio rows."
        )

    # =========================================================================
    # CURRENT AND TARGET WEIGHTS
    # =========================================================================

    result[
        "current_weight"
    ] = np.array(
        current_weights,
        dtype=float,
        copy=True,
    )

    result[
        "target_weight"
    ] = np.array(
        target_weights,
        dtype=float,
        copy=True,
    )

    result[
        "current_allocation_percent"
    ] = (
        result[
            "current_weight"
        ]
        *
        100.0
    )

    result[
        "target_allocation_percent"
    ] = (
        result[
            "target_weight"
        ]
        *
        100.0
    )

    # =========================================================================
    # WEIGHT CHANGE
    # =========================================================================

    result[
        "weight_change"
    ] = (
        result[
            "target_weight"
        ]
        -
        result[
            "current_weight"
        ]
    )

    result[
        "absolute_weight_change"
    ] = np.abs(
        result[
            "weight_change"
        ]
    )

    result[
        "weight_change_percent"
    ] = (
        result[
            "weight_change"
        ]
        *
        100.0
    )

    # =========================================================================
    # REBALANCING DIRECTION
    # =========================================================================

    tolerance = (
        cfg.FLOAT_TOLERANCE
    )

    result[
        "rebalance_direction"
    ] = np.select(
        [
            result[
                "weight_change"
            ]
            >
            tolerance,

            result[
                "weight_change"
            ]
            <
            -tolerance,
        ],
        [
            "BUY",
            "SELL",
        ],
        default="HOLD",
    )

    # =========================================================================
    # TURNOVER CONTRIBUTION
    # =========================================================================

    result[
        "turnover_contribution"
    ] = (
        result[
            "absolute_weight_change"
        ]
    )

    total_turnover = float(
        result[
            "turnover_contribution"
        ]
        .sum()
    )

    if total_turnover > cfg.FLOAT_TOLERANCE:

        result[
            "turnover_share"
        ] = (
            result[
                "turnover_contribution"
            ]
            /
            total_turnover
        )

    else:

        result[
            "turnover_share"
        ] = 0.0

    # =========================================================================
    # DQN-FORCED EXIT FLAG
    # =========================================================================

    result[
        "dqn_forced_exit"
    ] = (
        result[
            "trade_action"
        ]
        ==
        cfg.ACTION_SELL
    )

    # =========================================================================
    # TARGET ALLOCATION RANK
    # =========================================================================

    result[
        "target_allocation_rank"
    ] = (
        result[
            "target_weight"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    # =========================================================================
    # SORT
    # =========================================================================

    result = (
        result
        .sort_values(
            [
                "target_weight",
                "symbol",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return result


# =============================================================================
# ALIGN WITH RETURNS MATRIX
# =============================================================================

def align_weights_with_returns(
    transition: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """
    Align current and target portfolio weights with return-matrix columns.
    """

    transition_symbols = set(
        transition[
            "symbol"
        ]
        .astype(str)
    )

    return_symbols = set(
        returns.columns
        .astype(str)
    )

    missing_symbols = (
        transition_symbols
        -
        return_symbols
    )

    if missing_symbols:

        raise ValueError(
            "Portfolio symbols missing from returns matrix:\n"
            +
            "\n".join(
                sorted(
                    missing_symbols
                )
            )
        )

    # Preserve returns-matrix ordering.
    ordered_symbols = [
        str(symbol)
        for symbol
        in returns.columns
        if str(symbol)
        in transition_symbols
    ]

    aligned_transition = (
        transition
        .set_index(
            "symbol"
        )
        .loc[
            ordered_symbols
        ]
        .reset_index()
    )

    aligned_returns = (
        returns[
            ordered_symbols
        ]
        .copy()
    )

    # Writable copies.
    current_weights = np.array(
        aligned_transition[
            "current_weight"
        ],
        dtype=float,
        copy=True,
    )

    target_weights = np.array(
        aligned_transition[
            "target_weight"
        ],
        dtype=float,
        copy=True,
    )

    return (
        aligned_returns,
        current_weights,
        target_weights,
    )


# =============================================================================
# VALIDATE REBALANCING TRANSITION
# =============================================================================

def validate_rebalancing_transition(
    transition: pd.DataFrame,
) -> dict[str, bool]:
    """
    Validate current -> target portfolio transition.
    """

    current_weights = np.array(
        transition[
            "current_weight"
        ],
        dtype=float,
        copy=True,
    )

    target_weights = np.array(
        transition[
            "target_weight"
        ],
        dtype=float,
        copy=True,
    )

    weight_changes = np.array(
        transition[
            "weight_change"
        ],
        dtype=float,
        copy=True,
    )

    expected_weight_changes = (
        target_weights
        -
        current_weights
    )

    sell_mask = (
        transition[
            "trade_action"
        ]
        ==
        cfg.ACTION_SELL
    )

    checks: dict[str, bool] = {}

    # -------------------------------------------------------------------------
    # STRUCTURE
    # -------------------------------------------------------------------------

    checks[
        "transition_not_empty"
    ] = (
        not transition.empty
    )

    checks[
        "one_row_per_symbol"
    ] = (
        not transition[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    # -------------------------------------------------------------------------
    # CURRENT WEIGHTS
    # -------------------------------------------------------------------------

    checks[
        "current_weights_finite"
    ] = bool(
        np.isfinite(
            current_weights
        ).all()
    )

    checks[
        "current_weights_nonnegative"
    ] = bool(
        (
            current_weights
            >=
            -cfg.FLOAT_TOLERANCE
        ).all()
    )

    checks[
        "current_weights_sum_to_one"
    ] = bool(
        np.isclose(
            current_weights.sum(),
            cfg.WEIGHT_SUM_TARGET,
            atol=cfg.WEIGHT_SUM_TOLERANCE,
        )
    )

    # -------------------------------------------------------------------------
    # TARGET WEIGHTS
    # -------------------------------------------------------------------------

    checks[
        "target_weights_finite"
    ] = bool(
        np.isfinite(
            target_weights
        ).all()
    )

    checks[
        "target_weights_nonnegative"
    ] = bool(
        (
            target_weights
            >=
            -cfg.FLOAT_TOLERANCE
        ).all()
    )

    checks[
        "target_weights_sum_to_one"
    ] = bool(
        np.isclose(
            target_weights.sum(),
            cfg.WEIGHT_SUM_TARGET,
            atol=cfg.WEIGHT_SUM_TOLERANCE,
        )
    )

    # -------------------------------------------------------------------------
    # WEIGHT-CHANGE IDENTITY
    # -------------------------------------------------------------------------

    checks[
        "weight_change_identity"
    ] = bool(
        np.allclose(
            weight_changes,
            expected_weight_changes,
            atol=cfg.FLOAT_TOLERANCE,
        )
    )

    # -------------------------------------------------------------------------
    # DIRECTIONS
    # -------------------------------------------------------------------------

    checks[
        "rebalance_directions_valid"
    ] = bool(
        transition[
            "rebalance_direction"
        ]
        .isin(
            [
                "BUY",
                "SELL",
                "HOLD",
            ]
        )
        .all()
    )

    # -------------------------------------------------------------------------
    # AGENT-4 SELL -> TARGET ZERO
    # -------------------------------------------------------------------------

    if sell_mask.any():

        checks[
            "dqn_sell_target_weights_zero"
        ] = bool(
            np.allclose(
                transition.loc[
                    sell_mask,
                    "target_weight",
                ]
                .to_numpy(
                    dtype=float
                ),
                cfg.SELL_TARGET_WEIGHT,
                atol=cfg.FLOAT_TOLERANCE,
            )
        )

    else:

        checks[
            "dqn_sell_target_weights_zero"
        ] = True

    # -------------------------------------------------------------------------
    # TURNOVER
    # -------------------------------------------------------------------------

    checks[
        "turnover_contributions_nonnegative"
    ] = bool(
        (
            transition[
                "turnover_contribution"
            ]
            >=
            0
        )
        .all()
    )

    checks[
        "turnover_equals_absolute_weight_change"
    ] = bool(
        np.allclose(
            transition[
                "turnover_contribution"
            ].to_numpy(
                dtype=float
            ),
            transition[
                "absolute_weight_change"
            ].to_numpy(
                dtype=float
            ),
            atol=cfg.FLOAT_TOLERANCE,
        )
    )

    return checks


# =============================================================================
# JSON-SAFE METRICS
# =============================================================================

def metrics_to_json(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert NumPy scalar values into JSON-safe Python values.
    """

    result: dict[str, Any] = {}

    for key, value in metrics.items():

        if isinstance(
            value,
            (
                np.integer,
                int,
            ),
        ):

            result[
                key
            ] = int(
                value
            )

        elif isinstance(
            value,
            (
                np.floating,
                float,
            ),
        ):

            result[
                key
            ] = float(
                value
            )

        else:

            result[
                key
            ] = value

    return result


# =============================================================================
# COMPLETE REBALANCING ENVIRONMENT
# =============================================================================

def run_rebalancing_environment() -> dict[str, Any]:
    """
    Execute complete Agent-5 portfolio-state transition.
    """

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 5 - PORTFOLIO REBALANCING ENVIRONMENT"
    )

    print(
        "=" * 100
    )

    print()

    # =========================================================================
    # STEP 1 - LOAD RL + MPT OUTPUT
    # =========================================================================

    portfolio = (
        load_rl_adjusted_portfolio()
    )

    print(
        f"Portfolio stocks        : "
        f"{len(portfolio)}"
    )

    action_counts = (
        portfolio[
            "trade_action"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        f"Agent-4 BUY             : "
        f"{action_counts.get('BUY', 0)}"
    )

    print(
        f"Agent-4 HOLD            : "
        f"{action_counts.get('HOLD', 0)}"
    )

    print(
        f"Agent-4 SELL            : "
        f"{action_counts.get('SELL', 0)}"
    )

    # =========================================================================
    # STEP 2 - CURRENT PORTFOLIO
    # =========================================================================

    (
        current_weights,
        current_weight_source,
    ) = determine_current_weights(
        portfolio
    )

    print()

    print(
        f"Current weight source   : "
        f"{current_weight_source}"
    )

    print(
        f"Current weight sum      : "
        f"{current_weights.sum():.10f}"
    )

    # =========================================================================
    # STEP 3 - TARGET PORTFOLIO
    # =========================================================================

    target_weights = (
        determine_target_weights(
            portfolio
        )
    )

    print(
        f"Target weight sum       : "
        f"{target_weights.sum():.10f}"
    )

    # =========================================================================
    # STEP 4 - BUILD TRANSITION
    # =========================================================================

    transition = (
        build_rebalancing_transition(
            dataframe=portfolio,
            current_weights=current_weights,
            target_weights=target_weights,
        )
    )

    # =========================================================================
    # STEP 5 - DISPLAY TRANSITION
    # =========================================================================

    print()
    print(
        "=" * 100
    )

    print(
        "PORTFOLIO STATE TRANSITION"
    )

    print(
        "=" * 100
    )

    print()

    display = transition[
        [
            "symbol",
            "trade_action",
            "current_weight",
            "target_weight",
            "weight_change",
            "rebalance_direction",
        ]
    ].copy()

    display[
        "current_weight"
    ] = (
        display[
            "current_weight"
        ]
        *
        100.0
    )

    display[
        "target_weight"
    ] = (
        display[
            "target_weight"
        ]
        *
        100.0
    )

    display[
        "weight_change"
    ] = (
        display[
            "weight_change"
        ]
        *
        100.0
    )

    display = display.rename(
        columns={
            "current_weight":
                "current_%",
            "target_weight":
                "target_%",
            "weight_change":
                "change_%",
        }
    )

    print(
        display.to_string(
            index=False
        )
    )

    # =========================================================================
    # STEP 6 - LOAD RETURNS
    # =========================================================================

    returns = (
        load_returns_matrix()
    )

    (
        aligned_returns,
        current_metric_weights,
        target_metric_weights,
    ) = align_weights_with_returns(
        transition=transition,
        returns=returns,
    )

    # =========================================================================
    # STEP 7 - CURRENT PORTFOLIO METRICS
    # =========================================================================

    current_metrics = (
        calculate_portfolio_metrics(
            returns=aligned_returns,
            weights=current_metric_weights,
        )
    )

    # =========================================================================
    # STEP 8 - TARGET PORTFOLIO METRICS
    # =========================================================================

    target_metrics = (
        calculate_portfolio_metrics(
            returns=aligned_returns,
            weights=target_metric_weights,
            previous_weights=current_metric_weights,
        )
    )

    # =========================================================================
    # STEP 9 - TURNOVER
    # =========================================================================

    total_turnover = (
        portfolio_turnover(
            previous_weights=current_metric_weights,
            new_weights=target_metric_weights,
        )
    )

    rebalance_counts = (
        transition[
            "rebalance_direction"
        ]
        .value_counts()
        .to_dict()
    )

    # =========================================================================
    # STEP 10 - DISPLAY METRICS
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "PORTFOLIO STATE METRICS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Turnover                : "
        f"{total_turnover:.6%}"
    )

    print()

    print(
        f"Rebalance BUY           : "
        f"{rebalance_counts.get('BUY', 0)}"
    )

    print(
        f"Rebalance SELL          : "
        f"{rebalance_counts.get('SELL', 0)}"
    )

    print(
        f"Rebalance HOLD          : "
        f"{rebalance_counts.get('HOLD', 0)}"
    )

    # -------------------------------------------------------------------------
    # CURRENT
    # -------------------------------------------------------------------------

    print()

    print(
        "Current portfolio"
    )

    print(
        "-" * 100
    )

    print(
        f"Cumulative return       : "
        f"{current_metrics['cumulative_return']:.6%}"
    )

    print(
        f"Annualized return       : "
        f"{current_metrics['annualized_return']:.6%}"
    )

    print(
        f"Annualized volatility   : "
        f"{current_metrics['annualized_volatility']:.6%}"
    )

    print(
        f"Sharpe ratio            : "
        f"{current_metrics['sharpe_ratio']:.6f}"
    )

    print(
        f"Max drawdown            : "
        f"{current_metrics['max_drawdown']:.6%}"
    )

    print(
        f"Historical VaR 95%      : "
        f"{current_metrics['historical_var_95']:.6%}"
    )

    # -------------------------------------------------------------------------
    # TARGET
    # -------------------------------------------------------------------------

    print()

    print(
        "Target portfolio"
    )

    print(
        "-" * 100
    )

    print(
        f"Cumulative return       : "
        f"{target_metrics['cumulative_return']:.6%}"
    )

    print(
        f"Annualized return       : "
        f"{target_metrics['annualized_return']:.6%}"
    )

    print(
        f"Annualized volatility   : "
        f"{target_metrics['annualized_volatility']:.6%}"
    )

    print(
        f"Sharpe ratio            : "
        f"{target_metrics['sharpe_ratio']:.6f}"
    )

    print(
        f"Max drawdown            : "
        f"{target_metrics['max_drawdown']:.6%}"
    )

    print(
        f"Historical VaR 95%      : "
        f"{target_metrics['historical_var_95']:.6%}"
    )

    # =========================================================================
    # STEP 11 - VALIDATION
    # =========================================================================

    checks = (
        validate_rebalancing_transition(
            transition
        )
    )

    checks[
        "turnover_finite"
    ] = bool(
        np.isfinite(
            total_turnover
        )
    )

    checks[
        "turnover_nonnegative"
    ] = bool(
        total_turnover
        >=
        0
    )

    checks[
        "current_metrics_finite"
    ] = bool(
        all(
            np.isfinite(
                float(value)
            )
            for value
            in current_metrics.values()
        )
    )

    checks[
        "target_metrics_finite"
    ] = bool(
        all(
            np.isfinite(
                float(value)
            )
            for value
            in target_metrics.values()
        )
    )

    # -------------------------------------------------------------------------
    # EXPECTED SELL RULE
    # -------------------------------------------------------------------------

    sell_rows = (
        transition[
            "trade_action"
        ]
        ==
        cfg.ACTION_SELL
    )

    if sell_rows.any():

        checks[
            "all_dqn_sell_stocks_exit"
        ] = bool(
            (
                transition.loc[
                    sell_rows,
                    "target_weight",
                ]
                <=
                cfg.FLOAT_TOLERANCE
            )
            .all()
        )

    else:

        checks[
            "all_dqn_sell_stocks_exit"
        ] = True

    overall_pass = all(
        checks.values()
    )

    # =========================================================================
    # STEP 12 - DISPLAY VALIDATION
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "REBALANCING ENVIRONMENT VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    for name, passed in checks.items():

        print(
            f"{name:<60}: "
            f"{passed}"
        )

    print()

    print(
        "VALIDATION RESULT: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    if not overall_pass:

        failed = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        raise ValueError(
            "Rebalancing environment validation failed:\n"
            +
            "\n".join(
                f"  - {name}"
                for name
                in failed
            )
        )

    # =========================================================================
    # STEP 13 - SAVE TRANSITION
    # =========================================================================

    REBALANCING_TRANSITION_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transition.to_csv(
        REBALANCING_TRANSITION_CSV,
        index=False,
    )

    # =========================================================================
    # STEP 14 - REPORT
    # =========================================================================

    report = {
        "status":
            "COMPLETE",

        "agent":
            "Portfolio Rebalancing Agent",

        "stage":
            "Portfolio state transition",

        "rebalance_date":
            cfg.FINAL_REBALANCE_DATE,

        "current_weight_source":
            current_weight_source,

        "stocks":
            int(
                len(transition)
            ),

        "agent4_actions": {
            "BUY":
                int(
                    action_counts.get(
                        "BUY",
                        0,
                    )
                ),

            "HOLD":
                int(
                    action_counts.get(
                        "HOLD",
                        0,
                    )
                ),

            "SELL":
                int(
                    action_counts.get(
                        "SELL",
                        0,
                    )
                ),
        },

        "rebalancing_actions": {
            "BUY":
                int(
                    rebalance_counts.get(
                        "BUY",
                        0,
                    )
                ),

            "SELL":
                int(
                    rebalance_counts.get(
                        "SELL",
                        0,
                    )
                ),

            "HOLD":
                int(
                    rebalance_counts.get(
                        "HOLD",
                        0,
                    )
                ),
        },

        "turnover":
            float(
                total_turnover
            ),

        "current_weight_sum":
            float(
                current_metric_weights.sum()
            ),

        "target_weight_sum":
            float(
                target_metric_weights.sum()
            ),

        "current_portfolio_metrics":
            metrics_to_json(
                current_metrics
            ),

        "target_portfolio_metrics":
            metrics_to_json(
                target_metrics
            ),

        "portfolio_transition": {
            str(
                row[
                    "symbol"
                ]
            ): {
                "agent4_action":
                    str(
                        row[
                            "trade_action"
                        ]
                    ),

                "current_weight":
                    float(
                        row[
                            "current_weight"
                        ]
                    ),

                "target_weight":
                    float(
                        row[
                            "target_weight"
                        ]
                    ),

                "weight_change":
                    float(
                        row[
                            "weight_change"
                        ]
                    ),

                "rebalance_direction":
                    str(
                        row[
                            "rebalance_direction"
                        ]
                    ),

                "turnover_contribution":
                    float(
                        row[
                            "turnover_contribution"
                        ]
                    ),
            }

            for _, row
            in transition.iterrows()
        },

        "validation": {
            name:
                bool(
                    passed
                )
            for name, passed
            in checks.items()
        },

        "output":
            str(
                REBALANCING_TRANSITION_CSV
            ),

        "methodology_note":
            (
                "The provisional current portfolio uses Agent-3 "
                "risk-adjusted weights when available. The target "
                "portfolio uses the Agent-5 MPT solution after Agent-4 "
                "DQN feedback. Agent-4 SELL actions receive zero target "
                "allocation. BUY and HOLD remain eligible for MPT "
                "allocation."
            ),

        "evaluation_note":
            (
                "Portfolio metrics are calculated over the same 105-day "
                "historical estimation window used to construct the "
                "MPT solution. They are diagnostic in-sample statistics "
                "and must not be interpreted as a future post-rebalance "
                "backtest."
            ),
    }

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REBALANCING_ENVIRONMENT_SUMMARY,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # =========================================================================
    # STEP 15 - GENERATED FILES
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "GENERATED FILES"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Portfolio transition    : "
        f"{REBALANCING_TRANSITION_CSV}"
    )

    print(
        f"Environment summary     : "
        f"{REBALANCING_ENVIRONMENT_SUMMARY}"
    )

    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "REBALANCING ENVIRONMENT STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Current portfolio state : COMPLETE"
    )

    print(
        "Target portfolio state  : COMPLETE"
    )

    print(
        "Weight transition       : COMPLETE"
    )

    print(
        "DQN SELL enforcement    : COMPLETE"
    )

    print(
        "Turnover calculation    : COMPLETE"
    )

    print(
        "Portfolio metrics       : COMPLETE"
    )

    print(
        "State validation        : COMPLETE"
    )

    print()

    print(
        "REBALANCING ENVIRONMENT STATUS: COMPLETE"
    )

    return {
        "portfolio":
            portfolio,

        "transition":
            transition,

        "returns":
            aligned_returns,

        "current_weights":
            current_metric_weights,

        "target_weights":
            target_metric_weights,

        "current_metrics":
            current_metrics,

        "target_metrics":
            target_metrics,

        "turnover":
            total_turnover,

        "checks":
            checks,

        "report":
            report,
    }


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    run_rebalancing_environment()