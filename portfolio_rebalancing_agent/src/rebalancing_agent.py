"""
Agent 5 - Portfolio Rebalancing Agent
===============================================================================

PROJECT
-------
Agentic AI Portfolio Optimization and Algorithmic Trading

AGENT
-----
Agent 5 - Portfolio Rebalancing Agent

PURPOSE
-------
Create the final portfolio allocation after:

    Agent 4 DQN trade execution
            ↓
    Agent 5 Markowitz MPT optimization
            ↓
    RL feedback
            ↓
    Portfolio state transition
            ↓
    FINAL REBALANCED PORTFOLIO


THIS MODULE
-----------
This file is the high-level production layer for Agent 5.

It does NOT retrain any model.

It verifies that all previous Agent-5 stages completed successfully,
loads the final target portfolio state, validates the allocation and
writes the final Agent-5 output files.


FINAL OUTPUTS
-------------
1. outputs/rebalanced_portfolio.csv
2. outputs/rebalanced_portfolio.parquet
3. outputs/final_portfolio_allocation.csv
4. reports/agent5_summary.json


IMPORTANT
---------
Portfolio metrics calculated here use the historical 105-day estimation
window used to construct the final 2025-12-01 allocation.

They are diagnostic historical statistics.

They are NOT a future post-rebalance backtest.
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
)


# =============================================================================
# INPUT PATHS
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
# REQUIRED REPORTS
# =============================================================================

REQUIRED_STAGE_REPORTS = {
    "data_loader":
        cfg.DATA_LOADER_SUMMARY,

    "mpt_optimizer":
        cfg.MPT_OPTIMIZATION_SUMMARY,

    "rl_feedback":
        cfg.RL_FEEDBACK_SUMMARY,

    "rebalancing_environment":
        REBALANCING_ENVIRONMENT_SUMMARY,
}


# =============================================================================
# JSON LOADER
# =============================================================================

def load_json_report(
    path: Path,
    stage_name: str,
) -> dict[str, Any]:
    """
    Load and validate one Agent-5 stage report.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{stage_name} report was not found.\n"
            f"Expected:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(
            file
        )

    status = (
        str(
            report.get(
                "status",
                "",
            )
        )
        .upper()
        .strip()
    )

    if status not in {
        "COMPLETE",
        "PASS",
    }:

        raise ValueError(
            f"{stage_name} did not complete successfully.\n"
            f"Status: {status}"
        )

    return report


# =============================================================================
# VERIFY AGENT-5 PIPELINE
# =============================================================================

def verify_previous_stages() -> dict[str, dict[str, Any]]:
    """
    Verify that all required Agent-5 stages completed.
    """

    reports: dict[
        str,
        dict[str, Any],
    ] = {}

    for stage_name, path in REQUIRED_STAGE_REPORTS.items():

        reports[
            stage_name
        ] = load_json_report(
            path=path,
            stage_name=stage_name,
        )

    return reports


# =============================================================================
# LOAD REBALANCING TRANSITION
# =============================================================================

def load_rebalancing_transition() -> pd.DataFrame:
    """
    Load final current -> target portfolio transition.
    """

    if not REBALANCING_TRANSITION_CSV.exists():

        raise FileNotFoundError(
            "Rebalancing transition was not found.\n\n"
            f"Expected:\n"
            f"{REBALANCING_TRANSITION_CSV}\n\n"
            "Run rebalancing_environment.py first."
        )

    dataframe = pd.read_csv(
        REBALANCING_TRANSITION_CSV,
        low_memory=False,
    )

    if dataframe.empty:

        raise ValueError(
            "Rebalancing transition is empty."
        )

    required_columns = {
        "symbol",
        "trade_action",
        "current_weight",
        "target_weight",
        "weight_change",
        "rebalance_direction",
    }

    missing_columns = (
        required_columns
        -
        set(
            dataframe.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "Rebalancing transition is missing columns:\n"
            +
            "\n".join(
                sorted(
                    missing_columns
                )
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
    # ACTIONS
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

    dataframe[
        "rebalance_direction"
    ] = (
        dataframe[
            "rebalance_direction"
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
            "current_weight",
            "target_weight",
            "weight_change",
            "absolute_weight_change",
            "turnover_contribution",
            "mpt_weight",
            "rl_adjusted_weight",
            "q_hold",
            "q_buy",
            "q_sell",
            "selected_q_value",
            "agent3_score",
            "agent3_rank",
            "risk_adjusted_weight",
            "annualized_expected_return",
            "annualized_volatility",
        ]
        if column
        in dataframe.columns
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
    # SYMBOL VALIDATION
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
            "Duplicate symbols in rebalancing transition:\n"
            +
            "\n".join(
                sorted(
                    set(
                        duplicates
                    )
                )
            )
        )

    # -------------------------------------------------------------------------
    # WEIGHT VALIDATION
    # -------------------------------------------------------------------------

    for column in [
        "current_weight",
        "target_weight",
        "weight_change",
    ]:

        if dataframe[
            column
        ].isna().any():

            raise ValueError(
                f"{column} contains invalid values."
            )

    return dataframe


# =============================================================================
# BUILD FINAL REBALANCED PORTFOLIO
# =============================================================================

def build_final_portfolio(
    transition: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build final Agent-5 portfolio.

    final_weight = target_weight
    """

    final_portfolio = (
        transition
        .copy()
    )

    # =========================================================================
    # FINAL WEIGHT
    # =========================================================================

    final_portfolio[
        "final_weight"
    ] = (
        final_portfolio[
            "target_weight"
        ]
        .astype(float)
    )

    final_portfolio[
        "final_allocation_percent"
    ] = (
        final_portfolio[
            "final_weight"
        ]
        *
        100.0
    )

    # =========================================================================
    # CURRENT / TARGET %
    # =========================================================================

    final_portfolio[
        "current_allocation_percent"
    ] = (
        final_portfolio[
            "current_weight"
        ]
        *
        100.0
    )

    final_portfolio[
        "allocation_change_percent"
    ] = (
        final_portfolio[
            "weight_change"
        ]
        *
        100.0
    )

    # =========================================================================
    # ACTIVE / EXITED
    # =========================================================================

    final_portfolio[
        "portfolio_status"
    ] = np.where(
        final_portfolio[
            "final_weight"
        ]
        >
        cfg.FLOAT_TOLERANCE,

        "ACTIVE",

        "EXIT",
    )

    # =========================================================================
    # FINAL RANK
    # =========================================================================

    final_portfolio[
        "final_allocation_rank"
    ] = (
        final_portfolio[
            "final_weight"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    # =========================================================================
    # FINAL ACTION DESCRIPTION
    # =========================================================================

    final_portfolio[
        "final_decision"
    ] = np.select(
        [
            final_portfolio[
                "trade_action"
            ]
            ==
            cfg.ACTION_SELL,

            final_portfolio[
                "rebalance_direction"
            ]
            ==
            "BUY",

            final_portfolio[
                "rebalance_direction"
            ]
            ==
            "SELL",
        ],
        [
            "EXIT_POSITION",
            "INCREASE_ALLOCATION",
            "REDUCE_ALLOCATION",
        ],
        default="MAINTAIN_ALLOCATION",
    )

    # =========================================================================
    # SORT
    # =========================================================================

    final_portfolio = (
        final_portfolio
        .sort_values(
            [
                "final_weight",
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

    return final_portfolio


# =============================================================================
# FINAL ALLOCATION TABLE
# =============================================================================

def build_final_allocation_table(
    final_portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a clean presentation/output table.
    """

    preferred_columns = [
        "final_allocation_rank",
        "symbol",
        "trade_action",
        "rebalance_direction",
        "current_weight",
        "final_weight",
        "current_allocation_percent",
        "final_allocation_percent",
        "allocation_change_percent",
        "portfolio_status",
        "final_decision",
        "q_hold",
        "q_buy",
        "q_sell",
        "selected_q_value",
        "agent2_rank",
        "agent3_rank",
        "agent3_score",
        "risk_level",
        "risk_decision",
    ]

    columns = [
        column
        for column in preferred_columns
        if column
        in final_portfolio.columns
    ]

    return (
        final_portfolio[
            columns
        ]
        .copy()
    )


# =============================================================================
# ALIGN FINAL WEIGHTS WITH RETURNS
# =============================================================================

def align_final_weights(
    final_portfolio: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    """
    Align final portfolio weights with return matrix.
    """

    portfolio_symbols = set(
        final_portfolio[
            "symbol"
        ]
        .astype(str)
    )

    return_symbols = set(
        returns.columns
        .astype(str)
    )

    missing = (
        portfolio_symbols
        -
        return_symbols
    )

    if missing:

        raise ValueError(
            "Final portfolio symbols are missing from "
            "returns matrix:\n"
            +
            "\n".join(
                sorted(
                    missing
                )
            )
        )

    ordered_symbols = [
        str(
            symbol
        )
        for symbol
        in returns.columns

        if str(
            symbol
        )
        in portfolio_symbols
    ]

    aligned_returns = (
        returns[
            ordered_symbols
        ]
        .copy()
    )

    indexed_portfolio = (
        final_portfolio
        .set_index(
            "symbol"
        )
    )

    final_weights = np.array(
        indexed_portfolio.loc[
            ordered_symbols,
            "final_weight",
        ],
        dtype=float,
        copy=True,
    )

    return (
        aligned_returns,
        final_weights,
    )


# =============================================================================
# VALIDATE FINAL PORTFOLIO
# =============================================================================

def validate_final_portfolio(
    final_portfolio: pd.DataFrame,
) -> dict[str, bool]:
    """
    Validate final Agent-5 portfolio.
    """

    weights = np.array(
        final_portfolio[
            "final_weight"
        ],
        dtype=float,
        copy=True,
    )

    sell_mask = (
        final_portfolio[
            "trade_action"
        ]
        ==
        cfg.ACTION_SELL
    )

    active_mask = (
        final_portfolio[
            "final_weight"
        ]
        >
        cfg.FLOAT_TOLERANCE
    )

    checks: dict[str, bool] = {}

    checks[
        "portfolio_not_empty"
    ] = (
        not final_portfolio.empty
    )

    checks[
        "one_row_per_symbol"
    ] = (
        not final_portfolio[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    checks[
        "weights_finite"
    ] = bool(
        np.isfinite(
            weights
        ).all()
    )

    checks[
        "weights_nonnegative"
    ] = bool(
        (
            weights
            >=
            -cfg.FLOAT_TOLERANCE
        ).all()
    )

    checks[
        "weights_below_max"
    ] = bool(
        (
            weights
            <=
            (
                cfg.MAX_WEIGHT
                +
                cfg.FLOAT_TOLERANCE
            )
        ).all()
    )

    checks[
        "weights_sum_to_one"
    ] = bool(
        np.isclose(
            weights.sum(),
            cfg.WEIGHT_SUM_TARGET,
            atol=cfg.WEIGHT_SUM_TOLERANCE,
        )
    )

    checks[
        "agent4_actions_valid"
    ] = bool(
        final_portfolio[
            "trade_action"
        ]
        .isin(
            cfg.VALID_ACTIONS
        )
        .all()
    )

    checks[
        "rebalance_directions_valid"
    ] = bool(
        final_portfolio[
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

    # =========================================================================
    # SELL -> ZERO
    # =========================================================================

    if sell_mask.any():

        checks[
            "dqn_sell_weights_zero"
        ] = bool(
            np.allclose(
                final_portfolio.loc[
                    sell_mask,
                    "final_weight",
                ]
                .to_numpy(
                    dtype=float
                ),

                cfg.SELL_TARGET_WEIGHT,

                atol=cfg.FLOAT_TOLERANCE,
            )
        )

        checks[
            "dqn_sell_positions_exited"
        ] = bool(
            (
                final_portfolio.loc[
                    sell_mask,
                    "portfolio_status",
                ]
                ==
                "EXIT"
            )
            .all()
        )

    else:

        checks[
            "dqn_sell_weights_zero"
        ] = True

        checks[
            "dqn_sell_positions_exited"
        ] = True

    # =========================================================================
    # ACTIVE ASSETS
    # =========================================================================

    checks[
        "active_assets_exist"
    ] = bool(
        active_mask.any()
    )

    checks[
        "active_assets_are_not_dqn_sell"
    ] = bool(
        (
            final_portfolio.loc[
                active_mask,
                "trade_action",
            ]
            !=
            cfg.ACTION_SELL
        )
        .all()
    )

    # =========================================================================
    # TARGET = FINAL
    # =========================================================================

    checks[
        "final_weights_match_target"
    ] = bool(
        np.allclose(
            final_portfolio[
                "final_weight"
            ]
            .to_numpy(
                dtype=float
            ),

            final_portfolio[
                "target_weight"
            ]
            .to_numpy(
                dtype=float
            ),

            atol=cfg.FLOAT_TOLERANCE,
        )
    )

    return checks


# =============================================================================
# JSON-SAFE METRICS
# =============================================================================

def make_json_safe(
    mapping: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert NumPy values to Python-native JSON values.
    """

    result: dict[
        str,
        Any,
    ] = {}

    for key, value in mapping.items():

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
# SAVE PARQUET WITH FALLBACK
# =============================================================================

def save_final_outputs(
    final_portfolio: pd.DataFrame,
    allocation_table: pd.DataFrame,
) -> dict[str, Any]:
    """
    Save final Agent-5 output files.
    """

    cfg.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # COMPLETE CSV
    # -------------------------------------------------------------------------

    final_portfolio.to_csv(
        cfg.REBALANCED_PORTFOLIO_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # PARQUET
    # -------------------------------------------------------------------------

    parquet_saved = False
    parquet_error = None

    try:

        final_portfolio.to_parquet(
            cfg.REBALANCED_PORTFOLIO_PARQUET,
            index=False,
        )

        parquet_saved = True

    except Exception as error:

        parquet_error = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    # -------------------------------------------------------------------------
    # CLEAN FINAL ALLOCATION
    # -------------------------------------------------------------------------

    allocation_table.to_csv(
        cfg.FINAL_PORTFOLIO_ALLOCATION_CSV,
        index=False,
    )

    return {
        "rebalanced_portfolio_csv":
            str(
                cfg.REBALANCED_PORTFOLIO_CSV
            ),

        "rebalanced_portfolio_csv_saved":
            cfg.REBALANCED_PORTFOLIO_CSV.exists(),

        "rebalanced_portfolio_parquet":
            str(
                cfg.REBALANCED_PORTFOLIO_PARQUET
            ),

        "rebalanced_portfolio_parquet_saved":
            parquet_saved,

        "rebalanced_portfolio_parquet_error":
            parquet_error,

        "final_portfolio_allocation_csv":
            str(
                cfg.FINAL_PORTFOLIO_ALLOCATION_CSV
            ),

        "final_portfolio_allocation_saved":
            cfg.FINAL_PORTFOLIO_ALLOCATION_CSV.exists(),
    }


# =============================================================================
# COMPLETE AGENT 5
# =============================================================================

def run_rebalancing_agent() -> pd.DataFrame:
    """
    Execute the final Agent-5 portfolio construction stage.

    This does NOT rerun MPT optimization.

    It consumes the previously validated stage outputs and produces the
    final portfolio allocation.
    """

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 5 - PORTFOLIO REBALANCING AGENT"
    )

    print(
        "=" * 100
    )

    print()

    # =========================================================================
    # STEP 1 - VERIFY PREVIOUS STAGES
    # =========================================================================

    print(
        "Verifying previous Agent-5 stages..."
    )

    stage_reports = (
        verify_previous_stages()
    )

    for stage_name in REQUIRED_STAGE_REPORTS:

        print(
            f"{stage_name:<30}: COMPLETE"
        )

    # =========================================================================
    # STEP 2 - LOAD FINAL TRANSITION
    # =========================================================================

    print()

    transition = (
        load_rebalancing_transition()
    )

    print(
        f"Portfolio stocks        : "
        f"{len(transition)}"
    )

    # =========================================================================
    # STEP 3 - BUILD FINAL PORTFOLIO
    # =========================================================================

    final_portfolio = (
        build_final_portfolio(
            transition
        )
    )

    allocation_table = (
        build_final_allocation_table(
            final_portfolio
        )
    )

    # =========================================================================
    # STEP 4 - VALIDATE
    # =========================================================================

    checks = (
        validate_final_portfolio(
            final_portfolio
        )
    )

    overall_pass = all(
        checks.values()
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL PORTFOLIO ALLOCATION"
    )

    print(
        "=" * 100
    )

    print()

    display = (
        final_portfolio[
            [
                "symbol",
                "trade_action",
                "rebalance_direction",
                "final_weight",
                "portfolio_status",
                "final_decision",
            ]
        ]
        .copy()
    )

    display[
        "final_weight"
    ] = (
        display[
            "final_weight"
        ]
        *
        100.0
    )

    display = display.rename(
        columns={
            "final_weight":
                "final_weight_%",
        }
    )

    print(
        display.to_string(
            index=False
        )
    )

    # =========================================================================
    # STEP 5 - PORTFOLIO METRICS
    # =========================================================================

    returns = (
        load_returns_matrix()
    )

    (
        aligned_returns,
        final_weights,
    ) = align_final_weights(
        final_portfolio=final_portfolio,
        returns=returns,
    )

    final_metrics = (
        calculate_portfolio_metrics(
            returns=aligned_returns,
            weights=final_weights,
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL PORTFOLIO DIAGNOSTIC METRICS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Cumulative return       : "
        f"{final_metrics['cumulative_return']:.6%}"
    )

    print(
        f"Annualized return       : "
        f"{final_metrics['annualized_return']:.6%}"
    )

    print(
        f"Annualized volatility   : "
        f"{final_metrics['annualized_volatility']:.6%}"
    )

    print(
        f"Sharpe ratio            : "
        f"{final_metrics['sharpe_ratio']:.6f}"
    )

    print(
        f"Maximum drawdown        : "
        f"{final_metrics['max_drawdown']:.6%}"
    )

    print(
        f"Historical VaR 95%      : "
        f"{final_metrics['historical_var_95']:.6%}"
    )

    print(
        f"Effective assets        : "
        f"{final_metrics['effective_number_of_assets']:.4f}"
    )

    # =========================================================================
    # STEP 6 - VALIDATION DISPLAY
    # =========================================================================

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL PORTFOLIO VALIDATION"
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
            "Final Agent-5 portfolio validation failed:\n"
            +
            "\n".join(
                f"  - {name}"
                for name
                in failed
            )
        )

    # =========================================================================
    # STEP 7 - SAVE OUTPUTS
    # =========================================================================

    output_info = (
        save_final_outputs(
            final_portfolio=final_portfolio,
            allocation_table=allocation_table,
        )
    )

    # =========================================================================
    # STEP 8 - SUMMARY
    # =========================================================================

    action_counts = (
        final_portfolio[
            "trade_action"
        ]
        .value_counts()
        .to_dict()
    )

    rebalance_counts = (
        final_portfolio[
            "rebalance_direction"
        ]
        .value_counts()
        .to_dict()
    )

    active_assets = (
        final_portfolio.loc[
            final_portfolio[
                "final_weight"
            ]
            >
            cfg.FLOAT_TOLERANCE
        ]
        .copy()
    )

    exited_assets = (
        final_portfolio.loc[
            final_portfolio[
                "final_weight"
            ]
            <=
            cfg.FLOAT_TOLERANCE
        ]
        .copy()
    )

    summary = {
        "status":
            "COMPLETE",

        "agent":
            "Portfolio Rebalancing Agent",

        "agent_number":
            5,

        "rebalance_date":
            cfg.FINAL_REBALANCE_DATE,

        "method":
            "Markowitz MPT + Agent-4 DQN RL feedback",

        "stocks_received":
            int(
                len(
                    final_portfolio
                )
            ),

        "active_assets":
            int(
                len(
                    active_assets
                )
            ),

        "exited_assets":
            int(
                len(
                    exited_assets
                )
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

            "HOLD":
                int(
                    rebalance_counts.get(
                        "HOLD",
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
        },

        "final_weight_sum":
            float(
                final_portfolio[
                    "final_weight"
                ]
                .sum()
            ),

        "final_portfolio": {
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

                "rebalance_direction":
                    str(
                        row[
                            "rebalance_direction"
                        ]
                    ),

                "weight":
                    float(
                        row[
                            "final_weight"
                        ]
                    ),

                "allocation_percent":
                    float(
                        row[
                            "final_allocation_percent"
                        ]
                    ),

                "status":
                    str(
                        row[
                            "portfolio_status"
                        ]
                    ),

                "decision":
                    str(
                        row[
                            "final_decision"
                        ]
                    ),
            }

            for _, row
            in final_portfolio.iterrows()
        },

        "diagnostic_metrics":
            make_json_safe(
                final_metrics
            ),

        "validation": {
            name:
                bool(
                    passed
                )

            for name, passed
            in checks.items()
        },

        "stage_reports": {
            stage_name:
                str(
                    path
                )

            for stage_name, path
            in REQUIRED_STAGE_REPORTS.items()
        },

        "outputs":
            output_info,

        "methodology_note":
            (
                "Agent 5 combines Markowitz minimum-variance portfolio "
                "optimization with Agent-4 DQN trade feedback. Agent-4 "
                "SELL stocks are assigned zero final allocation. BUY and "
                "HOLD stocks remain eligible and receive MPT-determined "
                "weights. No additional unreported BUY/HOLD multipliers "
                "are introduced."
            ),

        "evaluation_note":
            (
                "The reported portfolio metrics use the same historical "
                "105-day period used for return/covariance estimation. "
                "They describe the historical properties of the final "
                "allocation and are not a post-2025-12-01 future backtest."
            ),
    }

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        cfg.AGENT5_SUMMARY,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # =========================================================================
    # GENERATED FILES
    # =========================================================================

    print()
    print(
        "=" * 100
    )

    print(
        "GENERATED FINAL AGENT-5 FILES"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Rebalanced portfolio CSV : "
        f"{cfg.REBALANCED_PORTFOLIO_CSV}"
    )

    print(
        f"Rebalanced portfolio PQ  : "
        f"{cfg.REBALANCED_PORTFOLIO_PARQUET}"
    )

    print(
        f"Final allocation         : "
        f"{cfg.FINAL_PORTFOLIO_ALLOCATION_CSV}"
    )

    print(
        f"Agent-5 summary          : "
        f"{cfg.AGENT5_SUMMARY}"
    )

    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    print()
    print(
        "=" * 100
    )

    print(
        "PORTFOLIO REBALANCING AGENT STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Data preparation         : COMPLETE"
    )

    print(
        "Portfolio metrics        : COMPLETE"
    )

    print(
        "MPT optimization         : COMPLETE"
    )

    print(
        "RL feedback              : COMPLETE"
    )

    print(
        "Portfolio transition     : COMPLETE"
    )

    print(
        "Final allocation         : COMPLETE"
    )

    print(
        "Output persistence       : COMPLETE"
    )

    print(
        "Portfolio validation     : COMPLETE"
    )

    print()

    print(
        "AGENT 5 CORE STATUS: COMPLETE"
    )

    return final_portfolio


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_agent5() -> pd.DataFrame:
    """
    Convenience wrapper.
    """

    return run_rebalancing_agent()


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    run_rebalancing_agent()