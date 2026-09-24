"""
Agent 5 - Portfolio Evaluator
===============================================================================

PROJECT
-------
Agentic AI Portfolio Optimization and Algorithmic Trading

AGENT
-----
Agent 5 - Portfolio Rebalancing Agent

PURPOSE
-------
Evaluate the final portfolio produced by Agent 5.

The evaluator compares:

    1. Current / provisional portfolio
       derived from Agent-3 risk-adjusted weights

    2. Equal-weight benchmark
       across final active assets

    3. Final Agent-5 portfolio
       Markowitz MPT + Agent-4 DQN feedback


METRICS
-------
- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Historical VaR 95%
- Portfolio turnover
- Herfindahl concentration index
- Effective number of assets
- Minimum / maximum weight


IMPORTANT METHODOLOGY NOTE
--------------------------
The available return matrix contains the historical 105-day period used
for Agent-5 MPT estimation.

Therefore the metrics in this file are historical diagnostic metrics.

They are NOT a future out-of-sample post-2025-12-01 portfolio backtest.

A true future backtest would require market observations after the
rebalance date.
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
# LOCAL OUTPUTS
# =============================================================================

EVALUATION_ALLOCATION_COMPARISON = (
    cfg.REPORT_DIR
    / "evaluation_allocation_comparison.csv"
)


# =============================================================================
# JSON-SAFE CONVERSION
# =============================================================================

def make_json_safe(
    mapping: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert NumPy scalar objects into ordinary Python values.
    """

    result: dict[str, Any] = {}

    for key, value in mapping.items():

        if isinstance(
            value,
            (
                np.integer,
                int,
            ),
        ):

            result[key] = int(
                value
            )

        elif isinstance(
            value,
            (
                np.floating,
                float,
            ),
        ):

            result[key] = float(
                value
            )

        elif isinstance(
            value,
            np.bool_,
        ):

            result[key] = bool(
                value
            )

        else:

            result[key] = value

    return result


# =============================================================================
# LOAD FINAL AGENT-5 PORTFOLIO
# =============================================================================

def load_final_portfolio() -> tuple[
    pd.DataFrame,
    str,
]:
    """
    Load final Agent-5 rebalanced portfolio.

    Prefer parquet.
    Fall back to CSV.
    """

    if cfg.REBALANCED_PORTFOLIO_PARQUET.exists():

        dataframe = pd.read_parquet(
            cfg.REBALANCED_PORTFOLIO_PARQUET
        )

        source = "PARQUET"

    elif cfg.REBALANCED_PORTFOLIO_CSV.exists():

        dataframe = pd.read_csv(
            cfg.REBALANCED_PORTFOLIO_CSV,
            low_memory=False,
        )

        source = "CSV"

    else:

        raise FileNotFoundError(
            "Final Agent-5 portfolio was not found.\n\n"
            f"Parquet:\n"
            f"{cfg.REBALANCED_PORTFOLIO_PARQUET}\n\n"
            f"CSV:\n"
            f"{cfg.REBALANCED_PORTFOLIO_CSV}\n\n"
            "Run rebalancing_agent.py first."
        )

    if dataframe.empty:

        raise ValueError(
            "Final Agent-5 portfolio is empty."
        )

    required_columns = {
        "symbol",
        "trade_action",
        "current_weight",
        "final_weight",
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
            "Final portfolio is missing columns:\n"
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
    # NUMERIC WEIGHTS
    # -------------------------------------------------------------------------

    for column in [
        "current_weight",
        "final_weight",
    ]:

        dataframe[
            column
        ] = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        )

    if dataframe[
        [
            "current_weight",
            "final_weight",
        ]
    ].isna().any().any():

        raise ValueError(
            "Portfolio contains invalid current/final weights."
        )

    if dataframe[
        "symbol"
    ].duplicated().any():

        raise ValueError(
            "Final portfolio contains duplicate symbols."
        )

    if not dataframe[
        "trade_action"
    ].isin(
        cfg.VALID_ACTIONS
    ).all():

        raise ValueError(
            "Final portfolio contains invalid Agent-4 actions."
        )

    return (
        dataframe,
        source,
    )


# =============================================================================
# ALIGN PORTFOLIO AND RETURNS
# =============================================================================

def align_portfolio_and_returns(
    portfolio: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """
    Align the portfolio rows with the historical return-matrix columns.
    """

    portfolio_symbols = set(
        portfolio[
            "symbol"
        ]
        .astype(str)
    )

    return_symbols = set(
        returns.columns
        .astype(str)
    )

    missing_symbols = (
        portfolio_symbols
        -
        return_symbols
    )

    if missing_symbols:

        raise ValueError(
            "Final portfolio symbols are missing from "
            "the return matrix:\n"
            +
            "\n".join(
                sorted(
                    missing_symbols
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
        portfolio
        .set_index(
            "symbol"
        )
    )

    aligned_portfolio = (
        indexed_portfolio
        .loc[
            ordered_symbols
        ]
        .reset_index()
    )

    current_weights = np.array(
        aligned_portfolio[
            "current_weight"
        ],
        dtype=float,
        copy=True,
    )

    final_weights = np.array(
        aligned_portfolio[
            "final_weight"
        ],
        dtype=float,
        copy=True,
    )

    return (
        aligned_portfolio,
        aligned_returns,
        current_weights,
        final_weights,
    )


# =============================================================================
# EQUAL-WEIGHT ACTIVE-ASSET BENCHMARK
# =============================================================================

def build_active_equal_weight_benchmark(
    portfolio: pd.DataFrame,
) -> np.ndarray:
    """
    Construct equal weighting across assets that remain active
    in the final Agent-5 portfolio.

    Final zero-weight / exited assets receive zero benchmark weight.
    """

    active_mask = (
        portfolio[
            "final_weight"
        ]
        .to_numpy(
            dtype=float
        )
        >
        cfg.FLOAT_TOLERANCE
    )

    active_count = int(
        active_mask.sum()
    )

    if active_count <= 0:

        raise ValueError(
            "Cannot construct benchmark because "
            "the final portfolio has no active assets."
        )

    weights = np.zeros(
        len(
            portfolio
        ),
        dtype=float,
    )

    weights[
        active_mask
    ] = (
        cfg.WEIGHT_SUM_TARGET
        /
        active_count
    )

    return weights


# =============================================================================
# METRIC DIFFERENCES
# =============================================================================

def calculate_metric_changes(
    current_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
) -> dict[str, float]:
    """
    Calculate descriptive changes from provisional portfolio
    to final Agent-5 allocation.

    No claim is made that a positive/negative change is automatically
    desirable; these values are reported descriptively.
    """

    return {
        "cumulative_return_change":
            float(
                final_metrics[
                    "cumulative_return"
                ]
                -
                current_metrics[
                    "cumulative_return"
                ]
            ),

        "annualized_return_change":
            float(
                final_metrics[
                    "annualized_return"
                ]
                -
                current_metrics[
                    "annualized_return"
                ]
            ),

        "annualized_volatility_change":
            float(
                final_metrics[
                    "annualized_volatility"
                ]
                -
                current_metrics[
                    "annualized_volatility"
                ]
            ),

        "sharpe_ratio_change":
            float(
                final_metrics[
                    "sharpe_ratio"
                ]
                -
                current_metrics[
                    "sharpe_ratio"
                ]
            ),

        "max_drawdown_change":
            float(
                final_metrics[
                    "max_drawdown"
                ]
                -
                current_metrics[
                    "max_drawdown"
                ]
            ),

        "historical_var_95_change":
            float(
                final_metrics[
                    "historical_var_95"
                ]
                -
                current_metrics[
                    "historical_var_95"
                ]
            ),

        "effective_assets_change":
            float(
                final_metrics[
                    "effective_number_of_assets"
                ]
                -
                current_metrics[
                    "effective_number_of_assets"
                ]
            ),
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_evaluation(
    portfolio: pd.DataFrame,
    returns: pd.DataFrame,
    current_weights: np.ndarray,
    benchmark_weights: np.ndarray,
    final_weights: np.ndarray,
    current_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    turnover: float,
) -> dict[str, bool]:
    """
    Validate final Agent-5 evaluation.
    """

    sell_mask = (
        portfolio[
            "trade_action"
        ]
        ==
        cfg.ACTION_SELL
    )

    active_mask = (
        portfolio[
            "final_weight"
        ]
        >
        cfg.FLOAT_TOLERANCE
    )

    checks: dict[str, bool] = {}

    # -------------------------------------------------------------------------
    # PORTFOLIO STRUCTURE
    # -------------------------------------------------------------------------

    checks[
        "portfolio_not_empty"
    ] = (
        not portfolio.empty
    )

    checks[
        "one_row_per_symbol"
    ] = (
        not portfolio[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    checks[
        "returns_not_empty"
    ] = (
        not returns.empty
    )

    checks[
        "return_observations_valid"
    ] = (
        len(
            returns
        )
        >=
        cfg.MIN_REQUIRED_RETURN_OBSERVATIONS
    )

    checks[
        "return_lookback_respected"
    ] = (
        len(
            returns
        )
        <=
        cfg.ESTIMATION_LOOKBACK_DAYS
    )

    # -------------------------------------------------------------------------
    # NO FUTURE DATA
    # -------------------------------------------------------------------------

    checks[
        "no_future_returns"
    ] = bool(
        (
            returns.index
            <=
            pd.Timestamp(
                cfg.FINAL_REBALANCE_DATE
            )
        )
        .all()
    )

    # -------------------------------------------------------------------------
    # CURRENT WEIGHTS
    # -------------------------------------------------------------------------

    checks[
        "current_weights_finite"
    ] = bool(
        np.isfinite(
            current_weights
        )
        .all()
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
    # BENCHMARK
    # -------------------------------------------------------------------------

    checks[
        "benchmark_weights_finite"
    ] = bool(
        np.isfinite(
            benchmark_weights
        )
        .all()
    )

    checks[
        "benchmark_weights_sum_to_one"
    ] = bool(
        np.isclose(
            benchmark_weights.sum(),
            cfg.WEIGHT_SUM_TARGET,
            atol=cfg.WEIGHT_SUM_TOLERANCE,
        )
    )

    # -------------------------------------------------------------------------
    # FINAL WEIGHTS
    # -------------------------------------------------------------------------

    checks[
        "final_weights_finite"
    ] = bool(
        np.isfinite(
            final_weights
        )
        .all()
    )

    checks[
        "final_weights_nonnegative"
    ] = bool(
        (
            final_weights
            >=
            -cfg.FLOAT_TOLERANCE
        )
        .all()
    )

    checks[
        "final_weights_sum_to_one"
    ] = bool(
        np.isclose(
            final_weights.sum(),
            cfg.WEIGHT_SUM_TARGET,
            atol=cfg.WEIGHT_SUM_TOLERANCE,
        )
    )

    # -------------------------------------------------------------------------
    # AGENT-4 SELL RULE
    # -------------------------------------------------------------------------

    if sell_mask.any():

        checks[
            "dqn_sell_final_weights_zero"
        ] = bool(
            np.allclose(
                portfolio.loc[
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

    else:

        checks[
            "dqn_sell_final_weights_zero"
        ] = True

    checks[
        "active_assets_exist"
    ] = bool(
        active_mask.any()
    )

    checks[
        "active_assets_not_dqn_sell"
    ] = bool(
        (
            portfolio.loc[
                active_mask,
                "trade_action",
            ]
            !=
            cfg.ACTION_SELL
        )
        .all()
    )

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------

    checks[
        "current_metrics_finite"
    ] = bool(
        all(
            np.isfinite(
                float(
                    value
                )
            )

            for value
            in current_metrics.values()
        )
    )

    checks[
        "benchmark_metrics_finite"
    ] = bool(
        all(
            np.isfinite(
                float(
                    value
                )
            )

            for value
            in benchmark_metrics.values()
        )
    )

    checks[
        "final_metrics_finite"
    ] = bool(
        all(
            np.isfinite(
                float(
                    value
                )
            )

            for value
            in final_metrics.values()
        )
    )

    # -------------------------------------------------------------------------
    # TURNOVER
    # -------------------------------------------------------------------------

    checks[
        "turnover_finite"
    ] = bool(
        np.isfinite(
            turnover
        )
    )

    checks[
        "turnover_nonnegative"
    ] = bool(
        turnover
        >=
        0
    )

    return checks


# =============================================================================
# ALLOCATION COMPARISON TABLE
# =============================================================================

def build_allocation_comparison(
    portfolio: pd.DataFrame,
    benchmark_weights: np.ndarray,
) -> pd.DataFrame:
    """
    Build stock-level comparison between current, benchmark and final
    portfolio weights.
    """

    comparison = (
        portfolio
        .copy()
    )

    comparison[
        "equal_weight_benchmark"
    ] = (
        benchmark_weights
    )

    comparison[
        "current_allocation_percent"
    ] = (
        comparison[
            "current_weight"
        ]
        *
        100.0
    )

    comparison[
        "benchmark_allocation_percent"
    ] = (
        comparison[
            "equal_weight_benchmark"
        ]
        *
        100.0
    )

    comparison[
        "final_allocation_percent"
    ] = (
        comparison[
            "final_weight"
        ]
        *
        100.0
    )

    preferred_columns = [
        "symbol",
        "trade_action",
        "rebalance_direction",
        "current_weight",
        "equal_weight_benchmark",
        "final_weight",
        "current_allocation_percent",
        "benchmark_allocation_percent",
        "final_allocation_percent",
        "risk_level",
        "agent3_rank",
        "portfolio_status",
        "final_decision",
    ]

    columns = [
        column

        for column
        in preferred_columns

        if column
        in comparison.columns
    ]

    return (
        comparison[
            columns
        ]
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


# =============================================================================
# COMPLETE EVALUATION
# =============================================================================

def run_evaluation() -> dict[str, Any]:
    """
    Execute Agent-5 historical diagnostic evaluation.
    """

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 5 - PORTFOLIO EVALUATOR"
    )

    print(
        "=" * 100
    )

    print()

    # =========================================================================
    # STEP 1 - FINAL PORTFOLIO
    # =========================================================================

    (
        portfolio,
        portfolio_source,
    ) = load_final_portfolio()

    print(
        f"Portfolio source        : "
        f"{portfolio_source}"
    )

    print(
        f"Portfolio stocks        : "
        f"{len(portfolio)}"
    )

    # =========================================================================
    # STEP 2 - RETURNS
    # =========================================================================

    returns = (
        load_returns_matrix()
    )

    (
        aligned_portfolio,
        aligned_returns,
        current_weights,
        final_weights,
    ) = align_portfolio_and_returns(
        portfolio=portfolio,
        returns=returns,
    )

    print(
        f"Return observations     : "
        f"{len(aligned_returns)}"
    )

    print(
        f"Evaluation start        : "
        f"{aligned_returns.index.min().date()}"
    )

    print(
        f"Evaluation end          : "
        f"{aligned_returns.index.max().date()}"
    )

    # =========================================================================
    # STEP 3 - EQUAL-WEIGHT BENCHMARK
    # =========================================================================

    benchmark_weights = (
        build_active_equal_weight_benchmark(
            aligned_portfolio
        )
    )

    active_count = int(
        (
            final_weights
            >
            cfg.FLOAT_TOLERANCE
        )
        .sum()
    )

    print(
        f"Final active assets     : "
        f"{active_count}"
    )

    # =========================================================================
    # STEP 4 - METRICS
    # =========================================================================

    current_metrics = (
        calculate_portfolio_metrics(
            returns=aligned_returns,
            weights=current_weights,
        )
    )

    benchmark_metrics = (
        calculate_portfolio_metrics(
            returns=aligned_returns,
            weights=benchmark_weights,
        )
    )

    final_metrics = (
        calculate_portfolio_metrics(
            returns=aligned_returns,
            weights=final_weights,
            previous_weights=current_weights,
        )
    )

    # =========================================================================
    # STEP 5 - TURNOVER
    # =========================================================================

    turnover = (
        portfolio_turnover(
            previous_weights=current_weights,
            new_weights=final_weights,
        )
    )

    # =========================================================================
    # STEP 6 - CHANGES
    # =========================================================================

    metric_changes = (
        calculate_metric_changes(
            current_metrics=current_metrics,
            final_metrics=final_metrics,
        )
    )

    # =========================================================================
    # DISPLAY
    # =========================================================================

    print()
    print(
        "=" * 100
    )

    print(
        "PORTFOLIO COMPARISON"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"{'Metric':<32}"
        f"{'Current':>20}"
        f"{'Equal Weight':>20}"
        f"{'Final Agent 5':>20}"
    )

    print(
        "-" * 92
    )

    print(
        f"{'Cumulative Return':<32}"
        f"{current_metrics['cumulative_return']:>19.6%}"
        f"{benchmark_metrics['cumulative_return']:>20.6%}"
        f"{final_metrics['cumulative_return']:>20.6%}"
    )

    print(
        f"{'Annualized Return':<32}"
        f"{current_metrics['annualized_return']:>19.6%}"
        f"{benchmark_metrics['annualized_return']:>20.6%}"
        f"{final_metrics['annualized_return']:>20.6%}"
    )

    print(
        f"{'Annualized Volatility':<32}"
        f"{current_metrics['annualized_volatility']:>19.6%}"
        f"{benchmark_metrics['annualized_volatility']:>20.6%}"
        f"{final_metrics['annualized_volatility']:>20.6%}"
    )

    print(
        f"{'Sharpe Ratio':<32}"
        f"{current_metrics['sharpe_ratio']:>20.6f}"
        f"{benchmark_metrics['sharpe_ratio']:>20.6f}"
        f"{final_metrics['sharpe_ratio']:>20.6f}"
    )

    print(
        f"{'Maximum Drawdown':<32}"
        f"{current_metrics['max_drawdown']:>19.6%}"
        f"{benchmark_metrics['max_drawdown']:>20.6%}"
        f"{final_metrics['max_drawdown']:>20.6%}"
    )

    print(
        f"{'Historical VaR 95%':<32}"
        f"{current_metrics['historical_var_95']:>19.6%}"
        f"{benchmark_metrics['historical_var_95']:>20.6%}"
        f"{final_metrics['historical_var_95']:>20.6%}"
    )

    print(
        f"{'Effective Assets':<32}"
        f"{current_metrics['effective_number_of_assets']:>20.4f}"
        f"{benchmark_metrics['effective_number_of_assets']:>20.4f}"
        f"{final_metrics['effective_number_of_assets']:>20.4f}"
    )

    print()

    print(
        f"Current -> Final turnover : "
        f"{turnover:.6%}"
    )

    # =========================================================================
    # ALLOCATION COMPARISON
    # =========================================================================

    allocation_comparison = (
        build_allocation_comparison(
            portfolio=aligned_portfolio,
            benchmark_weights=benchmark_weights,
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL ALLOCATION"
    )

    print(
        "=" * 100
    )

    print()

    allocation_display = (
        allocation_comparison[
            [
                "symbol",
                "trade_action",
                "current_allocation_percent",
                "benchmark_allocation_percent",
                "final_allocation_percent",
            ]
        ]
        .copy()
    )

    allocation_display = (
        allocation_display
        .rename(
            columns={
                "current_allocation_percent":
                    "current_%",
                "benchmark_allocation_percent":
                    "equal_%",
                "final_allocation_percent":
                    "final_%",
            }
        )
    )

    print(
        allocation_display.to_string(
            index=False
        )
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    checks = (
        validate_evaluation(
            portfolio=aligned_portfolio,
            returns=aligned_returns,
            current_weights=current_weights,
            benchmark_weights=benchmark_weights,
            final_weights=final_weights,
            current_metrics=current_metrics,
            benchmark_metrics=benchmark_metrics,
            final_metrics=final_metrics,
            turnover=turnover,
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
        "AGENT 5 EVALUATION VALIDATION"
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
            "Agent-5 evaluation validation failed:\n"
            +
            "\n".join(
                f"  - {name}"

                for name
                in failed
            )
        )

    # =========================================================================
    # SAVE ALLOCATION COMPARISON
    # =========================================================================

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    allocation_comparison.to_csv(
        EVALUATION_ALLOCATION_COMPARISON,
        index=False,
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    report = {
        "status":
            "COMPLETE",

        "agent":
            "Portfolio Rebalancing Agent",

        "evaluation_type":
            "historical_estimation_window_diagnostic",

        "rebalance_date":
            cfg.FINAL_REBALANCE_DATE,

        "portfolio_source":
            portfolio_source,

        "return_start":
            str(
                aligned_returns.index.min().date()
            ),

        "return_end":
            str(
                aligned_returns.index.max().date()
            ),

        "return_observations":
            int(
                len(
                    aligned_returns
                )
            ),

        "portfolio_stocks":
            int(
                len(
                    aligned_portfolio
                )
            ),

        "active_assets":
            int(
                active_count
            ),

        "current_portfolio":
            make_json_safe(
                current_metrics
            ),

        "equal_weight_active_asset_benchmark":
            make_json_safe(
                benchmark_metrics
            ),

        "final_agent5_portfolio":
            make_json_safe(
                final_metrics
            ),

        "current_to_final_changes":
            make_json_safe(
                metric_changes
            ),

        "current_to_final_turnover":
            float(
                turnover
            ),

        "final_weights": {
            str(
                row[
                    "symbol"
                ]
            ):
                float(
                    row[
                        "final_weight"
                    ]
                )

            for _, row
            in aligned_portfolio.iterrows()
        },

        "validation": {
            name:
                bool(
                    passed
                )

            for name, passed
            in checks.items()
        },

        "allocation_comparison":
            str(
                EVALUATION_ALLOCATION_COMPARISON
            ),

        "methodology_note":
            (
                "The final Agent-5 portfolio combines Markowitz "
                "minimum-variance optimization with Agent-4 DQN "
                "trade feedback. The evaluation compares the Agent-3 "
                "provisional portfolio, an equal-weight benchmark across "
                "final active assets, and the final Agent-5 allocation."
            ),

        "evaluation_limitation":
            (
                "All metrics use the same historical 105-day window "
                "used to estimate returns and covariance for the final "
                "2025-12-01 allocation. This is an in-sample diagnostic "
                "comparison, not a future post-rebalance backtest."
            ),
    }

    with open(
        cfg.EVALUATION_SUMMARY,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
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
        "GENERATED FILES"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Evaluation summary      : "
        f"{cfg.EVALUATION_SUMMARY}"
    )

    print(
        f"Allocation comparison   : "
        f"{EVALUATION_ALLOCATION_COMPARISON}"
    )

    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 5 EVALUATOR STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Final portfolio loading : COMPLETE"
    )

    print(
        "Historical returns      : COMPLETE"
    )

    print(
        "Current portfolio       : COMPLETE"
    )

    print(
        "Equal-weight benchmark  : COMPLETE"
    )

    print(
        "Final portfolio metrics : COMPLETE"
    )

    print(
        "Turnover calculation    : COMPLETE"
    )

    print(
        "Evaluation validation   : COMPLETE"
    )

    print()

    print(
        "EVALUATOR STATUS: COMPLETE"
    )

    return {
        "portfolio":
            aligned_portfolio,

        "returns":
            aligned_returns,

        "current_weights":
            current_weights,

        "benchmark_weights":
            benchmark_weights,

        "final_weights":
            final_weights,

        "current_metrics":
            current_metrics,

        "benchmark_metrics":
            benchmark_metrics,

        "final_metrics":
            final_metrics,

        "metric_changes":
            metric_changes,

        "turnover":
            turnover,

        "checks":
            checks,

        "report":
            report,
    }


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    run_evaluation()