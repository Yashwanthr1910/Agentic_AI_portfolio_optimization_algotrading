"""
Agent 5 - Agent 4 -> Agent 5 Integration Validator
===============================================================================

PROJECT
-------
Agentic AI Portfolio Optimization and Algorithmic Trading

PURPOSE
-------
Validate the interface between:

    Agent 4 - Trade Execution Agent
            ↓
    Agent 5 - Portfolio Rebalancing Agent

The validator confirms that:

1. Agent-4 final trade decisions are available.
2. Agent-5 final portfolio is available.
3. The same symbols reach Agent 5.
4. Agent-4 BUY / HOLD / SELL actions are preserved.
5. Agent-4 SELL stocks receive zero final portfolio weight.
6. Final Agent-5 weights are finite, non-negative and sum to one.
7. Final Agent-5 outputs and reports are complete.
8. The rebalance date is consistent with Agent-4 prediction date.


IMPORTANT
---------
Agent 5 is a paper-aligned implementation reconstruction.

Agent-4 SELL:
    -> zero final allocation

Agent-4 BUY / HOLD:
    -> remain eligible for MPT allocation

BUY/HOLD assets are not required to receive strictly positive weights,
because a constrained Markowitz optimizer may theoretically assign a
zero weight to an eligible asset.
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


# =============================================================================
# GENERIC INPUT LOADER
# =============================================================================

def read_with_fallback(
    parquet_path: Path,
    csv_path: Path,
    name: str,
) -> tuple[pd.DataFrame, str]:
    """
    Prefer parquet and fall back to CSV.
    """

    if parquet_path.exists():

        try:

            dataframe = pd.read_parquet(
                parquet_path
            )

            return (
                dataframe,
                "PARQUET",
            )

        except Exception as error:

            print(
                f"{name} parquet read failed: "
                f"{type(error).__name__}: {error}"
            )

    if csv_path.exists():

        dataframe = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        return (
            dataframe,
            "CSV",
        )

    raise FileNotFoundError(
        f"{name} not found.\n\n"
        f"Parquet:\n{parquet_path}\n\n"
        f"CSV:\n{csv_path}"
    )


# =============================================================================
# LOAD AGENT 4 OUTPUT
# =============================================================================

def load_agent4_decisions() -> tuple[
    pd.DataFrame,
    str,
]:
    """
    Load Agent-4 final DQN trade decisions.
    """

    (
        dataframe,
        source,
    ) = read_with_fallback(
        parquet_path=cfg.AGENT4_TRADE_DECISIONS_PARQUET,
        csv_path=cfg.AGENT4_TRADE_DECISIONS_CSV,
        name="Agent-4 trade decisions",
    )

    if dataframe.empty:

        raise ValueError(
            "Agent-4 trade decisions are empty."
        )

    required_columns = {
        "symbol",
        "prediction_date",
        "trade_action",
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
            "Agent-4 output is missing columns:\n"
            +
            "\n".join(
                sorted(
                    missing_columns
                )
            )
        )

    dataframe = dataframe.copy()

    dataframe[
        "symbol"
    ] = (
        dataframe[
            "symbol"
        ]
        .astype(str)
        .str.strip()
    )

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
        "prediction_date"
    ] = pd.to_datetime(
        dataframe[
            "prediction_date"
        ],
        errors="coerce",
    )

    if dataframe[
        "prediction_date"
    ].isna().any():

        raise ValueError(
            "Agent-4 contains invalid prediction dates."
        )

    if dataframe[
        "symbol"
    ].duplicated().any():

        raise ValueError(
            "Agent-4 contains duplicate symbols."
        )

    if not dataframe[
        "trade_action"
    ].isin(
        cfg.VALID_ACTIONS
    ).all():

        raise ValueError(
            "Agent-4 contains invalid trade actions."
        )

    return (
        dataframe,
        source,
    )


# =============================================================================
# LOAD FINAL AGENT 5 PORTFOLIO
# =============================================================================

def load_agent5_portfolio() -> tuple[
    pd.DataFrame,
    str,
]:
    """
    Load final Agent-5 portfolio.
    """

    (
        dataframe,
        source,
    ) = read_with_fallback(
        parquet_path=cfg.REBALANCED_PORTFOLIO_PARQUET,
        csv_path=cfg.REBALANCED_PORTFOLIO_CSV,
        name="Agent-5 rebalanced portfolio",
    )

    if dataframe.empty:

        raise ValueError(
            "Agent-5 final portfolio is empty."
        )

    required_columns = {
        "symbol",
        "trade_action",
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
            "Agent-5 final portfolio is missing columns:\n"
            +
            "\n".join(
                sorted(
                    missing_columns
                )
            )
        )

    dataframe = dataframe.copy()

    dataframe[
        "symbol"
    ] = (
        dataframe[
            "symbol"
        ]
        .astype(str)
        .str.strip()
    )

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
        "final_weight"
    ] = pd.to_numeric(
        dataframe[
            "final_weight"
        ],
        errors="coerce",
    )

    if dataframe[
        "final_weight"
    ].isna().any():

        raise ValueError(
            "Agent-5 contains invalid final weights."
        )

    if dataframe[
        "symbol"
    ].duplicated().any():

        raise ValueError(
            "Agent-5 contains duplicate symbols."
        )

    return (
        dataframe,
        source,
    )


# =============================================================================
# LOAD COMPLETION REPORT
# =============================================================================

def load_report(
    path: Path,
    name: str,
) -> dict[str, Any]:
    """
    Load one Agent-5 JSON report.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{name} was not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(
            file
        )

    return report


# =============================================================================
# VALIDATE REPORT STATUS
# =============================================================================

def report_is_complete(
    report: dict[str, Any],
) -> bool:
    """
    Accept COMPLETE or PASS as valid successful report states.
    """

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

    return status in {
        "COMPLETE",
        "PASS",
    }


# =============================================================================
# COMPARE AGENT 4 AND AGENT 5
# =============================================================================

def build_integration_comparison(
    agent4: pd.DataFrame,
    agent5: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one stock-level integration comparison table.
    """

    agent4_columns = [
        column
        for column in [
            "symbol",
            "prediction_date",
            "trade_action",
            "action_id",
            "q_hold",
            "q_buy",
            "q_sell",
            "selected_q_value",
        ]
        if column in agent4.columns
    ]

    agent4_view = (
        agent4[
            agent4_columns
        ]
        .copy()
    )

    rename_map = {
        column:
            f"agent4_{column}"

        for column
        in agent4_view.columns

        if column != "symbol"
    }

    agent4_view = agent4_view.rename(
        columns=rename_map
    )

    agent5_columns = [
        column
        for column in [
            "symbol",
            "trade_action",
            "current_weight",
            "mpt_weight",
            "rl_adjusted_weight",
            "target_weight",
            "final_weight",
            "portfolio_status",
            "final_decision",
            "rebalance_direction",
        ]
        if column in agent5.columns
    ]

    agent5_view = (
        agent5[
            agent5_columns
        ]
        .copy()
    )

    rename_map = {
        column:
            f"agent5_{column}"

        for column
        in agent5_view.columns

        if column != "symbol"
    }

    agent5_view = agent5_view.rename(
        columns=rename_map
    )

    comparison = agent4_view.merge(
        agent5_view,
        on="symbol",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    return comparison


# =============================================================================
# INTEGRATION VALIDATION
# =============================================================================

def validate_integration(
    agent4: pd.DataFrame,
    agent5: pd.DataFrame,
    comparison: pd.DataFrame,
    agent5_summary: dict[str, Any],
    evaluation_summary: dict[str, Any],
) -> dict[str, bool]:
    """
    Perform complete Agent-4 -> Agent-5 validation.
    """

    agent4_symbols = set(
        agent4[
            "symbol"
        ]
    )

    agent5_symbols = set(
        agent5[
            "symbol"
        ]
    )

    agent5_weights = np.array(
        agent5[
            "final_weight"
        ],
        dtype=float,
        copy=True,
    )

    sell_mask = (
        agent5[
            "trade_action"
        ]
        ==
        cfg.ACTION_SELL
    )

    buy_hold_mask = (
        agent5[
            "trade_action"
        ]
        .isin(
            cfg.MPT_ELIGIBLE_ACTIONS
        )
    )

    unique_agent4_dates = (
        agent4[
            "prediction_date"
        ]
        .dt.normalize()
        .unique()
    )

    expected_rebalance_date = pd.Timestamp(
        cfg.FINAL_REBALANCE_DATE
    ).normalize()

    checks: dict[str, bool] = {}

    # =========================================================================
    # STRUCTURE
    # =========================================================================

    checks[
        "agent4_not_empty"
    ] = (
        not agent4.empty
    )

    checks[
        "agent5_not_empty"
    ] = (
        not agent5.empty
    )

    checks[
        "agent4_unique_symbols"
    ] = (
        not agent4[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    checks[
        "agent5_unique_symbols"
    ] = (
        not agent5[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    # =========================================================================
    # SYMBOL TRANSFER
    # =========================================================================

    checks[
        "same_number_of_symbols"
    ] = (
        len(
            agent4
        )
        ==
        len(
            agent5
        )
    )

    checks[
        "same_symbol_universe"
    ] = (
        agent4_symbols
        ==
        agent5_symbols
    )

    checks[
        "all_symbols_merged"
    ] = bool(
        (
            comparison[
                "_merge"
            ]
            ==
            "both"
        )
        .all()
    )

    # =========================================================================
    # DATE CONSISTENCY
    # =========================================================================

    checks[
        "agent4_single_prediction_date"
    ] = (
        len(
            unique_agent4_dates
        )
        ==
        1
    )

    if len(
        unique_agent4_dates
    ) == 1:

        checks[
            "prediction_date_matches_rebalance_date"
        ] = (
            pd.Timestamp(
                unique_agent4_dates[
                    0
                ]
            )
            ==
            expected_rebalance_date
        )

    else:

        checks[
            "prediction_date_matches_rebalance_date"
        ] = False

    # =========================================================================
    # ACTION TRANSFER
    # =========================================================================

    checks[
        "agent4_actions_valid"
    ] = bool(
        agent4[
            "trade_action"
        ]
        .isin(
            cfg.VALID_ACTIONS
        )
        .all()
    )

    checks[
        "agent5_actions_valid"
    ] = bool(
        agent5[
            "trade_action"
        ]
        .isin(
            cfg.VALID_ACTIONS
        )
        .all()
    )

    if (
        "agent4_trade_action"
        in comparison.columns

        and

        "agent5_trade_action"
        in comparison.columns
    ):

        checks[
            "trade_actions_preserved"
        ] = bool(
            (
                comparison[
                    "agent4_trade_action"
                ]
                ==
                comparison[
                    "agent5_trade_action"
                ]
            )
            .all()
        )

    else:

        checks[
            "trade_actions_preserved"
        ] = False

    # =========================================================================
    # FINAL WEIGHTS
    # =========================================================================

    checks[
        "final_weights_finite"
    ] = bool(
        np.isfinite(
            agent5_weights
        )
        .all()
    )

    checks[
        "final_weights_nonnegative"
    ] = bool(
        (
            agent5_weights
            >=
            -cfg.FLOAT_TOLERANCE
        )
        .all()
    )

    checks[
        "final_weights_below_max"
    ] = bool(
        (
            agent5_weights
            <=
            (
                cfg.MAX_WEIGHT
                +
                cfg.FLOAT_TOLERANCE
            )
        )
        .all()
    )

    checks[
        "final_weights_sum_to_one"
    ] = bool(
        np.isclose(
            agent5_weights.sum(),
            cfg.WEIGHT_SUM_TARGET,
            atol=cfg.WEIGHT_SUM_TOLERANCE,
        )
    )

    # =========================================================================
    # DQN SELL -> ZERO
    # =========================================================================

    if sell_mask.any():

        checks[
            "agent4_sell_weights_zero"
        ] = bool(
            np.allclose(
                agent5.loc[
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
            "agent4_sell_weights_zero"
        ] = True

    # =========================================================================
    # SELL STATUS
    # =========================================================================

    if (
        sell_mask.any()

        and

        "portfolio_status"
        in agent5.columns
    ):

        checks[
            "agent4_sell_positions_exited"
        ] = bool(
            (
                agent5.loc[
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
            "agent4_sell_positions_exited"
        ] = True

    # =========================================================================
    # BUY / HOLD ELIGIBILITY
    # =========================================================================
    #
    # We only require that BUY/HOLD remain non-negative valid portfolio
    # candidates. MPT may theoretically assign zero to an eligible asset.
    # =========================================================================

    checks[
        "buy_hold_assets_retained"
    ] = bool(
        (
            agent5.loc[
                buy_hold_mask,
                "final_weight",
            ]
            >=
            -cfg.FLOAT_TOLERANCE
        )
        .all()
    )

    # =========================================================================
    # ACTION COUNTS
    # =========================================================================

    agent4_action_counts = (
        agent4[
            "trade_action"
        ]
        .value_counts()
        .to_dict()
    )

    agent5_action_counts = (
        agent5[
            "trade_action"
        ]
        .value_counts()
        .to_dict()
    )

    checks[
        "buy_action_count_preserved"
    ] = (
        int(
            agent4_action_counts.get(
                cfg.ACTION_BUY,
                0,
            )
        )
        ==
        int(
            agent5_action_counts.get(
                cfg.ACTION_BUY,
                0,
            )
        )
    )

    checks[
        "hold_action_count_preserved"
    ] = (
        int(
            agent4_action_counts.get(
                cfg.ACTION_HOLD,
                0,
            )
        )
        ==
        int(
            agent5_action_counts.get(
                cfg.ACTION_HOLD,
                0,
            )
        )
    )

    checks[
        "sell_action_count_preserved"
    ] = (
        int(
            agent4_action_counts.get(
                cfg.ACTION_SELL,
                0,
            )
        )
        ==
        int(
            agent5_action_counts.get(
                cfg.ACTION_SELL,
                0,
            )
        )
    )

    # =========================================================================
    # REPORT COMPLETION
    # =========================================================================

    checks[
        "agent5_summary_complete"
    ] = (
        report_is_complete(
            agent5_summary
        )
    )

    checks[
        "evaluation_summary_complete"
    ] = (
        report_is_complete(
            evaluation_summary
        )
    )

    return checks


# =============================================================================
# COMPLETE INTEGRATION VALIDATOR
# =============================================================================

def run_integration_validation() -> dict[str, Any]:
    """
    Execute final Agent-4 -> Agent-5 integration validation.
    """

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 4 -> AGENT 5 INTEGRATION VALIDATOR"
    )

    print(
        "=" * 100
    )

    print()

    # =========================================================================
    # LOAD AGENT 4
    # =========================================================================

    (
        agent4,
        agent4_source,
    ) = load_agent4_decisions()

    print(
        f"Agent-4 source          : "
        f"{agent4_source}"
    )

    print(
        f"Agent-4 stocks          : "
        f"{len(agent4)}"
    )

    # =========================================================================
    # LOAD AGENT 5
    # =========================================================================

    (
        agent5,
        agent5_source,
    ) = load_agent5_portfolio()

    print(
        f"Agent-5 source          : "
        f"{agent5_source}"
    )

    print(
        f"Agent-5 stocks          : "
        f"{len(agent5)}"
    )

    # =========================================================================
    # LOAD REPORTS
    # =========================================================================

    agent5_summary = (
        load_report(
            path=cfg.AGENT5_SUMMARY,
            name="Agent-5 summary",
        )
    )

    evaluation_summary = (
        load_report(
            path=cfg.EVALUATION_SUMMARY,
            name="Agent-5 evaluation summary",
        )
    )

    # =========================================================================
    # ACTION DISTRIBUTION
    # =========================================================================

    agent4_counts = (
        agent4[
            "trade_action"
        ]
        .value_counts()
        .to_dict()
    )

    agent5_counts = (
        agent5[
            "trade_action"
        ]
        .value_counts()
        .to_dict()
    )

    print()

    print(
        "AGENT-4 ACTIONS"
    )

    print(
        "-" * 100
    )

    print(
        f"BUY                     : "
        f"{agent4_counts.get('BUY', 0)}"
    )

    print(
        f"HOLD                    : "
        f"{agent4_counts.get('HOLD', 0)}"
    )

    print(
        f"SELL                    : "
        f"{agent4_counts.get('SELL', 0)}"
    )

    print()

    print(
        "AGENT-5 PRESERVED ACTIONS"
    )

    print(
        "-" * 100
    )

    print(
        f"BUY                     : "
        f"{agent5_counts.get('BUY', 0)}"
    )

    print(
        f"HOLD                    : "
        f"{agent5_counts.get('HOLD', 0)}"
    )

    print(
        f"SELL                    : "
        f"{agent5_counts.get('SELL', 0)}"
    )

    # =========================================================================
    # COMPARISON
    # =========================================================================

    comparison = (
        build_integration_comparison(
            agent4=agent4,
            agent5=agent5,
        )
    )

    # =========================================================================
    # VALIDATE
    # =========================================================================

    checks = (
        validate_integration(
            agent4=agent4,
            agent5=agent5,
            comparison=comparison,
            agent5_summary=agent5_summary,
            evaluation_summary=evaluation_summary,
        )
    )

    overall_pass = all(
        checks.values()
    )

    # =========================================================================
    # DISPLAY FINAL ALLOCATION
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "AGENT 4 -> AGENT 5 FINAL TRANSFER"
    )

    print(
        "=" * 100
    )

    print()

    display_columns = [
        column
        for column in [
            "symbol",
            "trade_action",
            "final_weight",
            "portfolio_status",
        ]
        if column in agent5.columns
    ]

    display = (
        agent5[
            display_columns
        ]
        .copy()
    )

    if "final_weight" in display.columns:

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
        display
        .sort_values(
            "final_weight_%",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # VALIDATION DISPLAY
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "AGENT 4 -> AGENT 5 VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    for name, passed in checks.items():

        print(
            f"{name:<65}: "
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
            "Agent-4 -> Agent-5 integration validation failed:\n"
            +
            "\n".join(
                f"  - {name}"

                for name
                in failed
            )
        )

    # =========================================================================
    # REPORT
    # =========================================================================

    sell_symbols = (
        agent5.loc[
            agent5[
                "trade_action"
            ]
            ==
            cfg.ACTION_SELL,
            "symbol",
        ]
        .tolist()
    )

    active_symbols = (
        agent5.loc[
            agent5[
                "final_weight"
            ]
            >
            cfg.FLOAT_TOLERANCE,
            "symbol",
        ]
        .tolist()
    )

    report = {
        "status":
            "PASS",

        "integration":
            "Agent 4 -> Agent 5",

        "rebalance_date":
            cfg.FINAL_REBALANCE_DATE,

        "agent4_source":
            agent4_source,

        "agent5_source":
            agent5_source,

        "agent4_stocks":
            int(
                len(
                    agent4
                )
            ),

        "agent5_stocks":
            int(
                len(
                    agent5
                )
            ),

        "agent4_actions": {
            "BUY":
                int(
                    agent4_counts.get(
                        "BUY",
                        0,
                    )
                ),

            "HOLD":
                int(
                    agent4_counts.get(
                        "HOLD",
                        0,
                    )
                ),

            "SELL":
                int(
                    agent4_counts.get(
                        "SELL",
                        0,
                    )
                ),
        },

        "agent5_preserved_actions": {
            "BUY":
                int(
                    agent5_counts.get(
                        "BUY",
                        0,
                    )
                ),

            "HOLD":
                int(
                    agent5_counts.get(
                        "HOLD",
                        0,
                    )
                ),

            "SELL":
                int(
                    agent5_counts.get(
                        "SELL",
                        0,
                    )
                ),
        },

        "active_symbols":
            active_symbols,

        "sell_exit_symbols":
            sell_symbols,

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
            in agent5.iterrows()
        },

        "validation": {
            name:
                bool(
                    passed
                )

            for name, passed
            in checks.items()
        },

        "methodology_note":
            (
                "The Agent-4 DQN BUY/HOLD/SELL decisions are transferred "
                "into Agent 5 without modification. SELL stocks are forced "
                "to zero final allocation. BUY and HOLD stocks remain "
                "eligible for Markowitz portfolio optimization."
            ),

        "system_limitation":
            (
                "The historical return evaluation in Agent 5 uses the "
                "same 105-day estimation interval used to construct the "
                "2025-12-01 allocation. It is diagnostic rather than a "
                "future post-rebalance backtest."
            ),
    }

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        cfg.AGENT4_AGENT5_VALIDATION,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    # =========================================================================
    # GENERATED FILE
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "GENERATED FILE"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Integration report      : "
        f"{cfg.AGENT4_AGENT5_VALIDATION}"
    )

    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "AGENT 4 -> AGENT 5 INTEGRATION STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Symbol transfer         : COMPLETE"
    )

    print(
        "Trade-action transfer   : COMPLETE"
    )

    print(
        "Prediction date         : COMPLETE"
    )

    print(
        "DQN SELL enforcement    : COMPLETE"
    )

    print(
        "Portfolio weights       : COMPLETE"
    )

    print(
        "Agent-5 reports         : COMPLETE"
    )

    print(
        "Integration validation  : COMPLETE"
    )

    print()

    print(
        "AGENT 4 -> AGENT 5 VALIDATION: PASS"
    )

    return {
        "agent4":
            agent4,

        "agent5":
            agent5,

        "comparison":
            comparison,

        "checks":
            checks,

        "report":
            report,
    }


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    run_integration_validation()