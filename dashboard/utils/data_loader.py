"""
dashboard/utils/data_loader.py

Central data-loading layer for the Streamlit dashboard.

The dashboard should NOT directly read Agent-1, Agent-2, Agent-3,
Agent-4, and Agent-5 files from every page.

Instead:

    Dashboard Page
         ↓
    data_loader.py
         ↓
    Validated project outputs

This provides one consistent source of truth for:

    - System status
    - Full five-agent stock journey
    - Agent-1 stock selection
    - Agent-2 trend prediction
    - Agent-3 risk assessment
    - Agent-4 DQN trade decisions
    - Agent-5 final portfolio
    - Portfolio evaluation
    - Runtime / stage timings
    - Integration validation

The loader is intentionally independent of Streamlit so it can also
be tested from the command line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# =============================================================================
# PROJECT ROOT
# =============================================================================

# Current file:
#
# project_root/
# └── dashboard/
#     └── utils/
#         └── data_loader.py
#
# parents[2] = project root

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# =============================================================================
# AGENT ROOTS
# =============================================================================

AGENT1_ROOT = (
    PROJECT_ROOT
    / "stock_selection_agent"
)

AGENT2_ROOT = (
    PROJECT_ROOT
    / "trend_prediction_agent"
)

AGENT3_ROOT = (
    PROJECT_ROOT
    / "risk_management_agent"
)

AGENT4_ROOT = (
    PROJECT_ROOT
    / "trade_execution_agent"
)

AGENT5_ROOT = (
    PROJECT_ROOT
    / "portfolio_rebalancing_agent"
)


# =============================================================================
# SYSTEM REPORT ROOT
# =============================================================================

SYSTEM_REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "system"
)

SYSTEM_LATEST_DIR = (
    SYSTEM_REPORT_ROOT
    / "latest"
)

SYSTEM_RUNS_DIR = (
    SYSTEM_REPORT_ROOT
    / "runs"
)


# =============================================================================
# SYSTEM FILES
# =============================================================================

FULL_TRACE_CSV = (
    SYSTEM_LATEST_DIR
    / "full_agent_trace.csv"
)

FULL_TRACE_PARQUET = (
    SYSTEM_LATEST_DIR
    / "full_agent_trace.parquet"
)

SYSTEM_SUMMARY_JSON = (
    SYSTEM_LATEST_DIR
    / "system_summary.json"
)

INTEGRATION_SUMMARY_JSON = (
    SYSTEM_LATEST_DIR
    / "integration_summary.json"
)

PREFLIGHT_SUMMARY_JSON = (
    SYSTEM_LATEST_DIR
    / "preflight_summary.json"
)

OUTPUT_COLLECTOR_SUMMARY_JSON = (
    SYSTEM_LATEST_DIR
    / "output_collector_summary.json"
)

STAGE_TIMINGS_CSV = (
    SYSTEM_LATEST_DIR
    / "stage_timings.csv"
)


# =============================================================================
# AGENT 1 FILES
# =============================================================================

AGENT1_SELECTED_CSV = (
    AGENT1_ROOT
    / "data"
    / "outputs"
    / "selected_stocks.csv"
)

AGENT1_SELECTED_PARQUET = (
    AGENT1_ROOT
    / "data"
    / "outputs"
    / "selected_stocks.parquet"
)

AGENT1_RANKINGS_CSV = (
    AGENT1_ROOT
    / "outputs"
    / "rankings.csv"
)

AGENT1_RANKINGS_PARQUET = (
    AGENT1_ROOT
    / "outputs"
    / "rankings.parquet"
)

AGENT1_DT_METRICS_CSV = (
    AGENT1_ROOT
    / "reports"
    / "results"
    / "decision_tree_metrics.csv"
)

AGENT1_FEATURE_IMPORTANCE_CSV = (
    AGENT1_ROOT
    / "reports"
    / "results"
    / "decision_tree_feature_importance.csv"
)

AGENT1_BGSTO_HISTORY_CSV = (
    AGENT1_ROOT
    / "reports"
    / "results"
    / "bgsto_optimization_history.csv"
)


# =============================================================================
# AGENT 2 FILES
# =============================================================================

AGENT2_PREDICTIONS_CSV = (
    AGENT2_ROOT
    / "outputs"
    / "trend_predictions.csv"
)

AGENT2_PREDICTIONS_PARQUET = (
    AGENT2_ROOT
    / "outputs"
    / "trend_predictions.parquet"
)


# =============================================================================
# AGENT 3 FILES
# =============================================================================

AGENT3_RISK_CSV = (
    AGENT3_ROOT
    / "outputs"
    / "risk_assessment.csv"
)

AGENT3_RISK_PARQUET = (
    AGENT3_ROOT
    / "outputs"
    / "risk_assessment.parquet"
)

AGENT3_SUMMARY_JSON = (
    AGENT3_ROOT
    / "reports"
    / "risk_summary.json"
)


# =============================================================================
# AGENT 4 FILES
# =============================================================================

AGENT4_TRADES_CSV = (
    AGENT4_ROOT
    / "outputs"
    / "trade_decisions.csv"
)

AGENT4_TRADES_PARQUET = (
    AGENT4_ROOT
    / "outputs"
    / "trade_decisions.parquet"
)

AGENT4_INTEGRATION_JSON = (
    AGENT4_ROOT
    / "reports"
    / "agent3_agent4_validation.json"
)


# =============================================================================
# AGENT 5 FILES
# =============================================================================

AGENT5_PORTFOLIO_CSV = (
    AGENT5_ROOT
    / "outputs"
    / "rebalanced_portfolio.csv"
)

AGENT5_PORTFOLIO_PARQUET = (
    AGENT5_ROOT
    / "outputs"
    / "rebalanced_portfolio.parquet"
)

AGENT5_FINAL_ALLOCATION_CSV = (
    AGENT5_ROOT
    / "outputs"
    / "final_portfolio_allocation.csv"
)

AGENT5_SUMMARY_JSON = (
    AGENT5_ROOT
    / "reports"
    / "agent5_summary.json"
)

AGENT5_EVALUATION_JSON = (
    AGENT5_ROOT
    / "reports"
    / "evaluation_summary.json"
)

AGENT5_ALLOCATION_COMPARISON_CSV = (
    AGENT5_ROOT
    / "reports"
    / "evaluation_allocation_comparison.csv"
)

AGENT5_REBALANCE_ENV_JSON = (
    AGENT5_ROOT
    / "reports"
    / "rebalancing_environment_summary.json"
)

AGENT5_INTEGRATION_JSON = (
    AGENT5_ROOT
    / "reports"
    / "agent4_agent5_validation.json"
)

AGENT5_MPT_SUMMARY_JSON = (
    AGENT5_ROOT
    / "reports"
    / "mpt_optimization_summary.json"
)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def path_ready(
    path: Path,
) -> bool:
    """
    Return True when a file exists and is non-empty.
    """

    try:

        return (
            path.exists()
            and
            path.is_file()
            and
            path.stat().st_size > 0
        )

    except OSError:

        return False


def normalize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize column names.
    """

    result = dataframe.copy()

    result.columns = [
        str(column)
        .strip()
        .lower()
        for column
        in result.columns
    ]

    return result


def normalize_symbols(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize stock-symbol representation.
    """

    result = dataframe.copy()

    if "symbol" in result.columns:

        result[
            "symbol"
        ] = (
            result[
                "symbol"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    return result


# =============================================================================
# GENERIC TABLE LOADER
# =============================================================================

def load_table(
    csv_path: Optional[Path] = None,
    parquet_path: Optional[Path] = None,
    required: bool = True,
) -> pd.DataFrame:
    """
    Load a dataframe.

    Preference:
        Parquet
        ↓
        CSV

    Parameters
    ----------
    required:
        If True, raise FileNotFoundError if neither file exists.

        If False, return an empty dataframe.
    """

    dataframe: Optional[
        pd.DataFrame
    ] = None

    if (
        parquet_path is not None
        and
        path_ready(
            parquet_path
        )
    ):

        dataframe = pd.read_parquet(
            parquet_path
        )

    elif (
        csv_path is not None
        and
        path_ready(
            csv_path
        )
    ):

        dataframe = pd.read_csv(
            csv_path,
            low_memory=False,
        )

    if dataframe is None:

        if required:

            raise FileNotFoundError(
                "Unable to locate required table.\n"
                f"CSV     : {csv_path}\n"
                f"Parquet : {parquet_path}"
            )

        return pd.DataFrame()

    dataframe = normalize_columns(
        dataframe
    )

    dataframe = normalize_symbols(
        dataframe
    )

    return dataframe


# =============================================================================
# GENERIC JSON LOADER
# =============================================================================

def load_json(
    path: Path,
    required: bool = True,
) -> Dict[str, Any]:
    """
    Load JSON report.
    """

    if not path_ready(
        path
    ):

        if required:

            raise FileNotFoundError(
                f"Required JSON file not found:\n{path}"
            )

        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            f"JSON root must be an object/dictionary:\n{path}"
        )

    return data


# =============================================================================
# SYSTEM LOADERS
# =============================================================================

def load_full_trace() -> pd.DataFrame:
    """
    Load unified Agent-1 -> Agent-5 stock trace.
    """

    return load_table(
        csv_path=FULL_TRACE_CSV,
        parquet_path=FULL_TRACE_PARQUET,
    )


def load_system_summary() -> Dict[str, Any]:
    """
    Load latest root pipeline summary.
    """

    return load_json(
        SYSTEM_SUMMARY_JSON
    )


def load_integration_summary() -> Dict[str, Any]:
    """
    Load complete Agent1 -> Agent5 validation report.
    """

    return load_json(
        INTEGRATION_SUMMARY_JSON
    )


def load_preflight_summary() -> Dict[str, Any]:

    return load_json(
        PREFLIGHT_SUMMARY_JSON
    )


def load_output_collector_summary() -> Dict[str, Any]:

    return load_json(
        OUTPUT_COLLECTOR_SUMMARY_JSON
    )


def load_stage_timings() -> pd.DataFrame:
    """
    Load pipeline execution durations.
    """

    if not path_ready(
        STAGE_TIMINGS_CSV
    ):

        return pd.DataFrame()

    dataframe = pd.read_csv(
        STAGE_TIMINGS_CSV
    )

    dataframe = normalize_columns(
        dataframe
    )

    if (
        "duration_seconds"
        in dataframe.columns
    ):

        dataframe[
            "duration_seconds"
        ] = pd.to_numeric(
            dataframe[
                "duration_seconds"
            ],
            errors="coerce",
        )

    return dataframe


# =============================================================================
# AGENT 1
# =============================================================================

def load_agent1_selected() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT1_SELECTED_CSV,
        parquet_path=AGENT1_SELECTED_PARQUET,
    )


def load_agent1_rankings() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT1_RANKINGS_CSV,
        parquet_path=AGENT1_RANKINGS_PARQUET,
    )


def load_agent1_dt_metrics() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT1_DT_METRICS_CSV,
        parquet_path=None,
        required=False,
    )


def load_agent1_feature_importance() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT1_FEATURE_IMPORTANCE_CSV,
        parquet_path=None,
        required=False,
    )


def load_agent1_bgsto_history() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT1_BGSTO_HISTORY_CSV,
        parquet_path=None,
        required=False,
    )


# =============================================================================
# AGENT 2
# =============================================================================

def load_agent2_predictions() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT2_PREDICTIONS_CSV,
        parquet_path=AGENT2_PREDICTIONS_PARQUET,
    )


# =============================================================================
# AGENT 3
# =============================================================================

def load_agent3_risk() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT3_RISK_CSV,
        parquet_path=AGENT3_RISK_PARQUET,
    )


def load_agent3_summary() -> Dict[str, Any]:

    return load_json(
        AGENT3_SUMMARY_JSON,
        required=False,
    )


# =============================================================================
# AGENT 4
# =============================================================================

def load_agent4_trades() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT4_TRADES_CSV,
        parquet_path=AGENT4_TRADES_PARQUET,
    )


def load_agent4_integration() -> Dict[str, Any]:

    return load_json(
        AGENT4_INTEGRATION_JSON,
        required=False,
    )


# =============================================================================
# AGENT 5
# =============================================================================

def load_agent5_portfolio() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT5_PORTFOLIO_CSV,
        parquet_path=AGENT5_PORTFOLIO_PARQUET,
    )


def load_agent5_final_allocation() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT5_FINAL_ALLOCATION_CSV,
        parquet_path=None,
    )


def load_agent5_summary() -> Dict[str, Any]:

    return load_json(
        AGENT5_SUMMARY_JSON,
        required=False,
    )


def load_agent5_evaluation() -> Dict[str, Any]:

    return load_json(
        AGENT5_EVALUATION_JSON,
        required=False,
    )


def load_agent5_allocation_comparison() -> pd.DataFrame:

    return load_table(
        csv_path=AGENT5_ALLOCATION_COMPARISON_CSV,
        parquet_path=None,
        required=False,
    )


def load_agent5_rebalancing_environment() -> Dict[str, Any]:

    return load_json(
        AGENT5_REBALANCE_ENV_JSON,
        required=False,
    )


def load_agent5_integration() -> Dict[str, Any]:

    return load_json(
        AGENT5_INTEGRATION_JSON,
        required=False,
    )


def load_agent5_mpt_summary() -> Dict[str, Any]:

    return load_json(
        AGENT5_MPT_SUMMARY_JSON,
        required=False,
    )


# =============================================================================
# FINAL PORTFOLIO HELPERS
# =============================================================================

def get_active_portfolio() -> pd.DataFrame:
    """
    Return only stocks with positive final allocation.
    """

    portfolio = (
        load_agent5_portfolio()
        .copy()
    )

    if (
        "final_weight"
        not in portfolio.columns
    ):

        raise ValueError(
            "Agent-5 portfolio does not contain final_weight."
        )

    portfolio[
        "final_weight"
    ] = pd.to_numeric(
        portfolio[
            "final_weight"
        ],
        errors="coerce",
    )

    active = (
        portfolio[
            portfolio[
                "final_weight"
            ]
            >
            1e-8
        ]
        .copy()
    )

    active = active.sort_values(
        "final_weight",
        ascending=False,
        kind="stable",
    )

    return active.reset_index(
        drop=True
    )


def get_exited_portfolio() -> pd.DataFrame:
    """
    Return stocks with zero final weight.
    """

    portfolio = (
        load_agent5_portfolio()
        .copy()
    )

    portfolio[
        "final_weight"
    ] = pd.to_numeric(
        portfolio[
            "final_weight"
        ],
        errors="coerce",
    )

    exited = (
        portfolio[
            portfolio[
                "final_weight"
            ]
            <=
            1e-8
        ]
        .copy()
    )

    return exited.reset_index(
        drop=True
    )


# =============================================================================
# ACTION COUNTS
# =============================================================================

def get_agent4_action_counts() -> Dict[str, int]:
    """
    BUY / HOLD / SELL distribution.
    """

    trades = (
        load_agent4_trades()
    )

    if (
        "trade_action"
        not in trades.columns
    ):

        return {
            "BUY": 0,
            "HOLD": 0,
            "SELL": 0,
        }

    actions = (
        trades[
            "trade_action"
        ]
        .astype(str)
        .str.upper()
    )

    return {
        "BUY":
            int(
                (
                    actions
                    ==
                    "BUY"
                ).sum()
            ),

        "HOLD":
            int(
                (
                    actions
                    ==
                    "HOLD"
                ).sum()
            ),

        "SELL":
            int(
                (
                    actions
                    ==
                    "SELL"
                ).sum()
            ),
    }


# =============================================================================
# RISK COUNTS
# =============================================================================

def get_agent3_risk_counts() -> Dict[str, int]:
    """
    LOW / MEDIUM / HIGH risk distribution.
    """

    risk = (
        load_agent3_risk()
    )

    if (
        "risk_level"
        not in risk.columns
    ):

        return {
            "LOW": 0,
            "MEDIUM": 0,
            "HIGH": 0,
        }

    values = (
        risk[
            "risk_level"
        ]
        .astype(str)
        .str.upper()
    )

    return {
        "LOW":
            int(
                (
                    values
                    ==
                    "LOW"
                ).sum()
            ),

        "MEDIUM":
            int(
                (
                    values
                    ==
                    "MEDIUM"
                ).sum()
            ),

        "HIGH":
            int(
                (
                    values
                    ==
                    "HIGH"
                ).sum()
            ),
    }


# =============================================================================
# TREND COUNTS
# =============================================================================

def get_agent2_trend_counts() -> Dict[str, int]:
    """
    TOP30 / NOT_TOP30 distribution.
    """

    predictions = (
        load_agent2_predictions()
    )

    if (
        "trend_class"
        not in predictions.columns
    ):

        return {
            "TOP30": 0,
            "NOT_TOP30": 0,
        }

    values = (
        predictions[
            "trend_class"
        ]
        .astype(str)
        .str.upper()
    )

    return {
        "TOP30":
            int(
                (
                    values
                    ==
                    "TOP30"
                ).sum()
            ),

        "NOT_TOP30":
            int(
                (
                    values
                    ==
                    "NOT_TOP30"
                ).sum()
            ),
    }


# =============================================================================
# PIPELINE STATUS
# =============================================================================

def get_system_status() -> Dict[str, Any]:
    """
    Return concise status information for dashboard homepage.
    """

    system = (
        load_system_summary()
    )

    integration = (
        load_integration_summary()
    )

    full_trace = (
        load_full_trace()
    )

    final_portfolio = (
        load_agent5_portfolio()
    )

    active = (
        get_active_portfolio()
    )

    exited = (
        get_exited_portfolio()
    )

    action_counts = (
        get_agent4_action_counts()
    )

    return {
        "system_status":
            system.get(
                "status",
                "UNKNOWN",
            ),

        "overall_pass":
            bool(
                system.get(
                    "overall_pass",
                    False,
                )
            ),

        "run_id":
            system.get(
                "run_id"
            ),

        "mode":
            system.get(
                "mode"
            ),

        "inference_date":
            integration.get(
                "inference_date",
            ),

        "total_stocks":
            len(
                full_trace
            ),

        "active_assets":
            len(
                active
            ),

        "exited_assets":
            len(
                exited
            ),

        "buy_count":
            action_counts[
                "BUY"
            ],

        "hold_count":
            action_counts[
                "HOLD"
            ],

        "sell_count":
            action_counts[
                "SELL"
            ],

        "final_weight_sum":
            (
                float(
                    pd.to_numeric(
                        final_portfolio[
                            "final_weight"
                        ],
                        errors="coerce",
                    )
                    .sum()
                )
                if "final_weight"
                in final_portfolio.columns
                else None
            ),

        "integration_status":
            integration.get(
                "status",
                "UNKNOWN",
            ),

        "integration_groups":
            integration.get(
                "group_status",
                {},
            ),
    }


# =============================================================================
# AVAILABLE RUNS
# =============================================================================

def list_saved_runs() -> pd.DataFrame:
    """
    List historical full-system run snapshots.

    This will later support experiment selection in Streamlit.
    """

    if not SYSTEM_RUNS_DIR.exists():

        return pd.DataFrame(
            columns=[
                "run_id",
                "path",
                "modified_time",
            ]
        )

    rows = []

    for directory in (
        SYSTEM_RUNS_DIR.iterdir()
    ):

        if not directory.is_dir():

            continue

        rows.append(
            {
                "run_id":
                    directory.name,

                "path":
                    str(
                        directory
                    ),

                "modified_time":
                    pd.Timestamp(
                        directory.stat().st_mtime,
                        unit="s",
                    ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:

        return dataframe

    dataframe = dataframe.sort_values(
        "modified_time",
        ascending=False,
    )

    return dataframe.reset_index(
        drop=True
    )


# =============================================================================
# DASHBOARD SOURCE VALIDATION
# =============================================================================

def validate_dashboard_sources() -> Dict[str, bool]:
    """
    Validate the minimum files needed by the dashboard.
    """

    checks = {
        "full_agent_trace":
            (
                path_ready(
                    FULL_TRACE_PARQUET
                )
                or
                path_ready(
                    FULL_TRACE_CSV
                )
            ),

        "system_summary":
            path_ready(
                SYSTEM_SUMMARY_JSON
            ),

        "integration_summary":
            path_ready(
                INTEGRATION_SUMMARY_JSON
            ),

        "agent1_output":
            (
                path_ready(
                    AGENT1_SELECTED_PARQUET
                )
                or
                path_ready(
                    AGENT1_SELECTED_CSV
                )
            ),

        "agent2_output":
            (
                path_ready(
                    AGENT2_PREDICTIONS_PARQUET
                )
                or
                path_ready(
                    AGENT2_PREDICTIONS_CSV
                )
            ),

        "agent3_output":
            (
                path_ready(
                    AGENT3_RISK_PARQUET
                )
                or
                path_ready(
                    AGENT3_RISK_CSV
                )
            ),

        "agent4_output":
            (
                path_ready(
                    AGENT4_TRADES_PARQUET
                )
                or
                path_ready(
                    AGENT4_TRADES_CSV
                )
            ),

        "agent5_output":
            (
                path_ready(
                    AGENT5_PORTFOLIO_PARQUET
                )
                or
                path_ready(
                    AGENT5_PORTFOLIO_CSV
                )
            ),
    }

    return checks


# =============================================================================
# COMPLETE DASHBOARD DATA BUNDLE
# =============================================================================

def load_dashboard_bundle() -> Dict[str, Any]:
    """
    Load the primary dashboard datasets at once.

    Useful for the main dashboard page.
    """

    return {
        "system_status":
            get_system_status(),

        "full_trace":
            load_full_trace(),

        "agent1_selected":
            load_agent1_selected(),

        "agent1_rankings":
            load_agent1_rankings(),

        "agent2_predictions":
            load_agent2_predictions(),

        "agent3_risk":
            load_agent3_risk(),

        "agent4_trades":
            load_agent4_trades(),

        "agent5_portfolio":
            load_agent5_portfolio(),

        "agent5_evaluation":
            load_agent5_evaluation(),

        "stage_timings":
            load_stage_timings(),

        "saved_runs":
            list_saved_runs(),
    }


# =============================================================================
# COMMAND-LINE SMOKE TEST
# =============================================================================

def main() -> int:
    """
    Verify dashboard data sources before building Streamlit UI.
    """

    print()
    print(
        "=" * 100
    )

    print(
        "STREAMLIT DASHBOARD DATA LOADER"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print()

    checks = (
        validate_dashboard_sources()
    )

    print(
        "DATA SOURCE CHECKS"
    )

    print(
        "-" * 100
    )

    for name, passed in (
        checks.items()
    ):

        print(
            f"{name:<40} : "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()

    if not all(
        checks.values()
    ):

        print(
            "DASHBOARD DATA STATUS: FAIL"
        )

        return 1

    # =========================================================================
    # LOAD MAIN DATA
    # =========================================================================

    bundle = (
        load_dashboard_bundle()
    )

    status = (
        bundle[
            "system_status"
        ]
    )

    print(
        "SYSTEM STATUS"
    )

    print(
        "-" * 100
    )

    print(
        f"Run ID               : "
        f"{status.get('run_id')}"
    )

    print(
        f"System status        : "
        f"{status.get('system_status')}"
    )

    print(
        f"Integration status   : "
        f"{status.get('integration_status')}"
    )

    print(
        f"Inference date       : "
        f"{status.get('inference_date')}"
    )

    print(
        f"Stocks               : "
        f"{status.get('total_stocks')}"
    )

    print(
        f"Active assets        : "
        f"{status.get('active_assets')}"
    )

    print(
        f"Exited assets        : "
        f"{status.get('exited_assets')}"
    )

    print(
        f"BUY / HOLD / SELL    : "
        f"{status.get('buy_count')} / "
        f"{status.get('hold_count')} / "
        f"{status.get('sell_count')}"
    )

    print(
        f"Final weight sum     : "
        f"{status.get('final_weight_sum')}"
    )

    print()

    print(
        "DATASET ROW COUNTS"
    )

    print(
        "-" * 100
    )

    dataframe_names = (
        "full_trace",
        "agent1_selected",
        "agent1_rankings",
        "agent2_predictions",
        "agent3_risk",
        "agent4_trades",
        "agent5_portfolio",
        "stage_timings",
        "saved_runs",
    )

    for name in dataframe_names:

        dataframe = (
            bundle[
                name
            ]
        )

        print(
            f"{name:<30} : "
            f"{len(dataframe)} rows"
        )

    print()

    print(
        "=" * 100
    )

    print(
        "DASHBOARD DATA STATUS: PASS"
    )

    print(
        "=" * 100
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )