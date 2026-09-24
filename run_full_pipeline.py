"""
run_full_pipeline.py

Root orchestration entry point for the complete five-agent
Agentic AI Portfolio Optimization and Algorithmic Trading system.

Pipeline
--------
Agent 1
    Stock Selection
    Decision Tree + BGSTO
        ↓
Agent 2
    Trend Prediction
    Dilated LSTM + Transformer + Progressive Attention
        ↓
Agent 3
    Risk Management
    Volatility + VaR + Sharpe
        ↓
Agent 4
    Trade Execution
    DQN BUY / HOLD / SELL
        ↓
Agent 5
    Portfolio Rebalancing
    Markowitz MPT + RL Feedback
        ↓
Unified Output Collector
        ↓
Full Integration Validator
        ↓
Final Portfolio

Modes
-----
verify
    Does NOT execute the agents.

    Validates the existing Agent-1 -> Agent-5 outputs,
    rebuilds the unified trace, and runs complete
    end-to-end integration validation.

run
    Executes Agent 1 -> Agent 5 sequentially.

    Each agent must complete successfully before the next
    agent is allowed to start.

    After the five agents:
        - output collector runs
        - system preflight verification runs
        - full integration validation runs
        - experiment snapshot is saved

Important
---------
This runner does NOT intentionally retrain the locked Agent-2
neural ensemble or Agent-4 DQN unless their own production
main.py explicitly does so.

The runner uses each existing agent's production entry point.

Usage
-----
Verify current outputs:

    .\\.venv\\Scripts\\python.exe .\\run_full_pipeline.py --mode verify

Run complete system:

    .\\.venv\\Scripts\\python.exe .\\run_full_pipeline.py --mode run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =============================================================================
# SYSTEM IMPORTS
# =============================================================================

from system import pipeline_config as cfg

from system.preflight import (
    run_preflight,
)

from system.output_collector import (
    build_full_agent_trace,
)

from system.integration_validator import (
    run_integration_validation,
)


# =============================================================================
# CONSTANTS
# =============================================================================

SUPPORTED_MODES = {
    "verify",
    "run",
}

SYSTEM_RUNS_ROOT = (
    cfg.SYSTEM_REPORT_ROOT
    / "runs"
)

CURRENT_RUN_REPORT = (
    cfg.SYSTEM_LATEST_DIR
    / "system_summary.json"
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def separator(
    char: str = "=",
    width: int = 110,
) -> str:
    return char * width


def header(
    title: str,
) -> None:

    print()
    print(
        separator()
    )

    print(
        title
    )

    print(
        separator()
    )


def section(
    title: str,
) -> None:

    print()
    print(
        title
    )

    print(
        "-" * 110
    )


def status_line(
    label: str,
    status: str,
    detail: str = "",
) -> None:

    print(
        f"{label:<50} : {status}"
    )

    if detail:

        print(
            f"{'':<50}   {detail}"
        )


# =============================================================================
# RUN ID
# =============================================================================

def create_run_id() -> str:
    """
    Create unique experiment/run identifier.
    """

    return datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )


# =============================================================================
# DIRECTORY INITIALIZATION
# =============================================================================

def ensure_directories() -> None:

    cfg.ensure_system_directories()

    SYSTEM_RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def path_ready(
    path: Path,
) -> bool:

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


def existing_path_from_group(
    group: Tuple[Path, ...],
) -> Optional[Path]:
    """
    Return first non-empty file from an OR-output group.
    """

    for path in group:

        if path_ready(
            path
        ):

            return path

    return None


def output_groups_exist(
    agent_number: int,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate every required output group for an agent.
    """

    definition = (
        cfg.get_agent_description(
            agent_number
        )
    )

    groups_report: Dict[str, Any] = {}

    all_valid = True

    for index, group in enumerate(
        definition.required_output_groups,
        start=1,
    ):

        existing = (
            existing_path_from_group(
                group
            )
        )

        group_valid = (
            existing is not None
        )

        groups_report[
            f"group_{index}"
        ] = {
            "valid":
                group_valid,

            "source":
                (
                    str(
                        existing
                    )
                    if existing
                    else None
                ),

            "candidates":
                [
                    str(
                        path
                    )
                    for path in group
                ],
        }

        if not group_valid:

            all_valid = False

    return (
        all_valid,
        groups_report,
    )


# =============================================================================
# OUTPUT FRESHNESS
# =============================================================================

def output_groups_fresh(
    agent_number: int,
    stage_started_at: float,
    tolerance_seconds: float = 3.0,
) -> Tuple[
    bool,
    Dict[str, Any],
]:
    """
    Confirm each required output group contains at least one file
    created or rewritten during the current agent execution.

    Why this matters
    ----------------
    Imagine:

        old Agent-3 output exists
        Agent-3 execution fails silently
        Agent-4 reads old output

    That could mix two different pipeline runs.

    Freshness checks protect against that problem.

    A small tolerance is included for filesystem timestamp resolution.
    """

    definition = (
        cfg.get_agent_description(
            agent_number
        )
    )

    report: Dict[str, Any] = {}

    overall = True

    threshold = (
        stage_started_at
        -
        tolerance_seconds
    )

    for index, group in enumerate(
        definition.required_output_groups,
        start=1,
    ):

        candidates = []

        fresh_files = []

        for path in group:

            if not path_ready(
                path
            ):

                candidates.append(
                    {
                        "path":
                            str(
                                path
                            ),

                        "exists":
                            False,

                        "modified":
                            None,

                        "fresh":
                            False,
                    }
                )

                continue

            modified_time = (
                path.stat().st_mtime
            )

            fresh = (
                modified_time
                >=
                threshold
            )

            candidates.append(
                {
                    "path":
                        str(
                            path
                        ),

                    "exists":
                        True,

                    "modified":
                        datetime.fromtimestamp(
                            modified_time
                        ).isoformat(
                            timespec="seconds"
                        ),

                    "fresh":
                        fresh,
                }
            )

            if fresh:

                fresh_files.append(
                    str(
                        path
                    )
                )

        group_fresh = (
            len(
                fresh_files
            )
            >
            0
        )

        report[
            f"group_{index}"
        ] = {
            "fresh":
                group_fresh,

            "fresh_files":
                fresh_files,

            "candidates":
                candidates,
        }

        if not group_fresh:

            overall = False

    return (
        overall,
        report,
    )


# =============================================================================
# SUBPROCESS EXECUTION
# =============================================================================

def run_python_script(
    script: Path,
    label: str,
) -> Dict[str, Any]:
    """
    Execute a Python script using the same interpreter as this runner.

    stdout/stderr remain attached to the current terminal so the user
    can see the agent's normal logs in real time.
    """

    if not path_ready(
        script
    ):

        raise FileNotFoundError(
            f"{label} script not found or empty:\n"
            f"{script}"
        )

    section(
        label
    )

    print(
        f"Python : {sys.executable}"
    )

    print(
        f"Script : {script}"
    )

    print(
        f"CWD    : {PROJECT_ROOT}"
    )

    print()

    started_at = (
        time.time()
    )

    started_iso = (
        datetime.now()
        .isoformat(
            timespec="seconds"
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            str(
                script
            ),
        ],
        cwd=str(
            PROJECT_ROOT
        ),
        check=False,
    )

    ended_at = (
        time.time()
    )

    ended_iso = (
        datetime.now()
        .isoformat(
            timespec="seconds"
        )
    )

    duration = (
        ended_at
        -
        started_at
    )

    success = (
        result.returncode
        ==
        0
    )

    print()
    print(
        "-" * 110
    )

    print(
        f"{label} return code : "
        f"{result.returncode}"
    )

    print(
        f"{label} duration    : "
        f"{duration:.2f} seconds"
    )

    print(
        f"{label} status      : "
        f"{'PASS' if success else 'FAIL'}"
    )

    return {
        "label":
            label,

        "script":
            str(
                script
            ),

        "started_at":
            started_iso,

        "ended_at":
            ended_iso,

        "start_epoch":
            started_at,

        "duration_seconds":
            duration,

        "return_code":
            result.returncode,

        "success":
            success,
    }


# =============================================================================
# SINGLE AGENT EXECUTION
# =============================================================================

def execute_agent(
    agent_number: int,
) -> Dict[str, Any]:
    """
    Execute one production agent.

    Conditions:
    1. Entrypoint must exist.
    2. Process must return code 0.
    3. Required output groups must exist.
    4. Required outputs must be refreshed during this execution.
    """

    definition = (
        cfg.get_agent_description(
            agent_number
        )
    )

    entrypoint = (
        cfg.resolve_agent_entrypoint(
            agent_number
        )
    )

    if entrypoint is None:

        raise FileNotFoundError(
            f"No valid entry point found for "
            f"Agent {agent_number}: "
            f"{definition.name}"
        )

    label = (
        f"AGENT {agent_number} - "
        f"{definition.name.upper()}"
    )

    stage_result = (
        run_python_script(
            script=entrypoint,
            label=label,
        )
    )

    if not stage_result[
        "success"
    ]:

        stage_result[
            "output_check"
        ] = False

        stage_result[
            "freshness_check"
        ] = False

        return stage_result

    # -------------------------------------------------------------------------
    # Required output existence
    # -------------------------------------------------------------------------

    (
        output_valid,
        output_report,
    ) = output_groups_exist(
        agent_number
    )

    stage_result[
        "output_check"
    ] = output_valid

    stage_result[
        "output_report"
    ] = output_report

    # -------------------------------------------------------------------------
    # Current-run freshness
    # -------------------------------------------------------------------------

    (
        freshness_valid,
        freshness_report,
    ) = output_groups_fresh(
        agent_number=agent_number,
        stage_started_at=stage_result[
            "start_epoch"
        ],
    )

    stage_result[
        "freshness_check"
    ] = freshness_valid

    stage_result[
        "freshness_report"
    ] = freshness_report

    stage_result[
        "success"
    ] = bool(
        stage_result[
            "success"
        ]
        and
        output_valid
        and
        freshness_valid
    )

    print()

    status_line(
        "Required output groups",
        (
            "PASS"
            if output_valid
            else "FAIL"
        ),
    )

    status_line(
        "Current-run output freshness",
        (
            "PASS"
            if freshness_valid
            else "FAIL"
        ),
    )

    print()

    if stage_result[
        "success"
    ]:

        print(
            f"AGENT {agent_number} STATUS: PASS"
        )

    else:

        print(
            f"AGENT {agent_number} STATUS: FAIL"
        )

    return stage_result


# =============================================================================
# SYSTEM STAGE TIMING
# =============================================================================

def run_internal_stage(
    label: str,
    function,
    *args,
    **kwargs,
) -> Tuple[
    Any,
    Dict[str, Any],
]:
    """
    Run internal Python system stage and measure execution time.
    """

    section(
        label
    )

    started = (
        time.time()
    )

    started_iso = (
        datetime.now()
        .isoformat(
            timespec="seconds"
        )
    )

    try:

        result = function(
            *args,
            **kwargs,
        )

        success = True

        error = None

    except Exception as exc:

        result = None

        success = False

        error = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    ended = (
        time.time()
    )

    ended_iso = (
        datetime.now()
        .isoformat(
            timespec="seconds"
        )
    )

    duration = (
        ended
        -
        started
    )

    stage = {
        "label":
            label,

        "started_at":
            started_iso,

        "ended_at":
            ended_iso,

        "duration_seconds":
            duration,

        "success":
            success,

        "error":
            error,
    }

    if success:

        print()
        print(
            f"{label} : PASS"
        )

    else:

        print()
        print(
            f"{label} : FAIL"
        )

        print(
            error
        )

    return (
        result,
        stage,
    )


# =============================================================================
# TIMINGS REPORT
# =============================================================================

def save_stage_timings(
    stages: List[
        Dict[str, Any]
    ],
) -> None:
    """
    Save stage execution timing table.
    """

    rows = []

    for stage in stages:

        rows.append(
            {
                "stage":
                    stage.get(
                        "label"
                    ),

                "started_at":
                    stage.get(
                        "started_at"
                    ),

                "ended_at":
                    stage.get(
                        "ended_at"
                    ),

                "duration_seconds":
                    stage.get(
                        "duration_seconds"
                    ),

                "status":
                    (
                        "PASS"
                        if stage.get(
                            "success"
                        )
                        else "FAIL"
                    ),

                "return_code":
                    stage.get(
                        "return_code"
                    ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        cfg.SYSTEM_STAGE_TIMINGS_CSV,
        index=False,
    )


# =============================================================================
# SNAPSHOT
# =============================================================================

def copy_if_ready(
    source: Path,
    destination: Path,
) -> Optional[str]:

    if not path_ready(
        source
    ):

        return None

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    return str(
        destination
    )


def snapshot_run(
    run_id: str,
) -> Dict[str, Any]:
    """
    Preserve the most important outputs of one successful system run.

    This is intentionally limited to lightweight output/report files.

    Large model files and historical market datasets are NOT copied.
    """

    run_dir = (
        SYSTEM_RUNS_ROOT
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = {
        # Agent 1
        "agent1_selected_stocks.csv":
            cfg.AGENT1_SELECTED_STOCKS_CSV,

        "agent1_selected_stocks.parquet":
            cfg.AGENT1_SELECTED_STOCKS_PARQUET,

        "agent1_rankings.csv":
            cfg.AGENT1_RANKINGS_CSV,

        # Agent 2
        "agent2_trend_predictions.csv":
            cfg.AGENT2_TREND_PREDICTIONS_CSV,

        "agent2_trend_predictions.parquet":
            cfg.AGENT2_TREND_PREDICTIONS_PARQUET,

        # Agent 3
        "agent3_risk_assessment.csv":
            cfg.AGENT3_RISK_CSV,

        "agent3_risk_assessment.parquet":
            cfg.AGENT3_RISK_PARQUET,

        # Agent 4
        "agent4_trade_decisions.csv":
            cfg.AGENT4_TRADE_DECISIONS_CSV,

        "agent4_trade_decisions.parquet":
            cfg.AGENT4_TRADE_DECISIONS_PARQUET,

        # Agent 5
        "agent5_rebalanced_portfolio.csv":
            cfg.AGENT5_REBALANCED_PORTFOLIO_CSV,

        "agent5_rebalanced_portfolio.parquet":
            cfg.AGENT5_REBALANCED_PORTFOLIO_PARQUET,

        "agent5_final_portfolio_allocation.csv":
            cfg.AGENT5_FINAL_ALLOCATION_CSV,

        # System
        "full_agent_trace.csv":
            cfg.SYSTEM_FULL_TRACE_CSV,

        "full_agent_trace.parquet":
            cfg.SYSTEM_FULL_TRACE_PARQUET,

        "preflight_summary.json":
            (
                cfg.SYSTEM_LATEST_DIR
                / "preflight_summary.json"
            ),

        "output_collector_summary.json":
            (
                cfg.SYSTEM_LATEST_DIR
                / "output_collector_summary.json"
            ),

        "integration_summary.json":
            cfg.SYSTEM_INTEGRATION_JSON,

        "stage_timings.csv":
            cfg.SYSTEM_STAGE_TIMINGS_CSV,
    }

    copied: Dict[str, Any] = {}

    for filename, source in (
        sources.items()
    ):

        destination = (
            run_dir
            / filename
        )

        copied[
            filename
        ] = copy_if_ready(
            source,
            destination,
        )

    return {
        "run_id":
            run_id,

        "run_directory":
            str(
                run_dir
            ),

        "files":
            copied,
    }


# =============================================================================
# SAVE SYSTEM SUMMARY
# =============================================================================

def save_system_summary(
    report: Dict[str, Any],
) -> None:

    cfg.ensure_system_directories()

    with open(
        CURRENT_RUN_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )


# =============================================================================
# VERIFY MODE
# =============================================================================

def execute_verify_mode(
    run_id: str,
) -> Dict[str, Any]:
    """
    Verify current five-agent outputs without executing any agent.
    """

    header(
        "FULL SYSTEM - VERIFY MODE"
    )

    stage_records: List[
        Dict[str, Any]
    ] = []

    # =========================================================================
    # PREFLIGHT VERIFY
    # =========================================================================

    (
        preflight_report,
        preflight_stage,
    ) = run_internal_stage(
        "SYSTEM PREFLIGHT - VERIFY",
        run_preflight,
        "verify",
    )

    stage_records.append(
        preflight_stage
    )

    if (
        not preflight_stage[
            "success"
        ]
        or
        not preflight_report
        or
        not preflight_report.get(
            "overall_pass",
            False,
        )
    ):

        raise RuntimeError(
            "System preflight verification failed."
        )

    # =========================================================================
    # OUTPUT COLLECTOR
    # =========================================================================

    (
        collector_result,
        collector_stage,
    ) = run_internal_stage(
        "SYSTEM OUTPUT COLLECTOR",
        build_full_agent_trace,
    )

    stage_records.append(
        collector_stage
    )

    if not collector_stage[
        "success"
    ]:

        raise RuntimeError(
            "Output collector failed."
        )

    # =========================================================================
    # INTEGRATION VALIDATOR
    # =========================================================================

    (
        integration_report,
        integration_stage,
    ) = run_internal_stage(
        "FULL SYSTEM INTEGRATION VALIDATION",
        run_integration_validation,
    )

    stage_records.append(
        integration_stage
    )

    if (
        not integration_stage[
            "success"
        ]
        or
        not integration_report
        or
        not integration_report.get(
            "overall_pass",
            False,
        )
    ):

        raise RuntimeError(
            "Full system integration validation failed."
        )

    # =========================================================================
    # TIMINGS
    # =========================================================================

    save_stage_timings(
        stage_records
    )

    return {
        "run_id":
            run_id,

        "mode":
            "verify",

        "status":
            "PASS",

        "overall_pass":
            True,

        "stages":
            stage_records,

        "preflight":
            preflight_report,

        "integration":
            integration_report,
    }


# =============================================================================
# RUN MODE
# =============================================================================

def execute_run_mode(
    run_id: str,
) -> Dict[str, Any]:
    """
    Execute the complete five-agent pipeline.
    """

    header(
        "FULL SYSTEM - RUN MODE"
    )

    stage_records: List[
        Dict[str, Any]
    ] = []

    agent_results: Dict[
        str,
        Dict[str, Any],
    ] = {}

    # =========================================================================
    # INITIAL RUN PREFLIGHT
    # =========================================================================

    (
        run_preflight_report,
        run_preflight_stage,
    ) = run_internal_stage(
        "PRE-RUN SYSTEM PREFLIGHT",
        run_preflight,
        "run",
    )

    stage_records.append(
        run_preflight_stage
    )

    if (
        not run_preflight_stage[
            "success"
        ]
        or
        not run_preflight_report
        or
        not run_preflight_report.get(
            "overall_pass",
            False,
        )
    ):

        raise RuntimeError(
            "Pre-run system preflight failed. "
            "Agents were not started."
        )

    # =========================================================================
    # AGENT 1 -> AGENT 5
    # =========================================================================

    for agent_number in (
        cfg.PIPELINE_ORDER
    ):

        result = execute_agent(
            agent_number
        )

        stage_records.append(
            result
        )

        agent_results[
            f"agent{agent_number}"
        ] = result

        if not result.get(
            "success",
            False,
        ):

            raise RuntimeError(
                f"Agent {agent_number} failed. "
                f"Pipeline stopped before Agent "
                f"{agent_number + 1 if agent_number < 5 else 'completion'}."
            )

    # =========================================================================
    # OUTPUT COLLECTOR
    # =========================================================================

    (
        collector_result,
        collector_stage,
    ) = run_internal_stage(
        "BUILDING UNIFIED FIVE-AGENT TRACE",
        build_full_agent_trace,
    )

    stage_records.append(
        collector_stage
    )

    if not collector_stage[
        "success"
    ]:

        raise RuntimeError(
            "All five agents completed, "
            "but output collection failed."
        )

    # =========================================================================
    # POST-RUN PREFLIGHT VERIFICATION
    # =========================================================================

    (
        final_preflight_report,
        final_preflight_stage,
    ) = run_internal_stage(
        "POST-RUN SYSTEM PREFLIGHT",
        run_preflight,
        "verify",
    )

    stage_records.append(
        final_preflight_stage
    )

    if (
        not final_preflight_stage[
            "success"
        ]
        or
        not final_preflight_report
        or
        not final_preflight_report.get(
            "overall_pass",
            False,
        )
    ):

        raise RuntimeError(
            "Post-run verification failed."
        )

    # =========================================================================
    # FULL INTEGRATION VALIDATION
    # =========================================================================

    (
        integration_report,
        integration_stage,
    ) = run_internal_stage(
        "FINAL AGENT 1 -> 5 INTEGRATION VALIDATION",
        run_integration_validation,
    )

    stage_records.append(
        integration_stage
    )

    if (
        not integration_stage[
            "success"
        ]
        or
        not integration_report
        or
        not integration_report.get(
            "overall_pass",
            False,
        )
    ):

        raise RuntimeError(
            "Final integration validation failed."
        )

    # =========================================================================
    # STAGE TIMINGS
    # =========================================================================

    save_stage_timings(
        stage_records
    )

    # =========================================================================
    # SNAPSHOT
    # =========================================================================

    snapshot = (
        snapshot_run(
            run_id
        )
    )

    return {
        "run_id":
            run_id,

        "mode":
            "run",

        "status":
            "PASS",

        "overall_pass":
            True,

        "agents":
            agent_results,

        "stages":
            stage_records,

        "initial_preflight":
            run_preflight_report,

        "final_preflight":
            final_preflight_report,

        "integration":
            integration_report,

        "snapshot":
            snapshot,
    }


# =============================================================================
# FINAL SUMMARY DISPLAY
# =============================================================================

def display_final_summary(
    report: Dict[str, Any],
) -> None:

    header(
        "AGENTIC AI PORTFOLIO OPTIMIZATION - FINAL SYSTEM SUMMARY"
    )

    print(
        f"Run ID               : "
        f"{report.get('run_id')}"
    )

    print(
        f"Mode                 : "
        f"{str(report.get('mode')).upper()}"
    )

    print(
        f"Inference date       : "
        f"{cfg.FINAL_INFERENCE_DATE}"
    )

    print(
        f"System status        : "
        f"{report.get('status')}"
    )

    print()

    if (
        report.get(
            "mode"
        )
        ==
        "run"
    ):

        print(
            "Agent execution"
        )

        print(
            "-" * 110
        )

        agent_results = (
            report.get(
                "agents",
                {},
            )
        )

        for number in (
            cfg.PIPELINE_ORDER
        ):

            result = (
                agent_results
                .get(
                    f"agent{number}",
                    {},
                )
            )

            status_line(
                (
                    f"Agent {number} - "
                    f"{cfg.AGENT_NAMES[number]}"
                ),
                (
                    "PASS"
                    if result.get(
                        "success",
                        False,
                    )
                    else "FAIL"
                ),
                (
                    f"{result.get('duration_seconds', 0):.2f} sec"
                ),
            )

        print()

    integration = (
        report.get(
            "integration",
            {}
        )
    )

    group_status = (
        integration.get(
            "group_status",
            {}
        )
    )

    if group_status:

        print(
            "Integration"
        )

        print(
            "-" * 110
        )

        for name, passed in (
            group_status.items()
        ):

            status_line(
                name,
                (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            )

    print()

    if path_ready(
        cfg.SYSTEM_FULL_TRACE_CSV
    ):

        print(
            f"Full agent trace     : "
            f"{cfg.SYSTEM_FULL_TRACE_CSV}"
        )

    if path_ready(
        cfg.AGENT5_FINAL_ALLOCATION_CSV
    ):

        print(
            f"Final portfolio      : "
            f"{cfg.AGENT5_FINAL_ALLOCATION_CSV}"
        )

    if path_ready(
        cfg.SYSTEM_INTEGRATION_JSON
    ):

        print(
            f"Integration report   : "
            f"{cfg.SYSTEM_INTEGRATION_JSON}"
        )

    if path_ready(
        cfg.SYSTEM_STAGE_TIMINGS_CSV
    ):

        print(
            f"Stage timings        : "
            f"{cfg.SYSTEM_STAGE_TIMINGS_CSV}"
        )

    snapshot = (
        report.get(
            "snapshot"
        )
    )

    if snapshot:

        print(
            f"Run snapshot         : "
            f"{snapshot.get('run_directory')}"
        )

    print()

    print(
        separator()
    )

    if report.get(
        "overall_pass",
        False,
    ):

        print(
            "AGENT 1 -> AGENT 2 -> AGENT 3 -> "
            "AGENT 4 -> AGENT 5 : COMPLETE"
        )

        print()

        print(
            "FULL SYSTEM STATUS: PASS"
        )

    else:

        print(
            "FULL SYSTEM STATUS: FAIL"
        )

    print(
        separator()
    )


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_arguments() -> (
    argparse.Namespace
):

    parser = argparse.ArgumentParser(
        description=(
            "Execute or verify the complete "
            "five-agent portfolio optimization system."
        )
    )

    parser.add_argument(
        "--mode",
        choices=sorted(
            SUPPORTED_MODES
        ),
        default="verify",
        help=(
            "verify: inspect existing outputs without "
            "rerunning agents; "
            "run: execute Agent 1 -> Agent 5."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    ensure_directories()

    arguments = (
        parse_arguments()
    )

    run_id = (
        create_run_id()
    )

    started_at = (
        datetime.now()
        .isoformat(
            timespec="seconds"
        )
    )

    header(
        "AGENTIC AI PORTFOLIO OPTIMIZATION "
        "- COMPLETE FIVE-AGENT SYSTEM"
    )

    print(
        f"Run ID               : "
        f"{run_id}"
    )

    print(
        f"Mode                 : "
        f"{arguments.mode.upper()}"
    )

    print(
        f"Project root         : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Python               : "
        f"{sys.executable}"
    )

    print(
        f"Inference date       : "
        f"{cfg.FINAL_INFERENCE_DATE}"
    )

    print()

    report: Dict[
        str,
        Any,
    ]

    try:

        if (
            arguments.mode
            ==
            "verify"
        ):

            report = (
                execute_verify_mode(
                    run_id
                )
            )

        elif (
            arguments.mode
            ==
            "run"
        ):

            report = (
                execute_run_mode(
                    run_id
                )
            )

        else:

            raise ValueError(
                f"Unsupported mode: "
                f"{arguments.mode}"
            )

        report[
            "started_at"
        ] = started_at

        report[
            "finished_at"
        ] = (
            datetime.now()
            .isoformat(
                timespec="seconds"
            )
        )

        save_system_summary(
            report
        )

        display_final_summary(
            report
        )

        return 0

    except KeyboardInterrupt:

        print()

        header(
            "SYSTEM EXECUTION INTERRUPTED"
        )

        print(
            "Execution was interrupted by the user."
        )

        return 130

    except Exception as error:

        failed_report = {
            "run_id":
                run_id,

            "mode":
                arguments.mode,

            "status":
                "FAIL",

            "overall_pass":
                False,

            "started_at":
                started_at,

            "finished_at":
                datetime.now()
                .isoformat(
                    timespec="seconds"
                ),

            "error_type":
                type(
                    error
                ).__name__,

            "error":
                str(
                    error
                ),
        }

        save_system_summary(
            failed_report
        )

        header(
            "FULL SYSTEM EXECUTION FAILED"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print()

        print(
            "The pipeline has stopped."
        )

        print(
            "Downstream agents were not intentionally "
            "executed after the failed stage."
        )

        print()

        print(
            f"Failure report : "
            f"{CURRENT_RUN_REPORT}"
        )

        print(
            separator()
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )