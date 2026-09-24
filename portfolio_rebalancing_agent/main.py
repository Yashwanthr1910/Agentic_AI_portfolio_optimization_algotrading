"""
Agent 5 - Portfolio Rebalancing Main Orchestrator
===============================================================================

PROJECT
-------
Agentic AI Portfolio Optimization and Algorithmic Trading

AGENT
-----
Agent 5 - Portfolio Rebalancing Agent

PURPOSE
-------
Run the complete Agent-5 pipeline in the correct order.

PIPELINE
--------
Agent 4 final DQN decisions
        ↓
1. Agent-5 Data Loader
        ↓
2. Portfolio Metrics Validation
        ↓
3. Markowitz MPT Optimizer
        ↓
4. RL Feedback
        ↓
5. Rebalancing Environment
        ↓
6. Final Rebalancing Agent
        ↓
7. Portfolio Evaluator
        ↓
8. Agent-4 -> Agent-5 Integration Validator
        ↓
AGENT 5 STATUS: COMPLETE


IMPORTANT
---------
This file does NOT retrain Agent 4.

It consumes the existing:

    trade_execution_agent/outputs/trade_decisions.parquet

and executes only Agent 5.


USAGE
-----
Full Agent-5 pipeline:

    python portfolio_rebalancing_agent/main.py

Verify existing outputs only:

    python portfolio_rebalancing_agent/main.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


# =============================================================================
# PATH SETUP
# =============================================================================

AGENT_ROOT = (
    Path(__file__)
    .resolve()
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


# =============================================================================
# IMPORT CONFIG
# =============================================================================

from config import config as cfg


# =============================================================================
# IMPORT AGENT-5 STAGES
# =============================================================================

from src.data_loader import (
    prepare_rebalancing_data,
)

from src.portfolio_metrics import (
    run_portfolio_metrics_smoke_test,
)

from src.mpt_optimizer import (
    run_mpt_optimization,
)

from src.rl_feedback import (
    run_rl_feedback,
)

from src.rebalancing_environment import (
    run_rebalancing_environment,
)

from src.rebalancing_agent import (
    run_rebalancing_agent,
)

from src.evaluator import (
    run_evaluation,
)

from src.integration_validator import (
    run_integration_validation,
)


# =============================================================================
# FINAL MAIN REPORT
# =============================================================================

MAIN_SUMMARY_PATH = (
    cfg.REPORT_DIR
    / "main_summary.json"
)


# =============================================================================
# STAGE DISPLAY
# =============================================================================

def print_header(
    title: str,
) -> None:
    """
    Print standard section header.
    """

    print()
    print(
        "=" * 100
    )

    print(
        title
    )

    print(
        "=" * 100
    )

    print()


# =============================================================================
# FILE CHECK
# =============================================================================

def file_exists(
    path: Path,
) -> bool:
    """
    Safe file-existence helper.
    """

    return Path(
        path
    ).exists()


# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================

def run_preflight_checks() -> dict[str, bool]:
    """
    Confirm required upstream Agent-4 / Agent-3 / market inputs exist.
    """

    print_header(
        "AGENT 5 - PREFLIGHT CHECKS"
    )

    checks = {
        "agent4_trade_decisions":
            (
                file_exists(
                    cfg.AGENT4_TRADE_DECISIONS_PARQUET
                )
                or
                file_exists(
                    cfg.AGENT4_TRADE_DECISIONS_CSV
                )
            ),

        "agent3_risk_assessment":
            (
                file_exists(
                    cfg.AGENT3_RISK_PARQUET
                )
                or
                file_exists(
                    cfg.AGENT3_RISK_CSV
                )
            ),

        "historical_market_data":
            (
                file_exists(
                    cfg.MARKET_DATA_PARQUET
                )
                or
                file_exists(
                    cfg.MARKET_DATA_CSV
                )
            ),
    }

    for name, passed in checks.items():

        print(
            f"{name:<55}: "
            f"{passed}"
        )

    overall_pass = all(
        checks.values()
    )

    print()

    print(
        "PREFLIGHT RESULT: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    if not overall_pass:

        failed = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        raise FileNotFoundError(
            "Agent-5 preflight failed:\n"
            +
            "\n".join(
                f"  - {name}"
                for name
                in failed
            )
        )

    return checks


# =============================================================================
# STAGE RUNNER
# =============================================================================

def run_stage(
    stage_number: int,
    stage_name: str,
    function,
) -> dict[str, Any]:
    """
    Run one Agent-5 stage and capture execution metadata.
    """

    print_header(
        f"STAGE {stage_number} - {stage_name}"
    )

    start_time = time.perf_counter()

    try:

        result = function()

        duration = (
            time.perf_counter()
            -
            start_time
        )

        print()

        print(
            f"{stage_name:<40}: PASS"
        )

        print(
            f"{'Duration':<40}: "
            f"{duration:.3f} seconds"
        )

        return {
            "status":
                "PASS",

            "duration_seconds":
                float(
                    duration
                ),

            "result":
                result,
        }

    except Exception as error:

        duration = (
            time.perf_counter()
            -
            start_time
        )

        print()

        print(
            f"{stage_name:<40}: FAIL"
        )

        print(
            f"{'Duration':<40}: "
            f"{duration:.3f} seconds"
        )

        print(
            f"{'Error type':<40}: "
            f"{type(error).__name__}"
        )

        print(
            f"{'Error':<40}: "
            f"{error}"
        )

        raise


# =============================================================================
# VERIFY EXISTING OUTPUTS
# =============================================================================

def verify_existing_outputs() -> dict[str, bool]:
    """
    Verify Agent-5 outputs without rerunning all computation.
    """

    print_header(
        "AGENT 5 - EXISTING OUTPUT VERIFICATION"
    )

    checks = {
        "returns_matrix":
            (
                file_exists(
                    cfg.RETURNS_MATRIX_PARQUET
                )
                or
                file_exists(
                    cfg.RETURNS_MATRIX_CSV
                )
            ),

        "covariance_matrix":
            file_exists(
                cfg.COVARIANCE_MATRIX_CSV
            ),

        "correlation_matrix":
            file_exists(
                cfg.CORRELATION_MATRIX_CSV
            ),

        "rebalancing_input":
            (
                file_exists(
                    cfg.REBALANCING_INPUT_PARQUET
                )
                or
                file_exists(
                    cfg.REBALANCING_INPUT_CSV
                )
            ),

        "mpt_solution":
            file_exists(
                cfg.MPT_SOLUTION_CSV
            ),

        "rl_adjusted_weights":
            file_exists(
                cfg.RL_ADJUSTED_WEIGHTS_CSV
            ),

        "rebalanced_portfolio_csv":
            file_exists(
                cfg.REBALANCED_PORTFOLIO_CSV
            ),

        "rebalanced_portfolio_parquet":
            file_exists(
                cfg.REBALANCED_PORTFOLIO_PARQUET
            ),

        "final_portfolio_allocation":
            file_exists(
                cfg.FINAL_PORTFOLIO_ALLOCATION_CSV
            ),

        "data_loader_summary":
            file_exists(
                cfg.DATA_LOADER_SUMMARY
            ),

        "portfolio_metrics_report":
            file_exists(
                cfg.PORTFOLIO_METRICS_REPORT
            ),

        "mpt_optimization_summary":
            file_exists(
                cfg.MPT_OPTIMIZATION_SUMMARY
            ),

        "rl_feedback_summary":
            file_exists(
                cfg.RL_FEEDBACK_SUMMARY
            ),

        "evaluation_summary":
            file_exists(
                cfg.EVALUATION_SUMMARY
            ),

        "agent5_summary":
            file_exists(
                cfg.AGENT5_SUMMARY
            ),

        "agent4_agent5_validation":
            file_exists(
                cfg.AGENT4_AGENT5_VALIDATION
            ),
    }

    for name, passed in checks.items():

        print(
            f"{name:<60}: "
            f"{passed}"
        )

    overall_pass = all(
        checks.values()
    )

    print()

    print(
        "OUTPUT VERIFICATION RESULT: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    return checks


# =============================================================================
# LOAD FINAL INTEGRATION REPORT
# =============================================================================

def load_final_integration_report() -> dict[str, Any]:
    """
    Load final Agent-4 -> Agent-5 validation report.
    """

    if not cfg.AGENT4_AGENT5_VALIDATION.exists():

        raise FileNotFoundError(
            "Agent-4 -> Agent-5 validation report "
            "was not found."
        )

    with open(
        cfg.AGENT4_AGENT5_VALIDATION,
        "r",
        encoding="utf-8",
    ) as file:

        report = json.load(
            file
        )

    return report


# =============================================================================
# VERIFY-ONLY MODE
# =============================================================================

def run_verify_only() -> None:
    """
    Verify existing Agent-5 results.

    No MPT re-optimization is performed.
    """

    print_header(
        "AGENT 5 - VERIFY ONLY MODE"
    )

    preflight = (
        run_preflight_checks()
    )

    output_checks = (
        verify_existing_outputs()
    )

    print()

    print(
        "Running final Agent-4 -> Agent-5 validator..."
    )

    integration_result = (
        run_integration_validation()
    )

    overall_pass = (
        all(
            preflight.values()
        )
        and
        all(
            output_checks.values()
        )
        and
        all(
            integration_result[
                "checks"
            ].values()
        )
    )

    print_header(
        "AGENT 5 FINAL VERIFICATION"
    )

    print(
        f"Preflight checks        : "
        f"{'PASS' if all(preflight.values()) else 'FAIL'}"
    )

    print(
        f"Output checks           : "
        f"{'PASS' if all(output_checks.values()) else 'FAIL'}"
    )

    print(
        f"Integration validation  : "
        f"{'PASS' if all(integration_result['checks'].values()) else 'FAIL'}"
    )

    print()

    print(
        "AGENT 5 STATUS: "
        f"{'COMPLETE' if overall_pass else 'INCOMPLETE'}"
    )

    if not overall_pass:

        raise RuntimeError(
            "Agent-5 verification failed."
        )


# =============================================================================
# COMPLETE AGENT-5 PIPELINE
# =============================================================================

def run_agent5_pipeline() -> dict[str, Any]:
    """
    Execute complete Agent-5 workflow.
    """

    pipeline_start = (
        time.perf_counter()
    )

    print_header(
        "AGENT 5 - PORTFOLIO REBALANCING PIPELINE"
    )

    print(
        f"Project root            : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Agent root              : "
        f"{AGENT_ROOT}"
    )

    print(
        f"Rebalance date          : "
        f"{cfg.FINAL_REBALANCE_DATE}"
    )

    print()

    print(
        "Method                  : "
        "Markowitz MPT + Agent-4 DQN RL feedback"
    )

    # =========================================================================
    # PREFLIGHT
    # =========================================================================

    preflight_checks = (
        run_preflight_checks()
    )

    # =========================================================================
    # STAGE RESULTS
    # =========================================================================

    stages: dict[
        str,
        dict[str, Any],
    ] = {}

    # =========================================================================
    # 1 - DATA LOADER
    # =========================================================================

    stages[
        "data_loader"
    ] = run_stage(
        stage_number=1,
        stage_name="DATA LOADER",
        function=prepare_rebalancing_data,
    )

    # =========================================================================
    # 2 - PORTFOLIO METRICS
    # =========================================================================

    stages[
        "portfolio_metrics"
    ] = run_stage(
        stage_number=2,
        stage_name="PORTFOLIO METRICS",
        function=run_portfolio_metrics_smoke_test,
    )

    # =========================================================================
    # 3 - MPT
    # =========================================================================

    stages[
        "mpt_optimizer"
    ] = run_stage(
        stage_number=3,
        stage_name="MARKOWITZ MPT OPTIMIZER",
        function=run_mpt_optimization,
    )

    # =========================================================================
    # 4 - RL FEEDBACK
    # =========================================================================

    stages[
        "rl_feedback"
    ] = run_stage(
        stage_number=4,
        stage_name="RL FEEDBACK",
        function=run_rl_feedback,
    )

    # =========================================================================
    # 5 - REBALANCING ENVIRONMENT
    # =========================================================================

    stages[
        "rebalancing_environment"
    ] = run_stage(
        stage_number=5,
        stage_name="REBALANCING ENVIRONMENT",
        function=run_rebalancing_environment,
    )

    # =========================================================================
    # 6 - FINAL AGENT
    # =========================================================================

    stages[
        "rebalancing_agent"
    ] = run_stage(
        stage_number=6,
        stage_name="PORTFOLIO REBALANCING AGENT",
        function=run_rebalancing_agent,
    )

    # =========================================================================
    # 7 - EVALUATOR
    # =========================================================================

    stages[
        "evaluator"
    ] = run_stage(
        stage_number=7,
        stage_name="PORTFOLIO EVALUATOR",
        function=run_evaluation,
    )

    # =========================================================================
    # 8 - INTEGRATION VALIDATOR
    # =========================================================================

    stages[
        "integration_validator"
    ] = run_stage(
        stage_number=8,
        stage_name="AGENT 4 -> AGENT 5 VALIDATOR",
        function=run_integration_validation,
    )

    # =========================================================================
    # OUTPUT CHECKS
    # =========================================================================

    output_checks = (
        verify_existing_outputs()
    )

    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    all_stage_pass = all(
        stage[
            "status"
        ]
        ==
        "PASS"

        for stage
        in stages.values()
    )

    integration_checks = (
        stages[
            "integration_validator"
        ][
            "result"
        ][
            "checks"
        ]
    )

    integration_pass = all(
        integration_checks.values()
    )

    preflight_pass = all(
        preflight_checks.values()
    )

    output_pass = all(
        output_checks.values()
    )

    overall_pass = bool(
        preflight_pass
        and
        all_stage_pass
        and
        integration_pass
        and
        output_pass
    )

    duration = (
        time.perf_counter()
        -
        pipeline_start
    )

    # =========================================================================
    # FINAL SUMMARY REPORT
    # =========================================================================

    final_integration_report = (
        load_final_integration_report()
    )

    summary = {
        "status":
            (
                "COMPLETE"
                if overall_pass
                else
                "FAIL"
            ),

        "agent":
            "Portfolio Rebalancing Agent",

        "agent_number":
            5,

        "rebalance_date":
            cfg.FINAL_REBALANCE_DATE,

        "pipeline":
            (
                "Agent 4 DQN -> "
                "Markowitz MPT -> "
                "RL Feedback -> "
                "Portfolio Rebalancing"
            ),

        "preflight": {
            name:
                bool(
                    value
                )

            for name, value
            in preflight_checks.items()
        },

        "stages": {
            name: {
                "status":
                    stage[
                        "status"
                    ],

                "duration_seconds":
                    float(
                        stage[
                            "duration_seconds"
                        ]
                    ),
            }

            for name, stage
            in stages.items()
        },

        "outputs": {
            name:
                bool(
                    value
                )

            for name, value
            in output_checks.items()
        },

        "integration_validation":
            {
                name:
                    bool(
                        value
                    )

                for name, value
                in integration_checks.items()
            },

        "final_weights":
            final_integration_report.get(
                "final_weights",
                {},
            ),

        "total_duration_seconds":
            float(
                duration
            ),

        "methodology_note":
            (
                "Agent 5 implements a paper-aligned portfolio "
                "rebalancing reconstruction combining Markowitz "
                "minimum-variance optimization with Agent-4 DQN "
                "trade feedback. SELL actions receive zero allocation; "
                "BUY and HOLD assets remain MPT eligible."
            ),

        "evaluation_limitation":
            (
                "Agent-5 portfolio diagnostic metrics use the same "
                "historical 105-day estimation window used to construct "
                "the final allocation. They are not a future "
                "post-rebalance backtest."
            ),
    }

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        MAIN_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # =========================================================================
    # FINAL DISPLAY
    # =========================================================================

    print_header(
        "AGENT 5 - FINAL PIPELINE SUMMARY"
    )

    print(
        f"{'Preflight':<40}: "
        f"{'PASS' if preflight_pass else 'FAIL'}"
    )

    for name, stage in stages.items():

        print(
            f"{name:<40}: "
            f"{stage['status']}"
        )

    print(
        f"{'Output verification':<40}: "
        f"{'PASS' if output_pass else 'FAIL'}"
    )

    print(
        f"{'Integration validation':<40}: "
        f"{'PASS' if integration_pass else 'FAIL'}"
    )

    print(
        f"{'Total duration':<40}: "
        f"{duration:.3f} seconds"
    )

    print()

    print(
        f"Main summary            : "
        f"{MAIN_SUMMARY_PATH}"
    )

    print()

    print(
        "=" * 100
    )

    if overall_pass:

        print(
            "AGENT 5 STATUS: COMPLETE"
        )

    else:

        print(
            "AGENT 5 STATUS: FAILED"
        )

    print(
        "=" * 100
    )

    if not overall_pass:

        raise RuntimeError(
            "Agent-5 pipeline did not complete successfully."
        )

    return {
        "status":
            "COMPLETE",

        "stages":
            stages,

        "preflight":
            preflight_checks,

        "outputs":
            output_checks,

        "integration":
            integration_checks,

        "summary":
            summary,
    }


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Agent 5 - Portfolio Rebalancing Agent"
        )
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Verify existing Agent-5 outputs without "
            "rerunning optimization."
        ),
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Main program entry point.
    """

    args = parse_arguments()

    try:

        if args.verify_only:

            run_verify_only()

        else:

            run_agent5_pipeline()

    except KeyboardInterrupt:

        print()

        print(
            "Agent-5 execution cancelled."
        )

        sys.exit(
            130
        )

    except Exception as error:

        print()
        print(
            "=" * 100
        )

        print(
            "AGENT 5 EXECUTION FAILED"
        )

        print(
            "=" * 100
        )

        print()

        print(
            f"Error type : "
            f"{type(error).__name__}"
        )

        print(
            f"Error      : "
            f"{error}"
        )

        print()

        traceback.print_exc()

        sys.exit(
            1
        )


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    main()