"""
Agent 4 - Trade Execution Agent Main Entry Point
===============================================================================

Default:
    Run final Agent-4 inference and validate Agent-3 -> Agent-4 integration.

This program DOES NOT train the DQN automatically.

Training is deliberately separated from production inference to prevent
accidental overwriting of a validated trained model.

Examples
--------

Complete production run:

    python trade_execution_agent/main.py

Prediction only:

    python trade_execution_agent/main.py --mode predict

Integration validation only:

    python trade_execution_agent/main.py --mode validate

Paper-aligned 2019-2023 backtest:

    python trade_execution_agent/main.py --mode backtest

2024 robustness evaluation:

    python trade_execution_agent/main.py --mode robustness
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# =============================================================================
# PATH SETUP
# =============================================================================

AGENT_ROOT = Path(
    __file__
).resolve().parent

PROJECT_ROOT = (
    AGENT_ROOT.parent
)

if str(
    AGENT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            AGENT_ROOT
        ),
    )

if str(
    PROJECT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from src.agent import (
    run_trade_execution_agent,
)

from src.integration_validator import (
    validate_agent3_agent4,
)

from src.evaluator import (
    run_evaluation,
    run_robustness_evaluation,
)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Agent 4 - DQN Trade Execution Agent"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "run",
            "predict",
            "validate",
            "backtest",
            "robustness",
        ],
        default="run",
        help=(
            "run = prediction + integration validation; "
            "predict = final DQN inference only; "
            "validate = Agent3->Agent4 validation only; "
            "backtest = 2019-2023 paper-aligned evaluation; "
            "robustness = 2024 robustness evaluation."
        ),
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 4 - TRADE EXECUTION"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Execution mode        : "
        f"{args.mode}"
    )

    # -------------------------------------------------------------------------
    # Full production run
    # -------------------------------------------------------------------------

    if args.mode == "run":

        print()
        print(
            "STEP 1/2 - FINAL TRADE INFERENCE"
        )

        run_trade_execution_agent()

        print()
        print(
            "STEP 2/2 - AGENT 3 -> AGENT 4 VALIDATION"
        )

        validate_agent3_agent4()

        print()
        print(
            "=" * 100
        )

        print(
            "AGENT 4 FINAL STATUS"
        )

        print(
            "=" * 100
        )

        print()

        print(
            "Trade execution       : COMPLETE"
        )

        print(
            "Integration validation: COMPLETE"
        )

        print()

        print(
            "AGENT 4 STATUS: COMPLETE"
        )

    # -------------------------------------------------------------------------
    # Prediction only
    # -------------------------------------------------------------------------

    elif args.mode == "predict":

        run_trade_execution_agent()

    # -------------------------------------------------------------------------
    # Validation only
    # -------------------------------------------------------------------------

    elif args.mode == "validate":

        validate_agent3_agent4()

    # -------------------------------------------------------------------------
    # Paper backtest
    # -------------------------------------------------------------------------

    elif args.mode == "backtest":

        run_evaluation()

    # -------------------------------------------------------------------------
    # Robustness
    # -------------------------------------------------------------------------

    elif args.mode == "robustness":

        run_robustness_evaluation()


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    main()