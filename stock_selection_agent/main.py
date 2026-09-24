"""
main.py

Main entry point for Agent 1:
Stock Selection Agent

Run:
    python main.py
"""

import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


# ============================================================
# ENSURE PROJECT ROOT IS IMPORTABLE
# ============================================================

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORT AGENT
# ============================================================

from src.agent import run_stock_selection_agent


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "AGENTIC AI PORTFOLIO OPTIMIZATION SYSTEM",
        flush=True,
    )

    print(
        "AGENT 1: STOCK SELECTION AGENT",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        f"Project directory: {PROJECT_ROOT}",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    try:

        run_stock_selection_agent()

    except KeyboardInterrupt:

        print(
            "\nExecution cancelled by user.",
            flush=True,
        )

        sys.exit(1)

    except Exception as error:

        print(
            "",
            flush=True,
        )

        print(
            "=" * 72,
            flush=True,
        )

        print(
            "AGENT EXECUTION FAILED",
            flush=True,
        )

        print(
            "=" * 72,
            flush=True,
        )

        print(
            f"Error: {error}",
            flush=True,
        )

        sys.exit(1)

    print(
        "",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "AGENT 1 COMPLETED SUCCESSFULLY",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        "Primary outputs:",
        flush=True,
    )

    print(
        "  data/outputs/selected_stocks.csv",
        flush=True,
    )

    print(
        "  outputs/rankings.csv",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        "Next development stage:",
        flush=True,
    )

    print(
        "  Agent 2 - Trend Prediction Agent",
        flush=True,
    )

    print(
        "",
        flush=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()