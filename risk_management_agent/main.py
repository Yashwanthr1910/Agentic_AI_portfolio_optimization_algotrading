"""
Risk Management Agent - Main Entry Point
===============================================================

Runs the complete Agent 3 pipeline:

    Agent 2 predictions
            ↓
    Data preparation
            ↓
    Historical volatility
            ↓
    Value at Risk
            ↓
    Sharpe ratio
            ↓
    Drawdown diagnostics
            ↓
    Combined risk scoring
            ↓
    Agent 2 -> Agent 3 integration validation
            ↓
    Final Agent 3 output

Paper-core Agent 3 metrics
--------------------------
- Historical Volatility
- Value at Risk
- Sharpe Ratio

Additional implementation features
----------------------------------
- Risk contribution decomposition
- Drawdown diagnostics
- Risk scoring
- Risk classification
- Provisional risk-adjusted allocation
"""

from pathlib import Path
import sys


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
RISK_AGENT_ROOT = CURRENT_FILE.parent

if str(RISK_AGENT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(RISK_AGENT_ROOT),
    )


# ============================================================
# IMPORTS
# ============================================================

from src.agent import run_risk_management_agent
from src.integration_validator import run_validation


# ============================================================
# DISPLAY
# ============================================================

def section(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "AGENT 3 - RISK MANAGEMENT PIPELINE"
    )

    print(
        "Starting complete Agent 3 execution..."
    )

    # --------------------------------------------------------
    # Run Agent 3 calculations
    # --------------------------------------------------------

    run_risk_management_agent()

    # --------------------------------------------------------
    # Validate Agent 2 -> Agent 3 integration
    # --------------------------------------------------------

    section(
        "RUNNING FINAL AGENT 2 -> AGENT 3 VALIDATION"
    )

    validation_report = run_validation()

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    section(
        "AGENT 3 FINAL STATUS"
    )

    status = validation_report.get(
        "status",
        "UNKNOWN",
    )

    print(
        f"Integration validation : {status}"
    )

    print()

    if status == "PASS":

        print(
            "AGENT 3 STATUS : COMPLETE"
        )

        print()

        print(
            "Final output is ready for Agent 4."
        )

    else:

        raise RuntimeError(
            "Agent 3 integration validation did not pass."
        )

    return validation_report


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    main()