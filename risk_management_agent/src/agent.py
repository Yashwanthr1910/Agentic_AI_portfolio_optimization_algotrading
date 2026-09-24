"""
Agent 3 - Risk Management Agent Orchestrator
===============================================================

Purpose
-------
Run the complete Agent 3 pipeline:

    1. Load Agent 2 predictions + historical market data
    2. Calculate historical volatility
    3. Calculate Value at Risk
    4. Calculate Sharpe ratio
    5. Calculate drawdown diagnostics
    6. Calculate combined risk score
    7. Produce final Agent 3 risk assessment

Paper-core Agent 3 metrics
--------------------------
- Historical Volatility
- Value at Risk
- Sharpe Ratio

Implementation enhancements
---------------------------
- Drawdown diagnostics
- Risk contribution decomposition
- Combined normalized scoring
- Risk-level classification
- Provisional risk-adjusted allocation
"""

from pathlib import Path
import sys


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

RISK_AGENT_ROOT = CURRENT_FILE.parents[1]

if str(RISK_AGENT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(RISK_AGENT_ROOT),
    )


# ============================================================
# IMPORT PIPELINE STAGES
# ============================================================

from src.data_loader import run_data_loader
from src.volatility import run_volatility_calculator
from src.var_calculator import run_var_calculator
from src.sharpe_calculator import run_sharpe_calculator
from src.drawdown_calculator import run_drawdown_calculator
from src.risk_scorer import run_risk_scorer


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
# AGENT CLASS
# ============================================================

class RiskManagementAgent:

    """
    Orchestrates the complete Agent 3 risk-management pipeline.
    """

    def __init__(self):

        self.results = {}


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    def run_data_preparation(self):

        section(
            "AGENT 3 - STEP 1/6 - DATA PREPARATION"
        )

        result = run_data_loader()

        self.results[
            "data_loader"
        ] = result

        return result


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    def run_volatility(self):

        section(
            "AGENT 3 - STEP 2/6 - HISTORICAL VOLATILITY"
        )

        result = run_volatility_calculator()

        self.results[
            "volatility"
        ] = result

        return result


    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    def run_var(self):

        section(
            "AGENT 3 - STEP 3/6 - VALUE AT RISK"
        )

        result = run_var_calculator()

        self.results[
            "var"
        ] = result

        return result


    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    def run_sharpe(self):

        section(
            "AGENT 3 - STEP 4/6 - SHARPE RATIO"
        )

        result = run_sharpe_calculator()

        self.results[
            "sharpe"
        ] = result

        return result


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    def run_drawdown(self):

        section(
            "AGENT 3 - STEP 5/6 - DRAWDOWN DIAGNOSTICS"
        )

        result = run_drawdown_calculator()

        self.results[
            "drawdown"
        ] = result

        return result


    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    def run_risk_scoring(self):

        section(
            "AGENT 3 - STEP 6/6 - RISK SCORING"
        )

        result = run_risk_scorer()

        self.results[
            "risk_scoring"
        ] = result

        return result


    # --------------------------------------------------------
    # FULL PIPELINE
    # --------------------------------------------------------

    def run(self):

        section(
            "AGENT 3 - RISK MANAGEMENT AGENT"
        )

        print(
            "Running complete Agent 3 pipeline."
        )

        print()

        print(
            "Paper-core metrics:"
        )

        print(
            "  Historical Volatility"
        )

        print(
            "  Value at Risk"
        )

        print(
            "  Sharpe Ratio"
        )

        print()

        print(
            "Supporting implementation enhancements:"
        )

        print(
            "  Risk contribution decomposition"
        )

        print(
            "  Drawdown diagnostics"
        )

        print(
            "  Combined risk scoring"
        )

        print(
            "  Provisional risk-adjusted allocation"
        )


        self.run_data_preparation()

        self.run_volatility()

        self.run_var()

        self.run_sharpe()

        self.run_drawdown()

        final_result = (
            self.run_risk_scoring()
        )


        section(
            "AGENT 3 PIPELINE COMPLETE"
        )

        print(
            "All Agent 3 calculation stages completed successfully."
        )

        return final_result


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_risk_management_agent():

    agent = RiskManagementAgent()

    return agent.run()


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_risk_management_agent()