"""
Agent 2 - Trend Prediction
Final Main Entry Point

Pipeline
--------
1. Run the locked 3-seed ensemble predictor.
2. Validate Agent 1 -> Agent 2 handoff.
3. Save final Agent 2 outputs.

This script DOES NOT:
- retrain models
- refit scaler
- rebuild historical training data
- rebuild historical sequences
- run target diagnostics
- run architecture ablation
- run seed stability
"""

from pathlib import Path
import sys
import time


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
TREND_AGENT_ROOT = CURRENT_FILE.parent

if str(TREND_AGENT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(TREND_AGENT_ROOT),
    )


# ============================================================
# FINAL AGENT 2 COMPONENTS
# ============================================================

from src.predictor import run_predictor
from src.integration_validator import run_validation


# ============================================================
# DISPLAY
# ============================================================

def section(title):

    print()
    print("=" * 95)
    print(title)
    print("=" * 95)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    section(
        "AGENT 2 - FINAL TREND PREDICTION PIPELINE"
    )

    print("Final configuration:")
    print()
    print("Architecture : Dilated LSTM -> Transformer -> Progressive Attention -> Softmax")
    print("Target       : relative_21d_top30")
    print("Sequence     : 60 trading observations")
    print("Features     : 30")
    print("Ensemble     : Seeds [42, 123, 456]")
    print()

    print("Mode:")
    print("FINAL INFERENCE")
    print()

    print("Training      : DISABLED")
    print("Scaler fitting: DISABLED")
    print("Model tuning  : DISABLED")

    # ========================================================
    # STAGE 1
    # ========================================================

    section(
        "STAGE 1/2 - FINAL AGENT 2 PREDICTION"
    )

    run_predictor()

    # ========================================================
    # STAGE 2
    # ========================================================

    section(
        "STAGE 2/2 - AGENT 1 -> AGENT 2 VALIDATION"
    )

    run_validation()

    # ========================================================
    # COMPLETE
    # ========================================================

    elapsed = (
        time.time()
        -
        start_time
    )

    section(
        "AGENT 2 PIPELINE COMPLETE"
    )

    print(
        f"Runtime : {elapsed:.2f} seconds"
    )

    print()
    print("AGENT 2 STATUS:")
    print("COMPLETE")

    print()
    print("Final outputs:")
    print(
        TREND_AGENT_ROOT
        / "outputs"
        / "trend_predictions.csv"
    )
    print(
        TREND_AGENT_ROOT
        / "outputs"
        / "trend_predictions.parquet"
    )

    print()
    print("NEXT:")
    print("AGENT 3 - RISK MANAGEMENT")


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    main()