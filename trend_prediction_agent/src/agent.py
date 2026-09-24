"""
Agent 2 - Trend Prediction Agent

This module orchestrates the internal Trend Prediction Agent pipeline.

Pipeline
--------
Agent 1 selected stocks
        ↓
data_loader.py
        ↓
data_preprocessor.py
        ↓
sequence_builder.py
        ↓
trainer.py
        ↓
predictor.py
        ↓
trend_predictions.csv

Individual stages remain modular and can still be run separately.
"""

from pathlib import Path
import sys


# ============================================================
# PROJECT PATH SETUP
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
TREND_AGENT_ROOT = CURRENT_FILE.parents[1]

if str(TREND_AGENT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(TREND_AGENT_ROOT),
    )


# ============================================================
# IMPORT AGENT 2 COMPONENTS
# ============================================================

from src.data_loader import run_data_loader
from src.data_preprocessor import run_data_preprocessor
from src.sequence_builder import run_sequence_builder
from src.trainer import train_trend_model
from src.predictor import run_prediction


# ============================================================
# TREND PREDICTION AGENT
# ============================================================

class TrendPredictionAgent:
    """
    Controller for Agent 2 - Trend Prediction Agent.

    Parameters
    ----------
    load_data : bool
        Run the historical data loading stage.

    preprocess : bool
        Run the preprocessing and scaling stage.

    build_sequence_data : bool
        Build LSTM/Transformer training sequences.

    train_model : bool
        Train the Dilated-LSTM + Transformer +
        Progressive Attention model.

    predict : bool
        Generate latest trend predictions.
    """

    def __init__(
        self,
        load_data=True,
        preprocess=True,
        build_sequence_data=True,
        train_model=True,
        predict=True,
    ):
        self.load_data = load_data
        self.preprocess = preprocess
        self.build_sequence_data = build_sequence_data
        self.train_model = train_model
        self.predict = predict


    # ========================================================
    # RUN COMPLETE AGENT
    # ========================================================

    def run(self):
        """
        Execute the configured Agent 2 pipeline.

        Returns
        -------
        pandas.DataFrame or None
            Trend prediction output when prediction is enabled.
        """

        print()
        print("=" * 90)
        print("AGENT 2 - TREND PREDICTION AGENT")
        print("=" * 90)
        print()

        predictions = None


        # ====================================================
        # STAGE 1 - LOAD DATA
        # ====================================================

        if self.load_data:

            print()
            print("=" * 90)
            print(
                "STAGE 1 - LOAD SELECTED STOCK HISTORY"
            )
            print("=" * 90)
            print()

            run_data_loader()

        else:

            print(
                "STAGE 1 skipped - "
                "using existing trend_data.parquet"
            )


        # ====================================================
        # STAGE 2 - PREPROCESS DATA
        # ====================================================

        if self.preprocess:

            print()
            print("=" * 90)
            print(
                "STAGE 2 - PREPROCESS TREND DATA"
            )
            print("=" * 90)
            print()

            run_data_preprocessor()

        else:

            print(
                "STAGE 2 skipped - "
                "using existing "
                "preprocessed_trend_data.parquet"
            )


        # ====================================================
        # STAGE 3 - BUILD SEQUENCES
        # ====================================================

        if self.build_sequence_data:

            print()
            print("=" * 90)
            print(
                "STAGE 3 - BUILD TREND SEQUENCES"
            )
            print("=" * 90)
            print()

            run_sequence_builder()

        else:

            print(
                "STAGE 3 skipped - "
                "using existing sequences.npz"
            )


        # ====================================================
        # STAGE 4 - TRAIN MODEL
        # ====================================================

        if self.train_model:

            print()
            print("=" * 90)
            print(
                "STAGE 4 - TRAIN TREND MODEL"
            )
            print("=" * 90)
            print()

            train_trend_model()

        else:

            print(
                "STAGE 4 skipped - "
                "using existing trained model"
            )


        # ====================================================
        # STAGE 5 - GENERATE PREDICTIONS
        # ====================================================

        if self.predict:

            print()
            print("=" * 90)
            print(
                "STAGE 5 - GENERATE TREND PREDICTIONS"
            )
            print("=" * 90)
            print()

            predictions = run_prediction()

        else:

            print(
                "STAGE 5 skipped - "
                "prediction disabled"
            )


        # ====================================================
        # COMPLETE
        # ====================================================

        print()
        print("=" * 90)
        print(
            "AGENT 2 - TREND PREDICTION COMPLETE"
        )
        print("=" * 90)
        print()

        if predictions is not None:

            print(
                f"Predictions generated for "
                f"{len(predictions)} stocks."
            )

            print()

        return predictions


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    agent = TrendPredictionAgent(
        load_data=True,
        preprocess=True,
        build_sequence_data=True,
        train_model=True,
        predict=True,
    )

    agent.run()