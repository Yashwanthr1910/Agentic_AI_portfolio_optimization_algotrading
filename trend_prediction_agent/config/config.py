"""
config.py

Agent 2 - Trend Prediction Agent
=================================

Central configuration for Agent 2.

CURRENT TARGET
--------------
Agent 2 now predicts relative cross-sectional stock performance.

For every prediction date:

    class 1 = stock belongs to the TOP 30% of eligible stocks
              according to its future 21-trading-day return.

    class 0 = stock does NOT belong to the TOP 30%.

This replaces the previous experimental absolute target:

    future 21-day return >= +2%

IMPORTANT
---------
The relative TOP-30% target is a methodological enhancement selected
from historical 2019-2024 target diagnostics.

It should not be described as an exact numerical target specified by
the research paper.

The model architecture remains paper-inspired:
Dilated LSTM -> Transformer -> Progressive Attention -> Softmax.
"""

from pathlib import Path


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

TREND_AGENT_ROOT = Path(
    __file__
).resolve().parents[1]

SYSTEM_ROOT = (
    TREND_AGENT_ROOT.parent
)


# ============================================================
# AGENT 1 ROOT
# ============================================================

STOCK_SELECTION_AGENT_ROOT = (
    SYSTEM_ROOT
    / "stock_selection_agent"
)


# ============================================================
# AGENT 1 INPUT FILES
# ============================================================

SELECTED_STOCKS_FILE = (
    STOCK_SELECTION_AGENT_ROOT
    / "data"
    / "outputs"
    / "selected_stocks.csv"
)

STOCK_RANKINGS_FILE = (
    STOCK_SELECTION_AGENT_ROOT
    / "outputs"
    / "rankings.csv"
)


# ============================================================
# HISTORICAL DATA SOURCES
# ============================================================

HISTORICAL_PRICE_FILE = (
    STOCK_SELECTION_AGENT_ROOT
    / "data"
    / "processed"
    / "nifty500_daily.parquet"
)

HISTORICAL_FEATURE_FILE = (
    STOCK_SELECTION_AGENT_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)


# ============================================================
# AGENT 2 DATA DIRECTORIES
# ============================================================

DATA_DIR = (
    TREND_AGENT_ROOT
    / "data"
)

INPUT_DIR = (
    DATA_DIR
    / "inputs"
)

RAW_DATA_DIR = (
    DATA_DIR
    / "raw"
)

PROCESSED_DATA_DIR = (
    DATA_DIR
    / "processed"
)


# ============================================================
# AGENT 2 DATA FILES
# ============================================================

TREND_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "trend_data.parquet"
)

TRAINING_UNIVERSE_FILE = (
    PROCESSED_DATA_DIR
    / "training_universe.parquet"
)

INFERENCE_UNIVERSE_FILE = (
    PROCESSED_DATA_DIR
    / "inference_universe.parquet"
)

PREPROCESSED_TREND_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "preprocessed_trend_data.parquet"
)

SCALER_FILE = (
    PROCESSED_DATA_DIR
    / "scaler.pkl"
)


# ============================================================
# MEMORY-EFFICIENT SEQUENCE FILES
# ============================================================

SEQUENCE_INDEX_FILE = (
    PROCESSED_DATA_DIR
    / "sequence_index.parquet"
)

SEQUENCE_METADATA_FILE = (
    PROCESSED_DATA_DIR
    / "sequence_metadata.json"
)

SEQUENCE_SUMMARY_FILE = (
    PROCESSED_DATA_DIR
    / "sequence_summary.csv"
)

SEQUENCE_STOCK_SUMMARY_FILE = (
    PROCESSED_DATA_DIR
    / "sequence_stock_summary.csv"
)


# ------------------------------------------------------------
# Legacy path retained only so older imports do not crash.
#
# The current training pipeline MUST NOT use this dense NPZ.
# ------------------------------------------------------------

SEQUENCE_FILE = (
    PROCESSED_DATA_DIR
    / "sequences.npz"
)


# ============================================================
# MODEL DIRECTORIES
# ============================================================

MODEL_DIR = (
    TREND_AGENT_ROOT
    / "models"
)

TREND_MODEL_DIR = (
    MODEL_DIR
    / "trend_model"
)


# ============================================================
# MODEL FILES
# ============================================================

BEST_MODEL_FILE = (
    TREND_MODEL_DIR
    / "best_trend_model.keras"
)

FINAL_MODEL_FILE = (
    TREND_MODEL_DIR
    / "trend_model.keras"
)

BEST_MODEL_WEIGHTS_FILE = (
    TREND_MODEL_DIR
    / "best_trend_model.weights.h5"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = (
    TREND_AGENT_ROOT
    / "outputs"
)

REPORT_DIR = (
    TREND_AGENT_ROOT
    / "reports"
)

RESULTS_DIR = (
    REPORT_DIR
    / "results"
)


# ============================================================
# OUTPUT FILES
# ============================================================

TREND_PREDICTIONS_CSV = (
    OUTPUT_DIR
    / "trend_predictions.csv"
)

TREND_PREDICTIONS_PARQUET = (
    OUTPUT_DIR
    / "trend_predictions.parquet"
)

TRAINING_HISTORY_FILE = (
    REPORT_DIR
    / "training_history.csv"
)

TREND_METRICS_FILE = (
    REPORT_DIR
    / "trend_metrics.csv"
)

TRAINING_SUMMARY_FILE = (
    REPORT_DIR
    / "training_summary.json"
)

TREND_EVALUATION_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "trend_evaluation_predictions.parquet"
)


# ============================================================
# DATASET DATE RANGE
# ============================================================

DATA_START_DATE = "2015-01-01"

DATA_END_DATE = "2025-12-31"


# ============================================================
# CHRONOLOGICAL DATASET SPLITS
# ============================================================

TRAIN_START_DATE = "2015-01-01"

TRAIN_END_DATE = "2018-12-31"


BACKTEST_START_DATE = "2019-01-01"

BACKTEST_END_DATE = "2024-12-31"


VALIDATION_START_DATE = "2025-01-01"

VALIDATION_END_DATE = "2025-12-31"


# ============================================================
# TRAINER INTERNAL VALIDATION
# ============================================================

# Exact cutoff used in the previous completed deep-model run.
#
# Keeping this unchanged provides a clean target A/B comparison.

INTERNAL_VALIDATION_CUTOFF = "2018-07-04"


# ============================================================
# REQUIRED AGENT 1 COLUMNS
# ============================================================

SELECTED_STOCK_REQUIRED_COLUMNS = [

    "symbol",

    "selection_rank",

    "buy_probability",

    "bgsto_stock_score",
]


# ============================================================
# MARKET COLUMNS
# ============================================================

MARKET_COLUMNS = [

    "date",

    "symbol",

    "open",

    "high",

    "low",

    "close",

    "adj_close",

    "volume",
]


# ============================================================
# FINAL AGENT 2 FEATURE SET
# ============================================================

# IMPORTANT:
#
# This is the exact 30-column feature order used by the scaler,
# sequence builder, trainer and future inference pipeline.

TREND_FEATURE_COLUMNS = [

    # Returns
    "return_1d",
    "return_5d",
    "return_21d",
    "return_63d",

    # Moving averages
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_200",

    # Exponential averages
    "ema_12",
    "ema_26",

    # Relative trend
    "close_to_sma_5",
    "close_to_sma_20",
    "close_to_sma_50",
    "close_to_sma_200",

    "sma_5_to_sma_20",
    "sma_20_to_sma_50",

    # Momentum
    "rsi_14",

    "macd",
    "macd_signal",
    "macd_hist",

    # Volatility
    "volatility_20d",
    "volatility_60d",

    # Risk adjusted return
    "sharpe_60d",
    "sharpe_252d",

    # Drawdown
    "drawdown",

    # Volume
    "volume_change_1d",
    "volume_ratio_20d",

    # Price behaviour
    "intraday_range",
    "gap_return",
]


FEATURE_COUNT = len(
    TREND_FEATURE_COLUMNS
)


# ============================================================
# LEAKAGE COLUMNS
# ============================================================

LEAKAGE_COLUMNS = [

    "future_price",

    "future_close",

    "future_price_21d",

    "future_return",

    "future_return_21d",

    "target",

    "target_date",

    "target_row",

    "cross_section_percentile",

    "predicted_target",

    "high_return_probability",
]


# ============================================================
# SEQUENCE SETTINGS
# ============================================================

SEQUENCE_LENGTH = 60


# ============================================================
# TARGET CONFIGURATION
# ============================================================

PREDICTION_HORIZON = 21


# Current target definition

TARGET_MODE = (
    "relative_top_fraction"
)


RELATIVE_TOP_FRACTION = 0.30


RELATIVE_PERCENTILE_CUTOFF = (
    1.0
    -
    RELATIVE_TOP_FRACTION
)


TARGET_COLUMN = "target"


TARGET_CLASS_NAMES = {

    0: "NOT_TOP30",

    1: "TOP30",
}


# ============================================================
# LEGACY ABSOLUTE TARGET
# ============================================================

# Historical only.
#
# This was the previous experimental target:
#
# 21-day future return >= +2%
#
# It must NOT be used by the new sequence builder.

HIGH_RETURN_THRESHOLD = 0.02


# ============================================================
# DILATED LSTM SETTINGS
# ============================================================

# Implementation defaults.
# Not claimed to be exact paper hyperparameters.

LSTM_UNITS = 64

LSTM_DROPOUT = 0.20

LSTM_RECURRENT_DROPOUT = 0.0


DILATION_RATES = [

    1,

    2,

    4,
]


# ============================================================
# TRANSFORMER SETTINGS
# ============================================================

TRANSFORMER_NUM_HEADS = 4

TRANSFORMER_KEY_DIM = 32

TRANSFORMER_FF_DIM = 128

TRANSFORMER_DROPOUT = 0.20

TRANSFORMER_BLOCKS = 2


# ============================================================
# PROGRESSIVE ATTENTION SETTINGS
# ============================================================

ATTENTION_HIDDEN_UNITS = 64

ATTENTION_DROPOUT = 0.10


# ============================================================
# OUTPUT CLASS SETTINGS
# ============================================================

NUM_CLASSES = 2

MODEL_OUTPUT_UNITS = 2


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 64

EPOCHS = 50

LEARNING_RATE = 0.001

EARLY_STOPPING_PATIENCE = 8

REDUCE_LR_PATIENCE = 4

REDUCE_LR_FACTOR = 0.50

MIN_LEARNING_RATE = 1e-6


# ============================================================
# CLASSIFICATION SETTINGS
# ============================================================

CLASSIFICATION_THRESHOLD = 0.50

USE_CLASS_WEIGHTS = True


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


# ============================================================
# SCALING
# ============================================================

SCALER_TYPE = "standard"


# ============================================================
# MODEL EXECUTION
# ============================================================

TRAIN_MODEL = True

LOAD_EXISTING_MODEL = False

GENERATE_PREDICTIONS = True


# ============================================================
# DIRECTORY CREATION
# ============================================================

def create_project_directories():

    directories = [

        DATA_DIR,

        INPUT_DIR,

        RAW_DATA_DIR,

        PROCESSED_DATA_DIR,

        MODEL_DIR,

        TREND_MODEL_DIR,

        OUTPUT_DIR,

        REPORT_DIR,

        RESULTS_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config():

    if SEQUENCE_LENGTH <= 0:

        raise ValueError(
            "SEQUENCE_LENGTH must be > 0."
        )

    if PREDICTION_HORIZON <= 0:

        raise ValueError(
            "PREDICTION_HORIZON must be > 0."
        )

    if TARGET_MODE not in {

        "relative_top_fraction",

        "absolute_return",

    }:

        raise ValueError(
            f"Unsupported TARGET_MODE: "
            f"{TARGET_MODE}"
        )

    if TARGET_MODE == "relative_top_fraction":

        if not (
            0
            <
            RELATIVE_TOP_FRACTION
            <
            1
        ):

            raise ValueError(
                "RELATIVE_TOP_FRACTION "
                "must be between 0 and 1."
            )

    if FEATURE_COUNT != 30:

        raise ValueError(
            "Agent 2 currently expects "
            "exactly 30 features. "
            f"Found {FEATURE_COUNT}."
        )

    if BATCH_SIZE <= 0:

        raise ValueError(
            "BATCH_SIZE must be > 0."
        )

    if EPOCHS <= 0:

        raise ValueError(
            "EPOCHS must be > 0."
        )

    if LEARNING_RATE <= 0:

        raise ValueError(
            "LEARNING_RATE must be > 0."
        )

    if not (
        0
        <=
        LSTM_DROPOUT
        <
        1
    ):

        raise ValueError(
            "Invalid LSTM_DROPOUT."
        )

    if not (
        0
        <=
        TRANSFORMER_DROPOUT
        <
        1
    ):

        raise ValueError(
            "Invalid TRANSFORMER_DROPOUT."
        )

    if NUM_CLASSES != 2:

        raise ValueError(
            "Agent 2 currently expects "
            "binary classification."
        )


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

def print_config():

    print()

    print("=" * 75)

    print(
        "AGENT 2 - TREND PREDICTION CONFIGURATION"
    )

    print("=" * 75)

    print()

    print(
        f"Trend Agent Root     : "
        f"{TREND_AGENT_ROOT}"
    )

    print(
        f"System Root          : "
        f"{SYSTEM_ROOT}"
    )

    print()

    print(
        f"Training data        : "
        f"{PREPROCESSED_TREND_DATA_FILE}"
    )

    print(
        f"Sequence index       : "
        f"{SEQUENCE_INDEX_FILE}"
    )

    print(
        f"Scaler               : "
        f"{SCALER_FILE}"
    )

    print()

    print(
        f"Sequence length      : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Prediction horizon   : "
        f"{PREDICTION_HORIZON}"
    )

    print(
        f"Feature count        : "
        f"{FEATURE_COUNT}"
    )

    print()

    print(
        f"Target mode          : "
        f"{TARGET_MODE}"
    )

    print(
        f"Relative top fraction: "
        f"{RELATIVE_TOP_FRACTION:.0%}"
    )

    print(
        f"Percentile cutoff    : "
        f"{RELATIVE_PERCENTILE_CUTOFF:.2f}"
    )

    print(
        f"Class 0              : "
        f"{TARGET_CLASS_NAMES[0]}"
    )

    print(
        f"Class 1              : "
        f"{TARGET_CLASS_NAMES[1]}"
    )

    print()

    print(
        f"Train                : "
        f"{TRAIN_START_DATE} -> "
        f"{TRAIN_END_DATE}"
    )

    print(
        f"Internal cutoff      : "
        f"{INTERNAL_VALIDATION_CUTOFF}"
    )

    print(
        f"Backtest             : "
        f"{BACKTEST_START_DATE} -> "
        f"{BACKTEST_END_DATE}"
    )

    print(
        f"Final validation     : "
        f"{VALIDATION_START_DATE} -> "
        f"{VALIDATION_END_DATE}"
    )

    print()

    print(
        f"Batch size           : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Maximum epochs       : "
        f"{EPOCHS}"
    )

    print(
        f"Learning rate        : "
        f"{LEARNING_RATE}"
    )

    print()

    print("=" * 75)


# ============================================================
# INITIALIZATION
# ============================================================

create_project_directories()

validate_config()


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print_config()