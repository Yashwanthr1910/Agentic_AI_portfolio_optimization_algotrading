"""
config_old.py

OLD CONFIGURATION BACKUP
========================

Central configuration for Agent 2:
Trend Prediction Agent

This file preserves the earlier Agent 2 configuration before the
relative cross-sectional TOP-30% target was introduced.

IMPORTANT
---------
This is a historical backup only.

Do NOT import this file in the new Agent 2 pipeline.
"""

from pathlib import Path


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

TREND_AGENT_ROOT = Path(
    __file__
).resolve().parents[1]

SYSTEM_ROOT = TREND_AGENT_ROOT.parent


# ============================================================
# AGENT 1 LOCATION
# ============================================================

STOCK_SELECTION_AGENT_ROOT = (
    SYSTEM_ROOT
    / "stock_selection_agent"
)


# ============================================================
# AGENT 1 INPUT
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
# HISTORICAL DATA SOURCE
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
# AGENT 2 PROCESSED FILES
# ============================================================

TREND_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "trend_data.parquet"
)

SEQUENCE_FILE = (
    PROCESSED_DATA_DIR
    / "sequences.npz"
)

SCALER_FILE = (
    PROCESSED_DATA_DIR
    / "scaler.pkl"
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

BEST_MODEL_FILE = (
    TREND_MODEL_DIR
    / "best_trend_model.keras"
)

FINAL_MODEL_FILE = (
    TREND_MODEL_DIR
    / "trend_model.keras"
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


# ============================================================
# DATASET DATE RANGE
# ============================================================

DATA_START_DATE = "2015-01-01"

DATA_END_DATE = "2025-12-31"


# ============================================================
# CHRONOLOGICAL SPLITS
# ============================================================

TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2018-12-31"

BACKTEST_START_DATE = "2019-01-01"
BACKTEST_END_DATE = "2024-12-31"

VALIDATION_START_DATE = "2025-01-01"
VALIDATION_END_DATE = "2025-12-31"


# ============================================================
# REQUIRED COLUMNS FROM AGENT 1
# ============================================================

SELECTED_STOCK_REQUIRED_COLUMNS = [

    "symbol",
    "selection_rank",
    "buy_probability",
    "bgsto_stock_score",
]


# ============================================================
# BASE MARKET COLUMNS
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
# OLD TREND FEATURE LIST
# ============================================================

TREND_FEATURE_COLUMNS = [

    "daily_return",

    "return_5d",
    "return_21d",
    "return_63d",

    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",

    "ema_12",
    "ema_26",

    "price_to_sma20",
    "price_to_sma50",
    "price_to_sma200",

    "rsi_14",

    "macd",
    "macd_signal",
    "macd_histogram",

    "volatility_20d",
    "volatility_60d",
    "volatility_252d",

    "sharpe_252d",

    "drawdown",

    "distance_52w_high",
    "distance_52w_low",

    "volume_change",
    "avg_volume_20d",
    "volume_ratio",

    "high_low_range",
    "open_close_return",
    "gap_return",
]


# ============================================================
# LEAKAGE COLUMNS
# ============================================================

LEAKAGE_COLUMNS = [

    "future_price_21d",
    "future_return_21d",

    "target",
    "predicted_target",

    "dataset_split",
]


# ============================================================
# SEQUENCE SETTINGS
# ============================================================

SEQUENCE_LENGTH = 60


# ============================================================
# OLD PREDICTION TARGET
# ============================================================

TARGET_COLUMN = None

PREDICTION_HORIZON = None


# ============================================================
# DILATED LSTM SETTINGS
# ============================================================

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
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 64

EPOCHS = 50

LEARNING_RATE = 0.001

EARLY_STOPPING_PATIENCE = 8

REDUCE_LR_PATIENCE = 4

MIN_LEARNING_RATE = 1e-6


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
            "SEQUENCE_LENGTH must be greater than 0."
        )

    if BATCH_SIZE <= 0:

        raise ValueError(
            "BATCH_SIZE must be greater than 0."
        )

    if EPOCHS <= 0:

        raise ValueError(
            "EPOCHS must be greater than 0."
        )

    if LEARNING_RATE <= 0:

        raise ValueError(
            "LEARNING_RATE must be greater than 0."
        )

    if not (
        0 <= LSTM_DROPOUT < 1
    ):

        raise ValueError(
            "LSTM_DROPOUT must be between 0 and 1."
        )

    if not (
        0 <= TRANSFORMER_DROPOUT < 1
    ):

        raise ValueError(
            "TRANSFORMER_DROPOUT must be between 0 and 1."
        )


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

def print_config():

    print("=" * 70)

    print(
        "TREND PREDICTION AGENT - OLD CONFIGURATION"
    )

    print("=" * 70)

    print(
        f"Trend Agent Root : "
        f"{TREND_AGENT_ROOT}"
    )

    print(
        f"System Root      : "
        f"{SYSTEM_ROOT}"
    )

    print(
        f"Agent 1 Input    : "
        f"{SELECTED_STOCKS_FILE}"
    )

    print(
        f"Feature Dataset  : "
        f"{HISTORICAL_FEATURE_FILE}"
    )

    print(
        f"Sequence Length  : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Training Period  : "
        f"{TRAIN_START_DATE} -> "
        f"{TRAIN_END_DATE}"
    )

    print(
        f"Backtest Period  : "
        f"{BACKTEST_START_DATE} -> "
        f"{BACKTEST_END_DATE}"
    )

    print(
        f"Validation Period: "
        f"{VALIDATION_START_DATE} -> "
        f"{VALIDATION_END_DATE}"
    )

    print(
        f"Number Features  : "
        f"{len(TREND_FEATURE_COLUMNS)}"
    )

    print(
        f"Batch Size       : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs           : "
        f"{EPOCHS}"
    )

    print("=" * 70)


# ============================================================
# INITIALIZATION
# ============================================================

create_project_directories()

validate_config()


if __name__ == "__main__":

    print_config()