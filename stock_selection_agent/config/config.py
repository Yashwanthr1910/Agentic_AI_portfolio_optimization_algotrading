"""
config.py

Central configuration for the Stock Selection Agent.

All major paths, dates, thresholds, model parameters,
portfolio settings and backtesting assumptions should
be defined here instead of being hard-coded across files.
"""

from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DIRECTORY PATHS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

DAILY_DATA_DIR = RAW_DATA_DIR / "daily"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

METADATA_DIR = DATA_DIR / "metadata"

DATA_OUTPUT_DIR = DATA_DIR / "outputs"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_RESULTS_DIR = REPORT_DIR / "results"


# ============================================================
# DATA FILES
# ============================================================

NIFTY500_CONSTITUENTS_FILE = (
    RAW_DATA_DIR
    / "nifty500_constituents.csv"
)

CLEANED_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "nifty500_daily.parquet"
)

FEATURES_FILE = (
    PROCESSED_DATA_DIR
    / "features.parquet"
)

LABELED_FEATURES_FILE = (
    PROCESSED_DATA_DIR
    / "labeled_features.parquet"
)

SECTOR_MAPPING_FILE = (
    METADATA_DIR
    / "sector_mapping.csv"
)


# ============================================================
# STOCK-SELECTION OUTPUT FILES
# ============================================================

DECISION_TREE_PREDICTIONS_FILE = (
    DATA_OUTPUT_DIR
    / "decision_tree_predictions.parquet"
)

BGSTO_SELECTED_CSV = (
    DATA_OUTPUT_DIR
    / "bgsto_selected_stocks.csv"
)

BGSTO_SELECTED_PARQUET = (
    DATA_OUTPUT_DIR
    / "bgsto_selected_stocks.parquet"
)

SELECTED_STOCKS_CSV = (
    DATA_OUTPUT_DIR
    / "selected_stocks.csv"
)

SELECTED_STOCKS_PARQUET = (
    DATA_OUTPUT_DIR
    / "selected_stocks.parquet"
)


# ============================================================
# FINAL OUTPUT FILES
# ============================================================

RANKINGS_CSV = (
    OUTPUT_DIR
    / "rankings.csv"
)

RANKINGS_PARQUET = (
    OUTPUT_DIR
    / "rankings.parquet"
)

PORTFOLIO_CSV = (
    OUTPUT_DIR
    / "portfolio.csv"
)

PORTFOLIO_PARQUET = (
    OUTPUT_DIR
    / "portfolio.parquet"
)

BACKTEST_RESULTS_CSV = (
    OUTPUT_DIR
    / "backtest_results.csv"
)

BACKTEST_RESULTS_PARQUET = (
    OUTPUT_DIR
    / "backtest_results.parquet"
)

PERFORMANCE_CSV = (
    OUTPUT_DIR
    / "performance.csv"
)


# ============================================================
# DATASET PERIOD
# ============================================================

DATA_START_DATE = "2015-01-01"

DATA_END_DATE = "2025-12-31"


# ============================================================
# PAPER-ALIGNED DATA SPLITS
# ============================================================

TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2018-12-31"

BACKTEST_START_DATE = "2019-01-01"
BACKTEST_END_DATE = "2024-12-31"

VALIDATION_START_DATE = "2025-01-01"
VALIDATION_END_DATE = "2025-12-31"


# ============================================================
# LABEL GENERATOR
# ============================================================

# Future prediction horizon
LOOKAHEAD_DAYS = 21

# BUY if future 21-day return exceeds 2%.
#
# This is an implementation assumption.
# The paper does not provide a fully reproducible
# numerical BUY-label threshold.
BUY_THRESHOLD = 0.02


# ============================================================
# DECISION TREE
# ============================================================

DECISION_TREE_RANDOM_STATE = 42

DECISION_TREE_MAX_DEPTH = 8

DECISION_TREE_MIN_SAMPLES_SPLIT = 100

DECISION_TREE_MIN_SAMPLES_LEAF = 50

DECISION_TREE_CLASS_WEIGHT = "balanced"

DECISION_TREE_MODEL_DIR = (
    MODEL_DIR
    / "decision_tree"
)

DECISION_TREE_MODEL_FILE = (
    DECISION_TREE_MODEL_DIR
    / "decision_tree.pkl"
)

DECISION_TREE_FEATURE_FILE = (
    DECISION_TREE_MODEL_DIR
    / "feature_columns.json"
)


# ============================================================
# BGSTO
# ============================================================

BGSTO_RANDOM_STATE = 42

BGSTO_MAX_CANDIDATES = 60

BGSTO_MIN_SELECTED_STOCKS = 10

BGSTO_MAX_SELECTED_STOCKS = 20

BGSTO_POPULATION_SIZE = 50

BGSTO_NUM_ITERATIONS = 100

BGSTO_CROSSOVER_RATE = 0.80

BGSTO_MUTATION_RATE = 0.05


# ============================================================
# CURRENT BGSTO PRACTICAL FITNESS WEIGHTS
# ============================================================
#
# IMPORTANT:
# These belong to our current reproducible implementation.
#
# They should not be described as the exact numerical
# fitness function from the paper.
# ============================================================

BGSTO_BUY_PROBABILITY_WEIGHT = 0.40

BGSTO_RETURN_WEIGHT = 0.25

BGSTO_SHARPE_WEIGHT = 0.20

BGSTO_LOW_VOLATILITY_WEIGHT = 0.15


# ============================================================
# STOCK SCORER
# ============================================================

FINAL_SCORE_MIN = 0.0

FINAL_SCORE_MAX = 100.0


# ============================================================
# PORTFOLIO
# ============================================================

TOP_N_STOCKS = 10

INITIAL_CAPITAL = 1_000_000.0

PORTFOLIO_WEIGHTING_METHOD = "equal"


# ============================================================
# BACKTEST
# ============================================================

TRANSACTION_COST_RATE = 0.0

TRADING_DAYS_PER_YEAR = 252


# ============================================================
# RISK / PERFORMANCE
# ============================================================

RISK_FREE_RATE = 0.0

VAR_CONFIDENCE_LEVEL = 0.95


# ============================================================
# FEATURE SETTINGS
# ============================================================

SMA_WINDOWS = [
    20,
    50,
    100,
    200
]

EMA_SHORT_WINDOW = 12

EMA_LONG_WINDOW = 26

RSI_WINDOW = 14

MACD_SIGNAL_WINDOW = 9

VOLATILITY_SHORT_WINDOW = 20

VOLATILITY_MEDIUM_WINDOW = 60

VOLATILITY_LONG_WINDOW = 252

SHARPE_WINDOW = 252

ROLLING_52W_WINDOW = 252

MAX_DRAWDOWN_WINDOW = 252

AVERAGE_VOLUME_WINDOW = 20


# ============================================================
# FEATURE-SET MODE
# ============================================================
#
# Current project uses "enhanced" because our existing
# feature_engineering.py includes technical indicators such
# as RSI, MACD, SMA and EMA.
#
# Later we can introduce:
#
#     FEATURE_SET_MODE = "paper"
#
# for a stricter paper-compatible experiment.
# ============================================================

FEATURE_SET_MODE = "enhanced"


# ============================================================
# AGENT EXECUTION
# ============================================================

RUN_DATA_LOADER = False

RUN_DATA_CLEANER = False

RUN_FEATURE_ENGINEERING = False

RUN_LABEL_GENERATOR = False

RUN_DECISION_TREE = True

RUN_BGSTO = True

RUN_STOCK_SELECTOR = True

RUN_STOCK_SCORER = True

RUN_PORTFOLIO = True

RUN_BACKTEST = True


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

def create_project_directories():

    directories = [
        RAW_DATA_DIR,
        DAILY_DATA_DIR,
        PROCESSED_DATA_DIR,
        METADATA_DIR,
        DATA_OUTPUT_DIR,
        OUTPUT_DIR,
        MODEL_DIR,
        DECISION_TREE_MODEL_DIR,
        REPORT_DIR,
        REPORT_RESULTS_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

def print_config():

    print(
        "============================================================"
    )

    print(
        "STOCK SELECTION AGENT CONFIGURATION"
    )

    print(
        "============================================================"
    )

    print(
        f"Project root       : {PROJECT_ROOT}"
    )

    print(
        f"Data period        : "
        f"{DATA_START_DATE} -> {DATA_END_DATE}"
    )

    print(
        f"Training period    : "
        f"{TRAIN_START_DATE} -> {TRAIN_END_DATE}"
    )

    print(
        f"Backtest period    : "
        f"{BACKTEST_START_DATE} -> {BACKTEST_END_DATE}"
    )

    print(
        f"Validation period  : "
        f"{VALIDATION_START_DATE} -> {VALIDATION_END_DATE}"
    )

    print(
        f"Lookahead          : {LOOKAHEAD_DAYS} days"
    )

    print(
        f"BUY threshold      : "
        f"{BUY_THRESHOLD * 100:.2f}%"
    )

    print(
        f"Portfolio size     : {TOP_N_STOCKS}"
    )

    print(
        f"Initial capital    : "
        f"{INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Feature mode       : {FEATURE_SET_MODE}"
    )

    print(
        "============================================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    create_project_directories()

    print_config()