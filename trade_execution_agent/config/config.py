"""
Agent 4 - Trade Execution Agent Configuration
===============================================================================

Paper-aligned DQN configuration.

Reference-paper temporal design
-------------------------------
Paper:
    2000-2018 -> historical calculations
    2019-2023 -> investing/backtesting

Available project data:
    2015-2025

Therefore this implementation uses:

    2015-2018 -> DQN historical training
    2019-2023 -> paper-aligned backtest
    2024      -> optional out-of-sample robustness test
    2025      -> final inference

Important
---------
The 2015 start is an implementation limitation caused by the available
historical data. It is not claimed as an exact replication of the paper.
"""

from __future__ import annotations

from pathlib import Path


# =============================================================================
# ROOT DIRECTORIES
# =============================================================================

TRADE_AGENT_ROOT = Path(
    __file__
).resolve().parents[1]

PROJECT_ROOT = (
    TRADE_AGENT_ROOT.parent
)

DATA_DIR = (
    TRADE_AGENT_ROOT
    / "data"
)

INPUT_DATA_DIR = (
    DATA_DIR
    / "inputs"
)

PROCESSED_DATA_DIR = (
    DATA_DIR
    / "processed"
)

PROCESSED_DIR = (
    PROCESSED_DATA_DIR
)

MODEL_DIR = (
    TRADE_AGENT_ROOT
    / "models"
)

OUTPUT_DIR = (
    TRADE_AGENT_ROOT
    / "outputs"
)

REPORT_DIR = (
    TRADE_AGENT_ROOT
    / "reports"
)

REPORTS_DIR = (
    REPORT_DIR
)

TEST_DIR = (
    TRADE_AGENT_ROOT
    / "tests"
)


# =============================================================================
# UPSTREAM AGENT-3 INPUT
# =============================================================================

AGENT3_RISK_PARQUET_PATH = (
    PROJECT_ROOT
    / "risk_management_agent"
    / "outputs"
    / "risk_assessment.parquet"
)

AGENT3_RISK_CSV_PATH = (
    PROJECT_ROOT
    / "risk_management_agent"
    / "outputs"
    / "risk_assessment.csv"
)

# Compatibility alias
AGENT3_RISK_PATH = (
    AGENT3_RISK_PARQUET_PATH
)


# =============================================================================
# HISTORICAL MARKET DATA
# =============================================================================

MARKET_DATA_PARQUET_PATH = (
    PROJECT_ROOT
    / "stock_selection_agent"
    / "data"
    / "processed"
    / "nifty500_daily.parquet"
)

MARKET_DATA_CSV_PATH = (
    PROJECT_ROOT
    / "stock_selection_agent"
    / "data"
    / "processed"
    / "nifty500_daily.csv"
)

# Compatibility alias
MARKET_DATA_PATH = (
    MARKET_DATA_PARQUET_PATH
)


# =============================================================================
# AGENT-4 PROCESSED DATA
# =============================================================================

STATE_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "state_data.parquet"
)

STATE_DATA_CSV_PATH = (
    PROCESSED_DATA_DIR
    / "state_data.csv"
)


DQN_TRAINING_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "dqn_training_data.parquet"
)

DQN_TRAINING_DATA_CSV = (
    PROCESSED_DATA_DIR
    / "dqn_training_data.csv"
)


# Paper-aligned 2019-2023 backtest data
DQN_INPUT_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "dqn_input_data.parquet"
)

DQN_INPUT_DATA_CSV = (
    PROCESSED_DATA_DIR
    / "dqn_input_data.csv"
)


# Optional 2024 robustness evaluation
DQN_ROBUSTNESS_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "dqn_robustness_data.parquet"
)

DQN_ROBUSTNESS_DATA_CSV = (
    PROCESSED_DATA_DIR
    / "dqn_robustness_data.csv"
)


DQN_INFERENCE_DATA_PATH = (
    PROCESSED_DATA_DIR
    / "dqn_inference_data.parquet"
)

DQN_INFERENCE_DATA_CSV = (
    PROCESSED_DATA_DIR
    / "dqn_inference_data.csv"
)


# Compatibility aliases
TRAINING_DATA_PATH = (
    DQN_TRAINING_DATA_PATH
)

EVALUATION_DATA_PATH = (
    DQN_INPUT_DATA_PATH
)

ROBUSTNESS_DATA_PATH = (
    DQN_ROBUSTNESS_DATA_PATH
)


# =============================================================================
# MODEL PATHS
# =============================================================================

DQN_MODEL_PATH = (
    MODEL_DIR
    / "dqn_model.keras"
)

DQN_WEIGHTS_PATH = (
    MODEL_DIR
    / "dqn_weights.weights.h5"
)

TARGET_MODEL_PATH = (
    MODEL_DIR
    / "target_model.keras"
)


# =============================================================================
# OUTPUT PATHS
# =============================================================================

TRADE_DECISIONS_CSV_PATH = (
    OUTPUT_DIR
    / "trade_decisions.csv"
)

TRADE_DECISIONS_PARQUET_PATH = (
    OUTPUT_DIR
    / "trade_decisions.parquet"
)


# =============================================================================
# REPORT PATHS
# =============================================================================

DATA_LOADER_SUMMARY_PATH = (
    REPORT_DIR
    / "data_loader_summary.json"
)

DQN_MODEL_SUMMARY_PATH = (
    REPORT_DIR
    / "dqn_model_summary.json"
)

DQN_AGENT_SUMMARY_PATH = (
    REPORT_DIR
    / "dqn_agent_summary.json"
)

TRAINING_HISTORY_PATH = (
    REPORT_DIR
    / "training_history.csv"
)

TRAINING_SUMMARY_PATH = (
    REPORT_DIR
    / "training_summary.json"
)

EVALUATION_REPORT_PATH = (
    REPORT_DIR
    / "evaluation_report.json"
)

ROBUSTNESS_REPORT_PATH = (
    REPORT_DIR
    / "robustness_report.json"
)

TRADE_SUMMARY_PATH = (
    REPORT_DIR
    / "trade_summary.json"
)

INTEGRATION_VALIDATION_PATH = (
    REPORT_DIR
    / "agent3_agent4_validation.json"
)


# =============================================================================
# PAPER-ALIGNED DQN PARAMETERS
# =============================================================================

LEARNING_RATE = 0.0001

GAMMA = 0.95

BATCH_SIZE = 32

EPISODES = 100

HIDDEN_UNITS = (
    64,
    64,
)

NUM_HIDDEN_LAYERS = 2

HIDDEN_ACTIVATION = "relu"

OUTPUT_ACTIVATION = "linear"

OPTIMIZER_NAME = "adam"

LOSS_FUNCTION = "huber"


# =============================================================================
# ACTION SPACE
# =============================================================================

ACTION_HOLD = 0

ACTION_BUY = 1

ACTION_SELL = 2

ACTION_NAMES = {
    ACTION_HOLD: "HOLD",
    ACTION_BUY: "BUY",
    ACTION_SELL: "SELL",
}

NUM_ACTIONS = 3


# =============================================================================
# STATE DEFINITION
# =============================================================================
#
# The paper states that the DQN uses a seven-node input layer and mentions
# market-price information including previous opening/closing prices,
# high, low and last traded price.
#
# The paper does not provide an unambiguous list of all seven variables.
#
# Therefore these seven causal market features are a transparent
# implementation reconstruction.
# =============================================================================

STATE_FEATURES = [
    "open_return_1d",
    "close_return_1d",
    "high_low_range",
    "gap_return",
    "volume_ratio_20d",
    "return_5d",
    "volatility_20d",
]

STATE_SIZE = len(
    STATE_FEATURES
)


# =============================================================================
# AGENT-3 REQUIRED CONTEXT
# =============================================================================

AGENT3_REQUIRED_COLUMNS = [
    "symbol",
    "prediction_date",
    "top30_probability",
    "ensemble_std",
    "trend_class",
    "agent2_rank",
    "historical_volatility_annual",
    "individual_var_percent",
    "sharpe_ratio",
    "risk_contribution_percent",
    "max_drawdown_percent",
    "paper_core_risk_score",
    "agent3_score",
    "agent3_rank",
    "risk_level",
    "risk_decision",
    "risk_adjusted_weight",
]


# =============================================================================
# PAPER-ALIGNED / AVAILABLE-DATA TEMPORAL SPLIT
# =============================================================================

# Closest available approximation to paper's pre-2019 historical period.
TRAIN_START_DATE = "2015-01-01"

TRAIN_END_DATE = "2018-12-31"


# Paper's investing / backtesting period.
EVALUATION_START_DATE = "2019-01-01"

EVALUATION_END_DATE = "2023-12-31"


# Additional test not claimed as paper-exact.
ROBUSTNESS_START_DATE = "2024-01-01"

ROBUSTNESS_END_DATE = "2024-12-31"


# Current final system inference date.
FINAL_INFERENCE_DATE = "2025-12-01"


# =============================================================================
# FEATURE PARAMETERS
# =============================================================================

VOLUME_LOOKBACK = 20

VOLATILITY_LOOKBACK = 20

RETURN_LOOKBACK = 5


# =============================================================================
# EXPERIENCE REPLAY
# =============================================================================
#
# Exact replay capacity is not specified by the paper.
# =============================================================================

REPLAY_BUFFER_CAPACITY = 50_000

MIN_REPLAY_SIZE = 1_000


# =============================================================================
# EPSILON-GREEDY EXPLORATION
# =============================================================================
#
# Exact epsilon schedule is not specified by the paper.
# =============================================================================

EPSILON_START = 1.0

EPSILON_MIN = 0.05

EPSILON_DECAY = 0.995


# =============================================================================
# TARGET NETWORK
# =============================================================================
#
# Paper specifies evaluation + target networks but does not provide a
# reproducible exact synchronization interval.
# =============================================================================

TARGET_UPDATE_FREQUENCY = 250


# =============================================================================
# TRADING ENVIRONMENT
# =============================================================================

INITIAL_CASH = 1_000_000.0

TRANSACTION_COST_RATE = 0.001

SLIPPAGE_RATE = 0.0

DEFAULT_TRADE_FRACTION = 0.10

MAX_POSITION_FRACTION = 0.25

ALLOW_SHORT_SELLING = False

ALLOW_FRACTIONAL_SHARES = False


# =============================================================================
# REWARD
# =============================================================================
#
# The paper connects reward to returns/losses and maximum drawdown.
#
# Exact step-level formula is not fully specified.
#
# Current implementation:
#
# reward =
#     RETURN_REWARD_WEIGHT * portfolio_return
#     - DRAWDOWN_PENALTY_WEIGHT * abs(current_drawdown)
#
# This must be described as an implementation reconstruction.
# =============================================================================

RETURN_REWARD_WEIGHT = 1.0

DRAWDOWN_PENALTY_WEIGHT = 1.0

REWARD_CLIP_MIN = -1.0

REWARD_CLIP_MAX = 1.0


# =============================================================================
# GENERAL
# =============================================================================

RANDOM_SEED = 42

TERMINATE_ON_DATA_END = True

MAX_STEPS_PER_EPISODE = None

FLOAT_TOLERANCE = 1e-12


# =============================================================================
# FINAL OUTPUT COLUMNS
# =============================================================================

OUTPUT_COLUMNS = [
    "symbol",
    "prediction_date",
    "q_hold",
    "q_buy",
    "q_sell",
    "action_id",
    "trade_action",
    "top30_probability",
    "agent2_rank",
    "agent3_score",
    "agent3_rank",
    "risk_level",
    "risk_decision",
    "risk_adjusted_weight",
]


# =============================================================================
# DIRECTORY CREATION
# =============================================================================

for directory in [
    DATA_DIR,
    INPUT_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    TEST_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

def validate_config() -> None:
    """
    Validate Agent-4 configuration.
    """

    if NUM_ACTIONS != 3:

        raise ValueError(
            "Agent 4 must use exactly three actions: "
            "HOLD, BUY, SELL."
        )

    if STATE_SIZE != len(
        STATE_FEATURES
    ):

        raise ValueError(
            "STATE_SIZE does not match STATE_FEATURES."
        )

    if STATE_SIZE != 7:

        raise ValueError(
            "Paper-aligned implementation expects seven state inputs."
        )

    if len(
        HIDDEN_UNITS
    ) != NUM_HIDDEN_LAYERS:

        raise ValueError(
            "Hidden-layer count mismatch."
        )

    if HIDDEN_UNITS != (
        64,
        64,
    ):

        raise ValueError(
            "Paper-aligned DQN requires hidden layers 64 -> 64."
        )

    if LEARNING_RATE <= 0:

        raise ValueError(
            "LEARNING_RATE must be positive."
        )

    if not (
        0.0
        <=
        GAMMA
        <=
        1.0
    ):

        raise ValueError(
            "GAMMA must be between 0 and 1."
        )

    if BATCH_SIZE <= 0:

        raise ValueError(
            "BATCH_SIZE must be positive."
        )

    if EPISODES != 100:

        raise ValueError(
            "Paper-aligned configuration expects 100 episodes."
        )

    if REPLAY_BUFFER_CAPACITY < BATCH_SIZE:

        raise ValueError(
            "Replay buffer must exceed batch size."
        )

    if MIN_REPLAY_SIZE < BATCH_SIZE:

        raise ValueError(
            "MIN_REPLAY_SIZE must be >= BATCH_SIZE."
        )

    if not (
        0.0
        <=
        EPSILON_MIN
        <=
        EPSILON_START
        <=
        1.0
    ):

        raise ValueError(
            "Invalid epsilon configuration."
        )

    if not (
        0.0
        <
        EPSILON_DECAY
        <=
        1.0
    ):

        raise ValueError(
            "EPSILON_DECAY must be in (0, 1]."
        )

    if INITIAL_CASH <= 0:

        raise ValueError(
            "INITIAL_CASH must be positive."
        )

    if TRANSACTION_COST_RATE < 0:

        raise ValueError(
            "TRANSACTION_COST_RATE cannot be negative."
        )

    if not (
        0.0
        <
        DEFAULT_TRADE_FRACTION
        <=
        1.0
    ):

        raise ValueError(
            "DEFAULT_TRADE_FRACTION must be in (0,1]."
        )

    if not (
        0.0
        <
        MAX_POSITION_FRACTION
        <=
        1.0
    ):

        raise ValueError(
            "MAX_POSITION_FRACTION must be in (0,1]."
        )

    train_start = TRAIN_START_DATE
    train_end = TRAIN_END_DATE

    eval_start = EVALUATION_START_DATE
    eval_end = EVALUATION_END_DATE

    robust_start = ROBUSTNESS_START_DATE
    robust_end = ROBUSTNESS_END_DATE

    if train_end >= eval_start:

        raise ValueError(
            "Training must end before paper backtesting starts."
        )

    if eval_end >= robust_start:

        raise ValueError(
            "Paper backtest must end before robustness period."
        )

    if robust_end >= FINAL_INFERENCE_DATE:

        raise ValueError(
            "Robustness period must end before final inference date."
        )


validate_config()


if __name__ == "__main__":

    print(
        "Agent-4 configuration validation: PASS"
    )

    print(
        f"Historical training : "
        f"{TRAIN_START_DATE} -> {TRAIN_END_DATE}"
    )

    print(
        f"Paper backtest      : "
        f"{EVALUATION_START_DATE} -> {EVALUATION_END_DATE}"
    )

    print(
        f"Robustness          : "
        f"{ROBUSTNESS_START_DATE} -> {ROBUSTNESS_END_DATE}"
    )

    print(
        f"Final inference     : "
        f"{FINAL_INFERENCE_DATE}"
    )