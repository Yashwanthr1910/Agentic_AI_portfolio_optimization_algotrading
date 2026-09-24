"""
Agent 5 - Portfolio Rebalancing Agent Configuration
===============================================================================

PROJECT:
    Agentic AI Portfolio Optimization and Algorithmic Trading

AGENT:
    Agent 5 - Portfolio Rebalancing Agent

CORE METHOD:
    Markowitz Modern Portfolio Theory (MPT)
    +
    Reinforcement-Learning / DQN feedback from Agent 4

PIPELINE:
    Agent 4
    BUY / SELL / HOLD
            ↓
    Historical returns
            ↓
    Mean returns
            ↓
    Standard deviation
            ↓
    Covariance / correlation
            ↓
    Markowitz MPT
            ↓
    RL-aware allocation
            ↓
    Final rebalanced portfolio


REFERENCE-PAPER ALIGNMENT
-------------------------
The reference paper performs portfolio rebalancing after DQN trade
execution and uses Markowitz Modern Portfolio Theory with reinforcement
learning.

The MPT stage uses:
    - Mean / expected return
    - Standard deviation
    - Covariance
    - Correlation
    - Asset weights
    - Portfolio return
    - Portfolio variance / risk
    - Target return

The paper describes the target return as an average return around the
minimum-risk portfolio solution.


IMPLEMENTATION RECONSTRUCTION
-----------------------------
The paper does not completely specify:
    - exact historical estimation window
    - exact numerical optimizer
    - exact portfolio weight limits
    - exact DQN BUY/HOLD/SELL-to-weight transformation

Therefore these details are explicitly configured and documented here.

Current RL feedback interpretation:

    SELL -> target portfolio weight = 0

    BUY  -> eligible for MPT optimization

    HOLD -> eligible for MPT optimization

No arbitrary BUY multiplier or HOLD multiplier is used.
"""

from __future__ import annotations

from pathlib import Path


# =============================================================================
# ROOT DIRECTORIES
# =============================================================================

AGENT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PROJECT_ROOT = (
    AGENT_ROOT.parent
)


# =============================================================================
# AGENT 5 DIRECTORY PATHS
# =============================================================================

DATA_DIR = (
    AGENT_ROOT
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

MODEL_DIR = (
    AGENT_ROOT
    / "models"
)

OUTPUT_DIR = (
    AGENT_ROOT
    / "outputs"
)

REPORT_DIR = (
    AGENT_ROOT
    / "reports"
)

TEST_DIR = (
    AGENT_ROOT
    / "tests"
)


# =============================================================================
# AGENT 4 INPUT
# =============================================================================
#
# Main upstream input for Agent 5.
#
# Agent 4 contains:
#
#   symbol
#   prediction_date
#   q_hold
#   q_buy
#   q_sell
#   action_id
#   trade_action
#
# and preserved Agent-2 / Agent-3 context.
# =============================================================================

AGENT4_OUTPUT_DIR = (
    PROJECT_ROOT
    / "trade_execution_agent"
    / "outputs"
)

AGENT4_TRADE_DECISIONS_PARQUET = (
    AGENT4_OUTPUT_DIR
    / "trade_decisions.parquet"
)

AGENT4_TRADE_DECISIONS_CSV = (
    AGENT4_OUTPUT_DIR
    / "trade_decisions.csv"
)


# Compatibility alias
AGENT4_TRADE_DECISIONS_PATH = (
    AGENT4_TRADE_DECISIONS_PARQUET
)


# =============================================================================
# AGENT 3 INPUT
# =============================================================================
#
# Agent 3 provides risk-management context:
#
#   historical volatility
#   VaR
#   Sharpe ratio
#   maximum drawdown
#   Agent-3 score
#   Agent-3 rank
#   risk level
#   risk decision
#   risk-adjusted weight
#
# =============================================================================

AGENT3_OUTPUT_DIR = (
    PROJECT_ROOT
    / "risk_management_agent"
    / "outputs"
)

AGENT3_RISK_PARQUET = (
    AGENT3_OUTPUT_DIR
    / "risk_assessment.parquet"
)

AGENT3_RISK_CSV = (
    AGENT3_OUTPUT_DIR
    / "risk_assessment.csv"
)


# Compatibility aliases
AGENT3_RISK_PATH = (
    AGENT3_RISK_PARQUET
)

AGENT3_RISK_ASSESSMENT_PARQUET = (
    AGENT3_RISK_PARQUET
)

AGENT3_RISK_ASSESSMENT_CSV = (
    AGENT3_RISK_CSV
)


# =============================================================================
# HISTORICAL MARKET DATA
# =============================================================================
#
# Historical NIFTY-500 daily price data produced by Agent 1.
#
# Used by Agent 5 to calculate:
#
#   daily returns
#   expected returns
#   standard deviation
#   covariance
#   correlation
#
# =============================================================================

MARKET_DATA_DIR = (
    PROJECT_ROOT
    / "stock_selection_agent"
    / "data"
    / "processed"
)

MARKET_DATA_PARQUET = (
    MARKET_DATA_DIR
    / "nifty500_daily.parquet"
)

MARKET_DATA_CSV = (
    MARKET_DATA_DIR
    / "nifty500_daily.csv"
)


# Compatibility aliases
MARKET_DATA_PATH = (
    MARKET_DATA_PARQUET
)

HISTORICAL_MARKET_PARQUET = (
    MARKET_DATA_PARQUET
)

HISTORICAL_MARKET_CSV = (
    MARKET_DATA_CSV
)


# =============================================================================
# PROCESSED AGENT-5 DATA
# =============================================================================

RETURNS_MATRIX_PARQUET = (
    PROCESSED_DATA_DIR
    / "returns_matrix.parquet"
)

RETURNS_MATRIX_CSV = (
    PROCESSED_DATA_DIR
    / "returns_matrix.csv"
)


COVARIANCE_MATRIX_CSV = (
    PROCESSED_DATA_DIR
    / "covariance_matrix.csv"
)


CORRELATION_MATRIX_CSV = (
    PROCESSED_DATA_DIR
    / "correlation_matrix.csv"
)


ASSET_STATISTICS_PARQUET = (
    PROCESSED_DATA_DIR
    / "asset_statistics.parquet"
)

ASSET_STATISTICS_CSV = (
    PROCESSED_DATA_DIR
    / "asset_statistics.csv"
)


REBALANCING_INPUT_PARQUET = (
    PROCESSED_DATA_DIR
    / "rebalancing_input.parquet"
)

REBALANCING_INPUT_CSV = (
    PROCESSED_DATA_DIR
    / "rebalancing_input.csv"
)


MPT_SOLUTION_CSV = (
    PROCESSED_DATA_DIR
    / "mpt_solution.csv"
)


RL_ADJUSTED_WEIGHTS_CSV = (
    PROCESSED_DATA_DIR
    / "rl_adjusted_weights.csv"
)


# =============================================================================
# FINAL OUTPUT FILES
# =============================================================================

REBALANCED_PORTFOLIO_PARQUET = (
    OUTPUT_DIR
    / "rebalanced_portfolio.parquet"
)

REBALANCED_PORTFOLIO_CSV = (
    OUTPUT_DIR
    / "rebalanced_portfolio.csv"
)

FINAL_PORTFOLIO_ALLOCATION_CSV = (
    OUTPUT_DIR
    / "final_portfolio_allocation.csv"
)


# =============================================================================
# REPORT FILES
# =============================================================================

DATA_LOADER_SUMMARY = (
    REPORT_DIR
    / "data_loader_summary.json"
)

MPT_OPTIMIZATION_SUMMARY = (
    REPORT_DIR
    / "mpt_optimization_summary.json"
)

RL_FEEDBACK_SUMMARY = (
    REPORT_DIR
    / "rl_feedback_summary.json"
)

PORTFOLIO_METRICS_REPORT = (
    REPORT_DIR
    / "portfolio_metrics.json"
)

EVALUATION_SUMMARY = (
    REPORT_DIR
    / "evaluation_summary.json"
)

AGENT4_AGENT5_VALIDATION = (
    REPORT_DIR
    / "agent4_agent5_validation.json"
)

AGENT5_SUMMARY = (
    REPORT_DIR
    / "agent5_summary.json"
)


# =============================================================================
# FINAL REBALANCING DATE
# =============================================================================

FINAL_REBALANCE_DATE = "2025-12-01"


# =============================================================================
# RETURN ESTIMATION
# =============================================================================
#
# Standard annualization assumption for daily equity data.
# =============================================================================

TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# ESTIMATION WINDOW
# =============================================================================
#
# The base paper reports a holding period of approximately five months.
#
# Approximate conversion:
#
#       21 trading days/month
#       ×
#       5 months
#       =
#       105 trading days
#
# This is an explicit implementation choice.
# =============================================================================

ESTIMATION_LOOKBACK_DAYS = 105


# Minimum number of common return observations required before covariance
# and correlation matrices are accepted.
MIN_REQUIRED_RETURN_OBSERVATIONS = 60


# =============================================================================
# MPT OBJECTIVE
# =============================================================================
#
# Portfolio optimization goal:
#
#       minimize portfolio variance
#
# while satisfying:
#
#       sum(weights) = 1
#
# and later:
#
#       portfolio expected return approximately equals target return
#
# =============================================================================

MPT_OBJECTIVE = "minimum_variance"


# =============================================================================
# TARGET RETURN
# =============================================================================
#
# The paper describes target return as average return.
#
# We therefore use the average expected return of MPT-eligible assets as
# the target-return reference.
# =============================================================================

TARGET_RETURN_MODE = "average_asset_mean"


# =============================================================================
# PORTFOLIO WEIGHT CONSTRAINTS
# =============================================================================

ALLOW_SHORT_SELLING = False


# Minimum position weight
MIN_WEIGHT = 0.0


# The paper does not report a reproducible maximum stock allocation.
#
# Using 1.0 avoids inventing an unsupported concentration limit.
#
# A diversification limit can be added later as an experimental
# enhancement, but it should then be explicitly documented.
MAX_WEIGHT = 1.0


# Fully invested portfolio.
WEIGHT_SUM_TARGET = 1.0


# =============================================================================
# AGENT 4 / DQN ACTIONS
# =============================================================================

ACTION_HOLD = "HOLD"

ACTION_BUY = "BUY"

ACTION_SELL = "SELL"


VALID_ACTIONS = {
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
}


# =============================================================================
# RL FEEDBACK INTO MPT
# =============================================================================
#
# IMPORTANT:
#
# The paper does not provide an exact mathematical equation that converts
# DQN BUY / SELL / HOLD decisions into Markowitz portfolio constraints.
#
# We therefore use a transparent and defensible reconstruction:
#
#
#       SELL
#          ↓
#       weight = 0
#
#
#       BUY
#          ↓
#       eligible for MPT
#
#
#       HOLD
#          ↓
#       eligible for MPT
#
#
# No artificial BUY multiplier is used.
#
# No artificial HOLD multiplier is used.
#
# This allows MPT to determine the allocation among eligible assets based
# on expected return and covariance.
# =============================================================================

USE_SELL_AS_ZERO_WEIGHT_CONSTRAINT = True

SELL_TARGET_WEIGHT = 0.0


MPT_ELIGIBLE_ACTIONS = {
    ACTION_BUY,
    ACTION_HOLD,
}


# =============================================================================
# NUMERICAL OPTIMIZATION
# =============================================================================
#
# scipy.optimize.minimize with SLSQP is suitable for:
#
#   equality constraints
#   inequality constraints
#   weight bounds
#
# =============================================================================

OPTIMIZER_METHOD = "SLSQP"

OPTIMIZER_MAX_ITERATIONS = 10_000

OPTIMIZER_FTOL = 1e-12


# =============================================================================
# COVARIANCE STABILIZATION
# =============================================================================
#
# A tiny value is added to the diagonal to improve numerical stability.
#
# This value is deliberately extremely small so that it does not materially
# alter the covariance structure.
# =============================================================================

COVARIANCE_RIDGE = 1e-8


# =============================================================================
# RISK-FREE RATE
# =============================================================================
#
# Currently zero to remain consistent with previous project modules unless
# another explicit risk-free-rate assumption is later introduced.
# =============================================================================

RISK_FREE_RATE = 0.0


# =============================================================================
# NUMERICAL TOLERANCES
# =============================================================================

FLOAT_TOLERANCE = 1e-8

WEIGHT_SUM_TOLERANCE = 1e-6

TARGET_RETURN_TOLERANCE = 1e-6


# =============================================================================
# PORTFOLIO METRIC SETTINGS
# =============================================================================

ANNUALIZATION_FACTOR = TRADING_DAYS_PER_YEAR

VOLATILITY_ANNUALIZATION_FACTOR = (
    TRADING_DAYS_PER_YEAR ** 0.5
)


# =============================================================================
# REQUIRED AGENT 4 COLUMNS
# =============================================================================

AGENT4_REQUIRED_COLUMNS = [
    "symbol",
    "prediction_date",
    "q_hold",
    "q_buy",
    "q_sell",
    "action_id",
    "trade_action",
]


# =============================================================================
# OPTIONAL AGENT 4 CONTEXT
# =============================================================================

AGENT4_CONTEXT_COLUMNS = [
    "market_state_date",
    "selected_q_value",
    "top30_probability",
    "agent2_rank",
    "agent3_score",
    "agent3_rank",
    "risk_level",
    "risk_decision",
    "risk_adjusted_weight",
]


# =============================================================================
# AGENT 3 CONTEXT COLUMNS
# =============================================================================

AGENT3_CONTEXT_COLUMNS = [
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
# FINAL EXPECTED AGENT 5 OUTPUT COLUMNS
# =============================================================================

FINAL_PORTFOLIO_COLUMNS = [
    "symbol",
    "prediction_date",
    "trade_action",
    "mpt_weight",
    "rl_adjusted_weight",
    "final_weight",
    "annualized_expected_return",
    "annualized_volatility",
    "agent3_score",
    "agent3_rank",
    "risk_level",
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
    Validate Agent-5 configuration before any optimization is performed.
    """

    # -------------------------------------------------------------------------
    # Trading calendar
    # -------------------------------------------------------------------------

    if TRADING_DAYS_PER_YEAR <= 0:

        raise ValueError(
            "TRADING_DAYS_PER_YEAR must be positive."
        )

    # -------------------------------------------------------------------------
    # Historical estimation
    # -------------------------------------------------------------------------

    if ESTIMATION_LOOKBACK_DAYS <= 0:

        raise ValueError(
            "ESTIMATION_LOOKBACK_DAYS must be positive."
        )

    if MIN_REQUIRED_RETURN_OBSERVATIONS <= 0:

        raise ValueError(
            "MIN_REQUIRED_RETURN_OBSERVATIONS must be positive."
        )

    if (
        MIN_REQUIRED_RETURN_OBSERVATIONS
        >
        ESTIMATION_LOOKBACK_DAYS
    ):

        raise ValueError(
            "MIN_REQUIRED_RETURN_OBSERVATIONS cannot exceed "
            "ESTIMATION_LOOKBACK_DAYS."
        )

    # -------------------------------------------------------------------------
    # MPT methodology
    # -------------------------------------------------------------------------

    if MPT_OBJECTIVE != "minimum_variance":

        raise ValueError(
            "Current Agent-5 implementation expects "
            "MPT_OBJECTIVE='minimum_variance'."
        )

    if (
        TARGET_RETURN_MODE
        !=
        "average_asset_mean"
    ):

        raise ValueError(
            "Paper-aligned target-return configuration requires "
            "TARGET_RETURN_MODE='average_asset_mean'."
        )

    # -------------------------------------------------------------------------
    # Weight constraints
    # -------------------------------------------------------------------------

    if not ALLOW_SHORT_SELLING:

        if MIN_WEIGHT < 0:

            raise ValueError(
                "MIN_WEIGHT cannot be negative when short selling "
                "is disabled."
            )

    if MAX_WEIGHT <= MIN_WEIGHT:

        raise ValueError(
            "MAX_WEIGHT must be greater than MIN_WEIGHT."
        )

    if not (
        0.0
        <
        WEIGHT_SUM_TARGET
        <=
        1.0
    ):

        raise ValueError(
            "WEIGHT_SUM_TARGET must be within (0, 1]."
        )

    # -------------------------------------------------------------------------
    # Agent-4 action configuration
    # -------------------------------------------------------------------------

    if len(
        VALID_ACTIONS
    ) != 3:

        raise ValueError(
            "Agent 5 expects exactly three Agent-4 actions."
        )

    if not (
        MPT_ELIGIBLE_ACTIONS
        <=
        VALID_ACTIONS
    ):

        raise ValueError(
            "MPT_ELIGIBLE_ACTIONS contains an unsupported action."
        )

    if USE_SELL_AS_ZERO_WEIGHT_CONSTRAINT:

        if (
            ACTION_SELL
            in
            MPT_ELIGIBLE_ACTIONS
        ):

            raise ValueError(
                "SELL cannot be MPT eligible while SELL uses "
                "zero target allocation."
            )

        if abs(
            SELL_TARGET_WEIGHT
        ) > FLOAT_TOLERANCE:

            raise ValueError(
                "SELL_TARGET_WEIGHT must equal zero."
            )

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    if OPTIMIZER_METHOD.upper() != "SLSQP":

        raise ValueError(
            "Current implementation expects SLSQP."
        )

    if OPTIMIZER_MAX_ITERATIONS <= 0:

        raise ValueError(
            "OPTIMIZER_MAX_ITERATIONS must be positive."
        )

    if OPTIMIZER_FTOL <= 0:

        raise ValueError(
            "OPTIMIZER_FTOL must be positive."
        )

    # -------------------------------------------------------------------------
    # Covariance
    # -------------------------------------------------------------------------

    if COVARIANCE_RIDGE < 0:

        raise ValueError(
            "COVARIANCE_RIDGE cannot be negative."
        )

    # -------------------------------------------------------------------------
    # Numerical tolerance
    # -------------------------------------------------------------------------

    if FLOAT_TOLERANCE <= 0:

        raise ValueError(
            "FLOAT_TOLERANCE must be positive."
        )

    if WEIGHT_SUM_TOLERANCE <= 0:

        raise ValueError(
            "WEIGHT_SUM_TOLERANCE must be positive."
        )

    if TARGET_RETURN_TOLERANCE <= 0:

        raise ValueError(
            "TARGET_RETURN_TOLERANCE must be positive."
        )


# Run immediately on import.
validate_config()


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 5 - PORTFOLIO REBALANCING CONFIGURATION"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Configuration validation : PASS"
    )

    print()

    print(
        f"Project root             : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Agent root               : "
        f"{AGENT_ROOT}"
    )

    print()

    print(
        f"Agent-4 input            : "
        f"{AGENT4_TRADE_DECISIONS_PARQUET}"
    )

    print(
        f"Agent-3 input            : "
        f"{AGENT3_RISK_PARQUET}"
    )

    print(
        f"Market input             : "
        f"{MARKET_DATA_PARQUET}"
    )

    print()

    print(
        f"Final rebalance date     : "
        f"{FINAL_REBALANCE_DATE}"
    )

    print()

    print(
        f"MPT objective            : "
        f"{MPT_OBJECTIVE}"
    )

    print(
        f"Target-return mode       : "
        f"{TARGET_RETURN_MODE}"
    )

    print()

    print(
        f"Return lookback          : "
        f"{ESTIMATION_LOOKBACK_DAYS} trading days"
    )

    print(
        f"Minimum observations     : "
        f"{MIN_REQUIRED_RETURN_OBSERVATIONS}"
    )

    print()

    print(
        f"Valid Agent-4 actions    : "
        f"{sorted(VALID_ACTIONS)}"
    )

    print(
        f"MPT eligible actions     : "
        f"{sorted(MPT_ELIGIBLE_ACTIONS)}"
    )

    print(
        f"SELL target weight       : "
        f"{SELL_TARGET_WEIGHT}"
    )

    print()

    print(
        f"Short selling            : "
        f"{ALLOW_SHORT_SELLING}"
    )

    print(
        f"Weight bounds            : "
        f"{MIN_WEIGHT} -> {MAX_WEIGHT}"
    )

    print(
        f"Weight sum target        : "
        f"{WEIGHT_SUM_TARGET}"
    )

    print()

    print(
        f"Optimizer                : "
        f"{OPTIMIZER_METHOD}"
    )

    print(
        f"Optimizer max iterations : "
        f"{OPTIMIZER_MAX_ITERATIONS}"
    )

    print(
        f"Optimizer FTOL           : "
        f"{OPTIMIZER_FTOL}"
    )

    print()

    print(
        f"Covariance ridge         : "
        f"{COVARIANCE_RIDGE}"
    )

    print()

    print(
        "=" * 100
    )

    print(
        "AGENT 5 CONFIGURATION STATUS: COMPLETE"
    )

    print(
        "=" * 100
    )