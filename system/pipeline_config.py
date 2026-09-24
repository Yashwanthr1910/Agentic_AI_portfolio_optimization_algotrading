"""
pipeline_config.py

Central configuration for the complete five-agent
Agentic AI Portfolio Optimization system.

This module does NOT execute any agent.

Its responsibilities are:

1. Define the project root.
2. Define Agent 1 -> Agent 5 directories.
3. Define candidate production entry points.
4. Define required output files.
5. Define cross-agent integration constraints.
6. Define final inference/rebalance date.
7. Define system report locations.
8. Provide reusable helper functions for the root pipeline.

Pipeline
--------
Agent 1
    Stock Selection
    Decision Tree + BGSTO
        ↓
Agent 2
    Trend Prediction
    Dilated LSTM + Transformer + Progressive Attention
        ↓
Agent 3
    Risk Management
    Volatility + VaR + Sharpe
        ↓
Agent 4
    Trade Execution
    DQN BUY / HOLD / SELL
        ↓
Agent 5
    Portfolio Rebalancing
    Markowitz MPT + RL Feedback
        ↓
Final Portfolio
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# PROJECT ROOT
# =============================================================================

# Current file:
#
# project_root/
# └── system/
#     └── pipeline_config.py
#
# parents[1] therefore gives:
#
# project_root/

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# SYSTEM INFORMATION
# =============================================================================

SYSTEM_NAME = (
    "Agentic AI Portfolio Optimization "
    "and Algorithmic Trading System"
)

SYSTEM_VERSION = "1.0"

NUMBER_OF_AGENTS = 5


# =============================================================================
# CURRENT FINAL INFERENCE CONFIGURATION
# =============================================================================

# All current final Agent 1 -> Agent 5 inference outputs correspond to:
#
#     2025-12-01
#
# Later this can be made dynamic for historical / live execution.

FINAL_INFERENCE_DATE = "2025-12-01"

FINAL_REBALANCE_DATE = FINAL_INFERENCE_DATE


# =============================================================================
# EXPECTED STOCK UNIVERSE
# =============================================================================

EXPECTED_SELECTED_STOCKS = 10

EXPECTED_AGENT2_STOCKS = 10

EXPECTED_AGENT3_STOCKS = 10

EXPECTED_AGENT4_STOCKS = 10

EXPECTED_AGENT5_STOCKS = 10


# =============================================================================
# VALID TRADE ACTIONS
# =============================================================================

ACTION_HOLD = "HOLD"
ACTION_BUY = "BUY"
ACTION_SELL = "SELL"

VALID_TRADE_ACTIONS = {
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
}


# =============================================================================
# CURRENT EXPECTED AGENT-4 DISTRIBUTION
# =============================================================================

# These values are useful for validating the CURRENT locked system.
#
# They are NOT universal rules.
#
# If models are retrained or a different inference date is used,
# this distribution may legitimately change.

CURRENT_EXPECTED_ACTION_COUNTS = {
    ACTION_BUY: 1,
    ACTION_HOLD: 7,
    ACTION_SELL: 2,
}


# =============================================================================
# AGENT ROOT DIRECTORIES
# =============================================================================

AGENT1_ROOT = (
    PROJECT_ROOT
    / "stock_selection_agent"
)

AGENT2_ROOT = (
    PROJECT_ROOT
    / "trend_prediction_agent"
)

AGENT3_ROOT = (
    PROJECT_ROOT
    / "risk_management_agent"
)

AGENT4_ROOT = (
    PROJECT_ROOT
    / "trade_execution_agent"
)

AGENT5_ROOT = (
    PROJECT_ROOT
    / "portfolio_rebalancing_agent"
)


AGENT_ROOTS: Dict[int, Path] = {
    1: AGENT1_ROOT,
    2: AGENT2_ROOT,
    3: AGENT3_ROOT,
    4: AGENT4_ROOT,
    5: AGENT5_ROOT,
}


# =============================================================================
# AGENT NAMES
# =============================================================================

AGENT_NAMES: Dict[int, str] = {
    1: "Stock Selection Agent",
    2: "Trend Prediction Agent",
    3: "Risk Management Agent",
    4: "Trade Execution Agent",
    5: "Portfolio Rebalancing Agent",
}


# =============================================================================
# AGENT 1 PATHS
# =============================================================================

AGENT1_SELECTED_STOCKS_CSV = (
    AGENT1_ROOT
    / "data"
    / "outputs"
    / "selected_stocks.csv"
)

AGENT1_SELECTED_STOCKS_PARQUET = (
    AGENT1_ROOT
    / "data"
    / "outputs"
    / "selected_stocks.parquet"
)

AGENT1_RANKINGS_CSV = (
    AGENT1_ROOT
    / "outputs"
    / "rankings.csv"
)

AGENT1_RANKINGS_PARQUET = (
    AGENT1_ROOT
    / "outputs"
    / "rankings.parquet"
)

AGENT1_DECISION_TREE_PREDICTIONS = (
    AGENT1_ROOT
    / "data"
    / "outputs"
    / "decision_tree_predictions.parquet"
)

AGENT1_BGSTO_SELECTION_CSV = (
    AGENT1_ROOT
    / "data"
    / "outputs"
    / "bgsto_selected_stocks.csv"
)

AGENT1_BGSTO_SELECTION_PARQUET = (
    AGENT1_ROOT
    / "data"
    / "outputs"
    / "bgsto_selected_stocks.parquet"
)

AGENT1_FEATURES_PARQUET = (
    AGENT1_ROOT
    / "data"
    / "processed"
    / "features.parquet"
)

AGENT1_MARKET_DATA_PARQUET = (
    AGENT1_ROOT
    / "data"
    / "processed"
    / "nifty500_daily.parquet"
)

AGENT1_LABELED_FEATURES = (
    AGENT1_ROOT
    / "data"
    / "processed"
    / "labeled_features.parquet"
)


# =============================================================================
# AGENT 2 PATHS
# =============================================================================

AGENT2_TREND_PREDICTIONS_CSV = (
    AGENT2_ROOT
    / "outputs"
    / "trend_predictions.csv"
)

AGENT2_TREND_PREDICTIONS_PARQUET = (
    AGENT2_ROOT
    / "outputs"
    / "trend_predictions.parquet"
)

AGENT2_INFERENCE_UNIVERSE = (
    AGENT2_ROOT
    / "data"
    / "processed"
    / "inference_universe.parquet"
)

AGENT2_TRAINING_UNIVERSE = (
    AGENT2_ROOT
    / "data"
    / "processed"
    / "training_universe.parquet"
)

AGENT2_PREPROCESSED_DATA = (
    AGENT2_ROOT
    / "data"
    / "processed"
    / "preprocessed_trend_data.parquet"
)

AGENT2_SCALER = (
    AGENT2_ROOT
    / "data"
    / "processed"
    / "scaler.pkl"
)

AGENT2_SEQUENCE_INDEX = (
    AGENT2_ROOT
    / "data"
    / "processed"
    / "sequence_index.parquet"
)

AGENT2_SEQUENCE_METADATA = (
    AGENT2_ROOT
    / "data"
    / "processed"
    / "sequence_metadata.json"
)

AGENT2_MODEL_DIR = (
    AGENT2_ROOT
    / "models"
    / "seed_stability"
)

AGENT2_SEED_42_WEIGHTS = (
    AGENT2_MODEL_DIR
    / "full_paper_model_seed_42.weights.h5"
)

AGENT2_SEED_123_WEIGHTS = (
    AGENT2_MODEL_DIR
    / "full_paper_model_seed_123.weights.h5"
)

AGENT2_SEED_456_WEIGHTS = (
    AGENT2_MODEL_DIR
    / "full_paper_model_seed_456.weights.h5"
)


# =============================================================================
# AGENT 3 PATHS
# =============================================================================

AGENT3_RISK_CSV = (
    AGENT3_ROOT
    / "outputs"
    / "risk_assessment.csv"
)

AGENT3_RISK_PARQUET = (
    AGENT3_ROOT
    / "outputs"
    / "risk_assessment.parquet"
)

AGENT3_RISK_SUMMARY = (
    AGENT3_ROOT
    / "reports"
    / "risk_summary.json"
)


# =============================================================================
# AGENT 4 PATHS
# =============================================================================

AGENT4_TRADE_DECISIONS_CSV = (
    AGENT4_ROOT
    / "outputs"
    / "trade_decisions.csv"
)

AGENT4_TRADE_DECISIONS_PARQUET = (
    AGENT4_ROOT
    / "outputs"
    / "trade_decisions.parquet"
)

AGENT4_PREDICTION_SUMMARY = (
    AGENT4_ROOT
    / "reports"
    / "prediction_summary.json"
)

AGENT3_AGENT4_VALIDATION = (
    AGENT4_ROOT
    / "reports"
    / "agent3_agent4_validation.json"
)


# =============================================================================
# AGENT 5 PATHS
# =============================================================================

AGENT5_REBALANCED_PORTFOLIO_CSV = (
    AGENT5_ROOT
    / "outputs"
    / "rebalanced_portfolio.csv"
)

AGENT5_REBALANCED_PORTFOLIO_PARQUET = (
    AGENT5_ROOT
    / "outputs"
    / "rebalanced_portfolio.parquet"
)

AGENT5_FINAL_ALLOCATION_CSV = (
    AGENT5_ROOT
    / "outputs"
    / "final_portfolio_allocation.csv"
)

AGENT5_SUMMARY = (
    AGENT5_ROOT
    / "reports"
    / "agent5_summary.json"
)

AGENT4_AGENT5_VALIDATION = (
    AGENT5_ROOT
    / "reports"
    / "agent4_agent5_validation.json"
)

AGENT5_EVALUATION_SUMMARY = (
    AGENT5_ROOT
    / "reports"
    / "evaluation_summary.json"
)


# =============================================================================
# SYSTEM REPORT PATHS
# =============================================================================

SYSTEM_REPORT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "system"
)

SYSTEM_LATEST_DIR = (
    SYSTEM_REPORT_ROOT
    / "latest"
)

SYSTEM_SUMMARY_JSON = (
    SYSTEM_LATEST_DIR
    / "system_summary.json"
)

SYSTEM_INTEGRATION_JSON = (
    SYSTEM_LATEST_DIR
    / "integration_summary.json"
)

SYSTEM_FULL_TRACE_CSV = (
    SYSTEM_LATEST_DIR
    / "full_agent_trace.csv"
)

SYSTEM_FULL_TRACE_PARQUET = (
    SYSTEM_LATEST_DIR
    / "full_agent_trace.parquet"
)

SYSTEM_STAGE_TIMINGS_CSV = (
    SYSTEM_LATEST_DIR
    / "stage_timings.csv"
)


# =============================================================================
# ENTRY-POINT CANDIDATES
# =============================================================================

# Some agents evolved through multiple development versions.
#
# Instead of assuming that every agent has exactly the same file layout,
# the system preflight will later select the FIRST EXISTING candidate.
#
# This avoids breaking the full pipeline because of an old/new location.

AGENT_ENTRYPOINT_CANDIDATES: Dict[int, Tuple[Path, ...]] = {

    # -------------------------------------------------------------------------
    # Agent 1
    # -------------------------------------------------------------------------
    1: (
        AGENT1_ROOT / "main.py",
        AGENT1_ROOT / "agent.py",
        AGENT1_ROOT / "src" / "agent.py",
    ),

    # -------------------------------------------------------------------------
    # Agent 2
    #
    # The final production execution path is main.py.
    # -------------------------------------------------------------------------
    2: (
        AGENT2_ROOT / "main.py",
    ),

    # -------------------------------------------------------------------------
    # Agent 3
    #
    # Prefer a root orchestrator if present.
    # Otherwise the validated risk scorer can be used.
    # -------------------------------------------------------------------------
    3: (
        AGENT3_ROOT / "main.py",
        AGENT3_ROOT / "src" / "agent.py",
        AGENT3_ROOT / "src" / "risk_scorer.py",
    ),

    # -------------------------------------------------------------------------
    # Agent 4
    #
    # Final production execution path.
    # -------------------------------------------------------------------------
    4: (
        AGENT4_ROOT / "main.py",
    ),

    # -------------------------------------------------------------------------
    # Agent 5
    #
    # Final production execution path.
    # -------------------------------------------------------------------------
    5: (
        AGENT5_ROOT / "main.py",
    ),
}


# =============================================================================
# OPTIONAL INTEGRATION VALIDATORS
# =============================================================================

AGENT_INTEGRATION_VALIDATOR_CANDIDATES: Dict[int, Tuple[Path, ...]] = {

    # Agent 2 normally validates Agent 1 -> Agent 2 itself.
    2: (
        AGENT2_ROOT
        / "src"
        / "integration_validator.py",
    ),

    # Agent 3 validation if available.
    3: (
        AGENT3_ROOT
        / "src"
        / "integration_validator.py",
    ),

    # Agent 4 validation.
    4: (
        AGENT4_ROOT
        / "src"
        / "integration_validator.py",
    ),

    # Agent 5 validation.
    5: (
        AGENT5_ROOT
        / "src"
        / "integration_validator.py",
    ),
}


# =============================================================================
# OUTPUT GROUPS
# =============================================================================

# Every inner tuple is an OR condition.
#
# Example:
#
#   (csv, parquet)
#
# means:
#
#   at least one of the files must exist.
#
# Multiple groups for an agent are AND conditions.

AGENT_REQUIRED_OUTPUT_GROUPS: Dict[int, Tuple[Tuple[Path, ...], ...]] = {

    # Agent 1
    1: (
        (
            AGENT1_SELECTED_STOCKS_CSV,
            AGENT1_SELECTED_STOCKS_PARQUET,
        ),
        (
            AGENT1_RANKINGS_CSV,
            AGENT1_RANKINGS_PARQUET,
        ),
    ),

    # Agent 2
    2: (
        (
            AGENT2_TREND_PREDICTIONS_CSV,
            AGENT2_TREND_PREDICTIONS_PARQUET,
        ),
    ),

    # Agent 3
    3: (
        (
            AGENT3_RISK_CSV,
            AGENT3_RISK_PARQUET,
        ),
    ),

    # Agent 4
    4: (
        (
            AGENT4_TRADE_DECISIONS_CSV,
            AGENT4_TRADE_DECISIONS_PARQUET,
        ),
    ),

    # Agent 5
    5: (
        (
            AGENT5_REBALANCED_PORTFOLIO_CSV,
            AGENT5_REBALANCED_PORTFOLIO_PARQUET,
        ),
        (
            AGENT5_FINAL_ALLOCATION_CSV,
        ),
    ),
}


# =============================================================================
# IMPORTANT SUPPORT FILES
# =============================================================================

# Files that should exist before performing normal inference.
#
# These are not necessarily generated during every pipeline run.

AGENT_SUPPORT_FILE_GROUPS: Dict[int, Tuple[Tuple[Path, ...], ...]] = {

    # Agent 1 historical/prepared data
    1: (
        (
            AGENT1_MARKET_DATA_PARQUET,
        ),
        (
            AGENT1_FEATURES_PARQUET,
        ),
        (
            AGENT1_LABELED_FEATURES,
        ),
    ),

    # Agent 2 locked inference resources
    2: (
        (
            AGENT2_SCALER,
        ),
        (
            AGENT2_SEQUENCE_METADATA,
        ),
        (
            AGENT2_SEED_42_WEIGHTS,
        ),
        (
            AGENT2_SEED_123_WEIGHTS,
        ),
        (
            AGENT2_SEED_456_WEIGHTS,
        ),
    ),

    # Agent 3 mainly consumes upstream output + market data.
    3: (
        (
            AGENT2_TREND_PREDICTIONS_CSV,
            AGENT2_TREND_PREDICTIONS_PARQUET,
        ),
        (
            AGENT1_MARKET_DATA_PARQUET,
        ),
    ),

    # Agent 4 requires Agent-3 output.
    4: (
        (
            AGENT3_RISK_CSV,
            AGENT3_RISK_PARQUET,
        ),
    ),

    # Agent 5 requires Agent-4 + Agent-3 + historical prices.
    5: (
        (
            AGENT4_TRADE_DECISIONS_CSV,
            AGENT4_TRADE_DECISIONS_PARQUET,
        ),
        (
            AGENT3_RISK_CSV,
            AGENT3_RISK_PARQUET,
        ),
        (
            AGENT1_MARKET_DATA_PARQUET,
        ),
    ),
}


# =============================================================================
# AGENT OUTPUT COLUMN CONTRACTS
# =============================================================================

AGENT1_REQUIRED_COLUMNS = {
    "symbol",
}

AGENT2_REQUIRED_COLUMNS = {
    "symbol",
    "prediction_date",
    "top30_probability",
    "trend_class",
    "agent2_rank",
}

AGENT3_REQUIRED_COLUMNS = {
    "symbol",
    "agent3_score",
    "agent3_rank",
    "risk_level",
    "risk_adjusted_weight",
}

AGENT4_REQUIRED_COLUMNS = {
    "symbol",
    "trade_action",
    "q_hold",
    "q_buy",
    "q_sell",
}

AGENT5_REQUIRED_COLUMNS = {
    "symbol",
    "trade_action",
    "final_weight",
}


AGENT_REQUIRED_COLUMNS: Dict[int, set[str]] = {
    1: AGENT1_REQUIRED_COLUMNS,
    2: AGENT2_REQUIRED_COLUMNS,
    3: AGENT3_REQUIRED_COLUMNS,
    4: AGENT4_REQUIRED_COLUMNS,
    5: AGENT5_REQUIRED_COLUMNS,
}


# =============================================================================
# PIPELINE ORDER
# =============================================================================

PIPELINE_ORDER = (
    1,
    2,
    3,
    4,
    5,
)


# =============================================================================
# DATA CLASS FOR AGENT DESCRIPTION
# =============================================================================

@dataclass(frozen=True)
class AgentDefinition:
    """
    Immutable metadata describing one pipeline agent.
    """

    number: int
    name: str
    root: Path
    entrypoint_candidates: Tuple[Path, ...]
    required_output_groups: Tuple[Tuple[Path, ...], ...]
    support_file_groups: Tuple[Tuple[Path, ...], ...]
    required_columns: set[str]


# =============================================================================
# BUILD AGENT DEFINITIONS
# =============================================================================

AGENTS: Dict[int, AgentDefinition] = {
    number: AgentDefinition(
        number=number,
        name=AGENT_NAMES[number],
        root=AGENT_ROOTS[number],
        entrypoint_candidates=AGENT_ENTRYPOINT_CANDIDATES[number],
        required_output_groups=AGENT_REQUIRED_OUTPUT_GROUPS[number],
        support_file_groups=AGENT_SUPPORT_FILE_GROUPS[number],
        required_columns=AGENT_REQUIRED_COLUMNS[number],
    )
    for number in PIPELINE_ORDER
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_system_directories() -> None:
    """
    Create directories used by the root pipeline.
    """

    SYSTEM_REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SYSTEM_LATEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def first_existing_path(
    candidates: Tuple[Path, ...],
) -> Optional[Path]:
    """
    Return the first existing path from a candidate group.

    Returns None if no candidate exists.
    """

    for path in candidates:

        if path.exists():

            return path

    return None


def resolve_agent_entrypoint(
    agent_number: int,
) -> Optional[Path]:
    """
    Resolve the production entry point for an agent.

    The first existing candidate is selected.
    """

    if agent_number not in AGENTS:

        raise ValueError(
            f"Invalid agent number: {agent_number}"
        )

    return first_existing_path(
        AGENTS[
            agent_number
        ].entrypoint_candidates
    )


def resolve_integration_validator(
    agent_number: int,
) -> Optional[Path]:
    """
    Return an optional integration-validator script if available.
    """

    candidates = (
        AGENT_INTEGRATION_VALIDATOR_CANDIDATES
        .get(
            agent_number,
            (),
        )
    )

    return first_existing_path(
        candidates
    )


def output_group_exists(
    group: Tuple[Path, ...],
) -> bool:
    """
    Return True when at least one file in an OR-group exists.
    """

    return any(
        path.exists()
        for path in group
    )


def all_agent_output_groups_exist(
    agent_number: int,
) -> bool:
    """
    Check whether every required output group exists for an agent.
    """

    if agent_number not in AGENTS:

        raise ValueError(
            f"Invalid agent number: {agent_number}"
        )

    groups = (
        AGENTS[
            agent_number
        ].required_output_groups
    )

    return all(
        output_group_exists(
            group
        )
        for group in groups
    )


def all_agent_support_groups_exist(
    agent_number: int,
) -> bool:
    """
    Check whether all required support-file groups exist.
    """

    if agent_number not in AGENTS:

        raise ValueError(
            f"Invalid agent number: {agent_number}"
        )

    groups = (
        AGENTS[
            agent_number
        ].support_file_groups
    )

    return all(
        output_group_exists(
            group
        )
        for group in groups
    )


def describe_path_group(
    group: Tuple[Path, ...],
) -> str:
    """
    Produce a readable representation of an OR-path group.
    """

    return " OR ".join(
        str(path)
        for path in group
    )


def get_agent_description(
    agent_number: int,
) -> AgentDefinition:
    """
    Return complete metadata for a pipeline agent.
    """

    if agent_number not in AGENTS:

        raise ValueError(
            f"Invalid agent number: {agent_number}"
        )

    return AGENTS[
        agent_number
    ]


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_configuration() -> Dict[str, bool]:
    """
    Validate only the static pipeline configuration.

    This does NOT validate dataframe contents.

    Detailed file/content validation belongs to preflight.py.
    """

    checks: Dict[str, bool] = {}

    checks[
        "project_root_exists"
    ] = PROJECT_ROOT.exists()

    checks[
        "system_directory_exists"
    ] = (
        PROJECT_ROOT
        / "system"
    ).exists()

    checks[
        "five_agents_configured"
    ] = len(
        AGENTS
    ) == 5

    checks[
        "pipeline_order_valid"
    ] = PIPELINE_ORDER == (
        1,
        2,
        3,
        4,
        5,
    )

    for number in PIPELINE_ORDER:

        definition = AGENTS[
            number
        ]

        checks[
            f"agent{number}_root_exists"
        ] = definition.root.exists()

        checks[
            f"agent{number}_entrypoint_exists"
        ] = (
            resolve_agent_entrypoint(
                number
            )
            is not None
        )

    return checks


# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

def print_configuration_summary() -> None:
    """
    Print a concise summary of the complete pipeline configuration.
    """

    ensure_system_directories()

    separator = "=" * 90

    print()
    print(separator)
    print(
        "AGENTIC AI PORTFOLIO OPTIMIZATION "
        "- SYSTEM CONFIGURATION"
    )
    print(separator)

    print()
    print(
        f"Project root       : {PROJECT_ROOT}"
    )

    print(
        f"System version     : {SYSTEM_VERSION}"
    )

    print(
        f"Inference date     : {FINAL_INFERENCE_DATE}"
    )

    print(
        f"Expected stocks    : {EXPECTED_SELECTED_STOCKS}"
    )

    print()
    print("PIPELINE")
    print("-" * 90)

    for number in PIPELINE_ORDER:

        definition = AGENTS[
            number
        ]

        entrypoint = resolve_agent_entrypoint(
            number
        )

        print(
            f"Agent {number}: "
            f"{definition.name}"
        )

        print(
            f"  Root       : "
            f"{definition.root}"
        )

        print(
            f"  Entrypoint : "
            f"{entrypoint if entrypoint else 'NOT FOUND'}"
        )

    print()
    print("SYSTEM OUTPUT DIRECTORY")
    print("-" * 90)

    print(
        SYSTEM_LATEST_DIR
    )

    print()
    print(separator)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print_configuration_summary()

    checks = validate_configuration()

    print()
    print(
        "CONFIGURATION CHECKS"
    )

    print(
        "-" * 90
    )

    for name, passed in checks.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{name:<45} : {status}"
        )

    print()

    if all(
        checks.values()
    ):

        print(
            "PIPELINE CONFIGURATION STATUS: PASS"
        )

    else:

        print(
            "PIPELINE CONFIGURATION STATUS: "
            "REQUIRES ATTENTION"
        )