"""
Agent 3 - Integration Validator
===============================================================

Purpose
-------
Validate the integration between:

    Agent 2 - Trend Prediction Agent
    Agent 3 - Risk Management Agent

The validator checks:

    1. Agent 2 prediction file exists
    2. Agent 3 risk assessment exists
    3. All Agent 2 symbols are preserved
    4. Prediction dates are consistent
    5. Agent 2 probabilities are unchanged
    6. Agent 2 ranks are preserved
    7. Agent 3 scores are valid
    8. Agent 3 ranks are complete
    9. Risk levels are valid
   10. Risk-adjusted weights sum to 1
   11. No negative weights exist
   12. Core risk metrics are present
   13. Output is suitable for Agent 4

This is an engineering validation layer, not a paper algorithm.
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
RISK_AGENT_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = RISK_AGENT_ROOT.parent

if str(RISK_AGENT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(RISK_AGENT_ROOT),
    )

from config import config as cfg


# ============================================================
# PATHS
# ============================================================

AGENT2_PARQUET = (
    PROJECT_ROOT
    / "trend_prediction_agent"
    / "outputs"
    / "trend_predictions.parquet"
)

AGENT2_CSV = (
    PROJECT_ROOT
    / "trend_prediction_agent"
    / "outputs"
    / "trend_predictions.csv"
)

AGENT3_PARQUET = (
    cfg.RISK_ASSESSMENT_PARQUET
)

AGENT3_CSV = (
    cfg.RISK_ASSESSMENT_CSV
)

REPORT_FILE = (
    cfg.REPORT_DIR
    / "agent2_agent3_validation.json"
)


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
# LOAD AGENT 2
# ============================================================

def load_agent2_predictions():

    section(
        "LOADING AGENT 2 PREDICTIONS"
    )

    if AGENT2_PARQUET.exists():

        data = pd.read_parquet(
            AGENT2_PARQUET
        )

        source = AGENT2_PARQUET

    elif AGENT2_CSV.exists():

        data = pd.read_csv(
            AGENT2_CSV
        )

        source = AGENT2_CSV

    else:

        raise FileNotFoundError(
            "Agent 2 prediction output was not found."
        )

    print(
        f"Source : {source}"
    )

    print(
        f"Rows   : {len(data)}"
    )

    print(
        f"Stocks : {data['symbol'].nunique()}"
    )

    data["symbol"] = (
        data["symbol"]
        .astype(str)
        .str.strip()
    )

    if "prediction_date" in data.columns:

        data["prediction_date"] = pd.to_datetime(
            data["prediction_date"]
        )

    return data


# ============================================================
# LOAD AGENT 3
# ============================================================

def load_agent3_assessment():

    section(
        "LOADING AGENT 3 RISK ASSESSMENT"
    )

    if AGENT3_PARQUET.exists():

        data = pd.read_parquet(
            AGENT3_PARQUET
        )

        source = AGENT3_PARQUET

    elif AGENT3_CSV.exists():

        data = pd.read_csv(
            AGENT3_CSV
        )

        source = AGENT3_CSV

    else:

        raise FileNotFoundError(
            "Agent 3 risk assessment output was not found."
        )

    print(
        f"Source : {source}"
    )

    print(
        f"Rows   : {len(data)}"
    )

    print(
        f"Stocks : {data['symbol'].nunique()}"
    )

    data["symbol"] = (
        data["symbol"]
        .astype(str)
        .str.strip()
    )

    if "prediction_date" in data.columns:

        data["prediction_date"] = pd.to_datetime(
            data["prediction_date"]
        )

    return data


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

def validate_required_columns(
    agent2,
    agent3,
):

    section(
        "VALIDATING REQUIRED COLUMNS"
    )

    required_agent2 = [

        "symbol",
        "prediction_date",
        "top30_probability",
        "ensemble_std",
        "trend_class",
        "agent2_rank",

    ]

    required_agent3 = [

        "symbol",
        "prediction_date",

        "agent2_rank",
        "top30_probability",
        "ensemble_std",
        "trend_class",

        "historical_volatility_annual",
        "individual_var_percent",
        "sharpe_ratio",

        "risk_contribution_percent",
        "component_var_percent",

        "max_drawdown_percent",
        "current_drawdown_percent",

        "paper_core_risk_score",

        "agent3_score",
        "agent3_rank",

        "risk_level",
        "risk_decision",

        "risk_adjusted_weight",
        "risk_adjusted_weight_percent",

    ]

    missing_agent2 = [

        col
        for col in required_agent2
        if col not in agent2.columns

    ]

    missing_agent3 = [

        col
        for col in required_agent3
        if col not in agent3.columns

    ]

    print(
        f"Agent 2 required columns : "
        f"{len(missing_agent2) == 0}"
    )

    print(
        f"Agent 3 required columns : "
        f"{len(missing_agent3) == 0}"
    )

    if missing_agent2:

        raise ValueError(
            "Agent 2 missing columns:\n"
            + "\n".join(
                missing_agent2
            )
        )

    if missing_agent3:

        raise ValueError(
            "Agent 3 missing columns:\n"
            + "\n".join(
                missing_agent3
            )
        )


# ============================================================
# SYMBOL COVERAGE
# ============================================================

def validate_symbol_coverage(
    agent2,
    agent3,
):

    section(
        "VALIDATING SYMBOL COVERAGE"
    )

    symbols2 = set(
        agent2["symbol"]
    )

    symbols3 = set(
        agent3["symbol"]
    )

    missing = sorted(
        symbols2 - symbols3
    )

    extra = sorted(
        symbols3 - symbols2
    )

    exact_match = (
        symbols2 == symbols3
    )

    print(
        f"Agent 2 stocks : {len(symbols2)}"
    )

    print(
        f"Agent 3 stocks : {len(symbols3)}"
    )

    print(
        f"Exact match    : {exact_match}"
    )

    if missing:

        print(
            f"Missing in Agent 3 : {missing}"
        )

    if extra:

        print(
            f"Extra in Agent 3   : {extra}"
        )

    return exact_match


# ============================================================
# MERGE FOR FIELD VALIDATION
# ============================================================

def merge_agents(
    agent2,
    agent3,
):

    merged = agent2.merge(

        agent3,

        on="symbol",

        how="inner",

        suffixes=(
            "_agent2",
            "_agent3",
        ),

        validate="one_to_one",
    )

    return merged


# ============================================================
# AGENT 2 FIELD PRESERVATION
# ============================================================

def validate_agent2_fields(
    merged,
):

    section(
        "VALIDATING AGENT 2 FIELD PRESERVATION"
    )

    date_match = (

        merged[
            "prediction_date_agent2"
        ]
        ==
        merged[
            "prediction_date_agent3"
        ]

    ).all()

    probability_match = np.allclose(

        merged[
            "top30_probability_agent2"
        ].astype(float),

        merged[
            "top30_probability_agent3"
        ].astype(float),

        atol=1e-12,

        rtol=0,
    )

    uncertainty_match = np.allclose(

        merged[
            "ensemble_std_agent2"
        ].astype(float),

        merged[
            "ensemble_std_agent3"
        ].astype(float),

        atol=1e-12,

        rtol=0,
    )

    rank_match = (

        merged[
            "agent2_rank_agent2"
        ].astype(int)
        ==
        merged[
            "agent2_rank_agent3"
        ].astype(int)

    ).all()

    trend_class_match = (

        merged[
            "trend_class_agent2"
        ].astype(str)
        ==
        merged[
            "trend_class_agent3"
        ].astype(str)

    ).all()

    print(
        f"Prediction dates preserved : {date_match}"
    )

    print(
        f"Probabilities preserved     : {probability_match}"
    )

    print(
        f"Ensemble std preserved      : {uncertainty_match}"
    )

    print(
        f"Agent 2 ranks preserved     : {rank_match}"
    )

    print(
        f"Trend classes preserved     : {trend_class_match}"
    )

    return {

        "prediction_dates_preserved":
            bool(date_match),

        "probabilities_preserved":
            bool(probability_match),

        "ensemble_std_preserved":
            bool(uncertainty_match),

        "agent2_ranks_preserved":
            bool(rank_match),

        "trend_classes_preserved":
            bool(trend_class_match),

    }


# ============================================================
# AGENT 3 VALIDATION
# ============================================================

def validate_agent3_metrics(
    agent3,
):

    section(
        "VALIDATING AGENT 3 METRICS"
    )

    checks = {}

    # --------------------------------------------------------
    # Paper-core measures
    # --------------------------------------------------------

    volatility_valid = (

        np.isfinite(
            agent3[
                "historical_volatility_annual"
            ]
        ).all()

        and

        (
            agent3[
                "historical_volatility_annual"
            ]
            >=
            0
        ).all()
    )

    var_valid = (

        np.isfinite(
            agent3[
                "individual_var_percent"
            ]
        ).all()

        and

        (
            agent3[
                "individual_var_percent"
            ]
            >=
            0
        ).all()
    )

    sharpe_valid = (

        np.isfinite(
            agent3[
                "sharpe_ratio"
            ]
        ).all()
    )

    # --------------------------------------------------------
    # Agent 3 scores
    # --------------------------------------------------------

    score_valid = (

        np.isfinite(
            agent3[
                "agent3_score"
            ]
        ).all()

        and

        agent3[
            "agent3_score"
        ]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
    )

    core_score_valid = (

        np.isfinite(
            agent3[
                "paper_core_risk_score"
            ]
        ).all()

        and

        agent3[
            "paper_core_risk_score"
        ]
        .between(
            0,
            1,
            inclusive="both",
        )
        .all()
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    expected_ranks = set(
        range(
            1,
            len(agent3) + 1
        )
    )

    actual_ranks = set(

        agent3[
            "agent3_rank"
        ]
        .astype(int)
    )

    ranks_valid = (
        expected_ranks
        ==
        actual_ranks
    )

    # --------------------------------------------------------
    # Risk levels
    # --------------------------------------------------------

    allowed_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    risk_levels_valid = (

        set(
            agent3[
                "risk_level"
            ]
        )
        .issubset(
            allowed_levels
        )
    )

    # --------------------------------------------------------
    # Risk decisions
    # --------------------------------------------------------

    allowed_decisions = {

        "FAVORABLE",
        "ACCEPTABLE",
        "WATCH",
        "CAUTION",

    }

    decisions_valid = (

        set(
            agent3[
                "risk_decision"
            ]
        )
        .issubset(
            allowed_decisions
        )
    )

    # --------------------------------------------------------
    # Weights
    # --------------------------------------------------------

    weights = (

        agent3[
            "risk_adjusted_weight"
        ]
        .astype(float)
    )

    weight_sum_valid = np.isclose(

        weights.sum(),

        1.0,

        atol=1e-8,
    )

    weight_nonnegative = (

        weights
        >=
        0
    ).all()

    # --------------------------------------------------------
    # Uniqueness
    # --------------------------------------------------------

    symbols_unique = (

        agent3[
            "symbol"
        ]
        .nunique()
        ==
        len(agent3)
    )

    checks.update({

        "historical_volatility_valid":
            bool(volatility_valid),

        "individual_var_valid":
            bool(var_valid),

        "sharpe_valid":
            bool(sharpe_valid),

        "paper_core_score_valid":
            bool(core_score_valid),

        "agent3_score_valid":
            bool(score_valid),

        "agent3_ranks_valid":
            bool(ranks_valid),

        "risk_levels_valid":
            bool(risk_levels_valid),

        "risk_decisions_valid":
            bool(decisions_valid),

        "weight_sum_valid":
            bool(weight_sum_valid),

        "weights_nonnegative":
            bool(weight_nonnegative),

        "symbols_unique":
            bool(symbols_unique),

    })

    for name, value in checks.items():

        print(
            f"{name:<35} : {value}"
        )

    return checks


# ============================================================
# AGENT 4 READINESS
# ============================================================

def validate_agent4_readiness(
    agent3,
):

    section(
        "VALIDATING AGENT 4 READINESS"
    )

    # Agent 4 will later consume these as state/features.
    #
    # We are not fixing the final DQN state vector here yet.
    # This only ensures the information is available.

    candidate_columns = [

        "symbol",

        "top30_probability",

        "ensemble_std",

        "trend_class",

        "historical_volatility_annual",

        "individual_var_percent",

        "sharpe_ratio",

        "risk_contribution_percent",

        "max_drawdown_percent",

        "agent3_score",

        "agent3_rank",

        "risk_level",

        "risk_adjusted_weight",

    ]

    missing = [

        column

        for column
        in candidate_columns

        if column
        not in agent3.columns
    ]

    ready = (
        len(missing)
        ==
        0
    )

    print(
        f"Required candidate state fields present : {ready}"
    )

    if missing:

        print()

        print(
            "Missing Agent 4 candidate fields:"
        )

        for column in missing:

            print(
                f"  - {column}"
            )

    return {

        "agent4_ready":
            ready,

        "candidate_columns":
            candidate_columns,

        "missing_columns":
            missing,

    }


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_integration():

    section(
        "AGENT 2 -> AGENT 3 INTEGRATION VALIDATION"
    )

    agent2 = (
        load_agent2_predictions()
    )

    agent3 = (
        load_agent3_assessment()
    )

    validate_required_columns(
        agent2,
        agent3,
    )

    symbol_match = (
        validate_symbol_coverage(
            agent2,
            agent3,
        )
    )

    merged = (
        merge_agents(
            agent2,
            agent3,
        )
    )

    preservation = (
        validate_agent2_fields(
            merged
        )
    )

    agent3_checks = (
        validate_agent3_metrics(
            agent3
        )
    )

    agent4_readiness = (
        validate_agent4_readiness(
            agent3
        )
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    all_checks = {

        "symbol_coverage_exact":
            bool(symbol_match),

        **preservation,

        **agent3_checks,

        "agent4_ready":
            bool(
                agent4_readiness[
                    "agent4_ready"
                ]
            ),

    }

    passed = all(
        all_checks.values()
    )

    section(
        "INTEGRATION VALIDATION SUMMARY"
    )

    for name, value in all_checks.items():

        print(
            f"{name:<40} : {value}"
        )

    print()

    if passed:

        print(
            "VALIDATION RESULT: PASS"
        )

    else:

        print(
            "VALIDATION RESULT: FAIL"
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {

        "agent2_rows":
            int(
                len(agent2)
            ),

        "agent3_rows":
            int(
                len(agent3)
            ),

        "agent2_symbols":
            int(
                agent2[
                    "symbol"
                ]
                .nunique()
            ),

        "agent3_symbols":
            int(
                agent3[
                    "symbol"
                ]
                .nunique()
            ),

        "checks":
            all_checks,

        "agent4_candidate_columns":
            agent4_readiness[
                "candidate_columns"
            ],

        "status":
            "PASS"
            if passed
            else "FAIL",

    }

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print()

    print(
        f"Report saved:"
    )

    print(
        REPORT_FILE
    )

    if not passed:

        failed = [

            name

            for name, value
            in all_checks.items()

            if not value
        ]

        raise ValueError(
            "Agent 2 -> Agent 3 integration validation failed:\n"
            +
            "\n".join(
                failed
            )
        )

    return report


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_validation():

    return validate_integration()


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_validation()