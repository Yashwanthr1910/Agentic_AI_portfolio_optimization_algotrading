"""
Agent 3 - Combined Risk Scoring and Allocation
===============================================================

Purpose
-------
Combine:

    Agent-2 trend probability
    Historical volatility
    Value at Risk
    Sharpe ratio
    Drawdown

into a transparent Agent-3 risk-adjusted ranking.

Paper alignment
---------------
The paper states that the Risk Management Agent:

    - analyzes historical volatility
    - analyzes Value at Risk
    - analyzes Sharpe ratio
    - adjusts portfolio allocation according to risk levels

However, the paper does NOT clearly provide an exact stock-level
scoring function.

Therefore:

    paper_core_risk_score

is our reproducible implementation based only on the paper-core
metrics.

The final:

    agent3_score

also incorporates Agent-2 trend probability and drawdown.

This final combination is an IMPLEMENTATION ENHANCEMENT and should
not be described as an exact equation from the paper.

Scoring logic
-------------
Higher is always better.

Volatility:
    lower volatility -> higher score

VaR:
    lower VaR -> higher score

Sharpe:
    higher Sharpe -> higher score

Drawdown:
    smaller drawdown magnitude -> higher score

Trend:
    higher Agent-2 P(TOP30) -> higher score
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

if str(RISK_AGENT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(RISK_AGENT_ROOT),
    )


from config import config as cfg


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
# INPUT PATHS
# ============================================================

def get_input_paths():

    return {

        "risk_metrics":
            cfg.PROCESSED_DIR
            / "risk_metrics_with_drawdown.parquet",

        "portfolio_var":
            cfg.PROCESSED_DIR
            / "portfolio_var.json",

        "portfolio_sharpe":
            cfg.PROCESSED_DIR
            / "portfolio_sharpe.json",

        "portfolio_drawdown":
            cfg.PROCESSED_DIR
            / "portfolio_drawdown.json",

    }


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

def validate_input_files():

    section(
        "VALIDATING RISK-SCORER INPUT FILES"
    )

    missing = []

    for name, path in get_input_paths().items():

        exists = Path(
            path
        ).exists()

        print(
            f"{name:<25} : "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:

            missing.append(
                str(path)
            )

    if missing:

        raise FileNotFoundError(

            "Risk scoring cannot continue. "
            "Missing files:\n\n"

            +
            "\n".join(
                missing
            )
        )


# ============================================================
# LOAD INPUTS
# ============================================================

def load_inputs():

    section(
        "LOADING AGENT 3 RISK METRICS"
    )

    metrics = pd.read_parquet(

        cfg.PROCESSED_DIR
        / "risk_metrics_with_drawdown.parquet"
    )

    with open(

        cfg.PROCESSED_DIR
        / "portfolio_var.json",

        "r",

        encoding="utf-8",

    ) as file:

        portfolio_var = json.load(
            file
        )


    with open(

        cfg.PROCESSED_DIR
        / "portfolio_sharpe.json",

        "r",

        encoding="utf-8",

    ) as file:

        portfolio_sharpe = json.load(
            file
        )


    with open(

        cfg.PROCESSED_DIR
        / "portfolio_drawdown.json",

        "r",

        encoding="utf-8",

    ) as file:

        portfolio_drawdown = json.load(
            file
        )


    print(
        f"Stocks loaded : "
        f"{len(metrics)}"
    )

    print(
        f"Columns       : "
        f"{len(metrics.columns)}"
    )

    print()

    print(
        f"Portfolio VaR        : "
        f"{portfolio_var['portfolio_var_percent']:.4f}%"
    )

    print(
        f"Portfolio Sharpe     : "
        f"{portfolio_sharpe['sharpe_ratio']:.4f}"
    )

    print(
        f"Portfolio Max DD     : "
        f"{portfolio_drawdown['max_drawdown_percent']:.2f}%"
    )

    return {

        "metrics":
            metrics,

        "portfolio_var":
            portfolio_var,

        "portfolio_sharpe":
            portfolio_sharpe,

        "portfolio_drawdown":
            portfolio_drawdown,

    }


# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

def validate_required_columns(
    metrics,
):

    required = [

        "symbol",

        "prediction_date",

        "top30_probability",

        "ensemble_std",

        "trend_class",

        "agent2_rank",

        "historical_volatility_annual",

        "individual_var_fraction",

        "individual_var_percent",

        "risk_contribution_fraction",

        "risk_contribution_percent",

        "component_var_percent",

        "sharpe_ratio",

        "max_drawdown",

        "max_drawdown_percent",

        "current_drawdown",

        "current_drawdown_percent",

    ]

    missing = [

        column

        for column in required

        if column
        not in metrics.columns
    ]

    if missing:

        raise ValueError(

            "Risk scorer is missing required columns:\n\n"

            +
            "\n".join(
                missing
            )
        )

    print(
        "Required columns : PASS"
    )


# ============================================================
# PERCENTILE SCORING
# ============================================================

def percentile_score(
    series,
    higher_is_better=True,
):

    """
    Produce a 0-1 cross-sectional score.

    Highest desirable value -> score close to 1.
    Lowest desirable value  -> score close to 0.

    We use rank percentile so metrics with very different
    numerical scales can be combined without arbitrary
    standard-deviation assumptions.
    """

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    if series.isna().any():

        raise ValueError(
            "Cannot rank metric containing missing values."
        )

    ranks = series.rank(

        method="average",

        pct=True,

        ascending=True,
    )

    if higher_is_better:

        score = ranks

    else:

        # Lowest value should receive highest score.

        score = (

            1.0
            -
            ranks

            +
            (
                1.0
                /
                len(series)
            )
        )

    return score.clip(
        0.0,
        1.0,
    )


# ============================================================
# PAPER-CORE RISK SCORES
# ============================================================

def calculate_core_risk_scores(
    metrics,
):

    section(
        "CALCULATING PAPER-CORE RISK SCORES"
    )

    data = metrics.copy()

    # --------------------------------------------------------
    # Historical volatility
    #
    # Lower is better.
    # --------------------------------------------------------

    data[
        "volatility_score"
    ] = percentile_score(

        data[
            "historical_volatility_annual"
        ],

        higher_is_better=False,
    )

    # --------------------------------------------------------
    # Individual VaR
    #
    # Lower is better.
    # --------------------------------------------------------

    data[
        "var_score"
    ] = percentile_score(

        data[
            "individual_var_fraction"
        ],

        higher_is_better=False,
    )

    # --------------------------------------------------------
    # Sharpe ratio
    #
    # Higher is better.
    # --------------------------------------------------------

    data[
        "sharpe_score"
    ] = percentile_score(

        data[
            "sharpe_ratio"
        ],

        higher_is_better=True,
    )

    # --------------------------------------------------------
    # PAPER-CORE RISK SCORE
    #
    # Equal weighting of the three explicitly stated
    # Agent-3 risk measures.
    #
    # This equal-weight combination is our implementation
    # choice because the paper does not specify weights.
    # --------------------------------------------------------

    data[
        "paper_core_risk_score"
    ] = (

        data[
            "volatility_score"
        ]

        +

        data[
            "var_score"
        ]

        +

        data[
            "sharpe_score"
        ]

    ) / 3.0


    print(
        data[
            [
                "symbol",
                "volatility_score",
                "var_score",
                "sharpe_score",
                "paper_core_risk_score",
            ]
        ]
        .sort_values(

            "paper_core_risk_score",

            ascending=False,

        )
        .to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}",
        )
    )

    return data


# ============================================================
# SUPPORTING SCORES
# ============================================================

def calculate_supporting_scores(
    data,
):

    section(
        "CALCULATING SUPPORTING SCORES"
    )

    result = data.copy()

    # --------------------------------------------------------
    # Drawdown
    #
    # max_drawdown is negative.
    #
    # We rank absolute drawdown magnitude:
    #
    # -20% is better than -40%.
    # --------------------------------------------------------

    drawdown_magnitude = (

        result[
            "max_drawdown"
        ]
        .abs()
    )

    result[
        "drawdown_score"
    ] = percentile_score(

        drawdown_magnitude,

        higher_is_better=False,
    )

    # --------------------------------------------------------
    # Agent-2 trend signal
    #
    # P(TOP30) already lies within [0,1].
    # --------------------------------------------------------

    result[
        "trend_score"
    ] = (

        result[
            "top30_probability"
        ]
        .clip(
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Ensemble uncertainty
    #
    # This is retained as diagnostic information.
    #
    # It is NOT used in the paper-core financial risk score.
    # --------------------------------------------------------

    result[
        "model_confidence_score"
    ] = percentile_score(

        result[
            "ensemble_std"
        ],

        higher_is_better=False,
    )

    return result


# ============================================================
# FINAL AGENT 3 SCORE
# ============================================================

def calculate_agent3_score(
    data,
):

    section(
        "CALCULATING FINAL AGENT 3 SCORE"
    )

    weight_sum = (

        cfg.TREND_SCORE_WEIGHT

        +

        cfg.CORE_RISK_SCORE_WEIGHT

        +

        cfg.DRAWDOWN_SCORE_WEIGHT
    )

    if not np.isclose(

        weight_sum,

        1.0,

        atol=1e-12,

    ):

        raise ValueError(

            "Agent-3 score weights must sum to 1.0. "
            f"Current sum = {weight_sum}"
        )

    result = data.copy()

    result[
        "agent3_score"
    ] = (

        cfg.TREND_SCORE_WEIGHT
        *
        result[
            "trend_score"
        ]

        +

        cfg.CORE_RISK_SCORE_WEIGHT
        *
        result[
            "paper_core_risk_score"
        ]

        +

        cfg.DRAWDOWN_SCORE_WEIGHT
        *
        result[
            "drawdown_score"
        ]
    )

    result[
        "agent3_rank"
    ] = (

        result[
            "agent3_score"
        ]
        .rank(

            method="first",

            ascending=False,
        )
        .astype(int)
    )

    result = (

        result
        .sort_values(
            "agent3_rank"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Trend weight      : "
        f"{cfg.TREND_SCORE_WEIGHT:.2f}"
    )

    print(
        f"Core risk weight  : "
        f"{cfg.CORE_RISK_SCORE_WEIGHT:.2f}"
    )

    print(
        f"Drawdown weight   : "
        f"{cfg.DRAWDOWN_SCORE_WEIGHT:.2f}"
    )

    print()

    print(
        result[
            [
                "agent3_rank",
                "symbol",
                "trend_score",
                "paper_core_risk_score",
                "drawdown_score",
                "agent3_score",
            ]
        ]
        .to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}",
        )
    )

    return result


# ============================================================
# RISK LEVEL
# ============================================================

def classify_risk_levels(
    data,
):

    section(
        "CLASSIFYING RISK LEVELS"
    )

    result = data.copy()

    # --------------------------------------------------------
    # Higher paper_core_risk_score means a better / safer
    # relative risk profile.
    # --------------------------------------------------------

    def classify(score):

        if score >= cfg.LOW_RISK_THRESHOLD:

            return "LOW"

        if score <= cfg.HIGH_RISK_THRESHOLD:

            return "HIGH"

        return "MEDIUM"

    result[
        "risk_level"
    ] = (

        result[
            "paper_core_risk_score"
        ]
        .apply(
            classify
        )
    )

    print(
        result[
            [
                "agent3_rank",
                "symbol",
                "paper_core_risk_score",
                "risk_level",
            ]
        ]
        .to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}",
        )
    )

    return result


# ============================================================
# RISK-ADJUSTED ALLOCATION
# ============================================================

def calculate_provisional_allocation(
    data,
):

    section(
        "CALCULATING RISK-ADJUSTED PROVISIONAL ALLOCATION"
    )

    result = data.copy()

    scores = (

        result[
            "agent3_score"
        ]
        .clip(
            lower=0.0
        )
    )

    if scores.sum() <= 0:

        raise ValueError(
            "Agent-3 scores cannot produce allocation."
        )

    raw_weights = (

        scores
        /
        scores.sum()
    )

    # --------------------------------------------------------
    # Cap individual positions.
    #
    # This is a provisional Agent-3 risk-control rule.
    #
    # Agent 5 will later perform final portfolio optimization.
    # --------------------------------------------------------

    capped = raw_weights.clip(

        lower=cfg.MIN_PROVISIONAL_WEIGHT,

        upper=cfg.MAX_PROVISIONAL_WEIGHT,
    )

    # Iteratively redistribute remaining weight to uncapped
    # positions so total allocation remains approximately 1.

    weights = capped.copy()

    for _ in range(100):

        total = weights.sum()

        difference = (
            1.0
            -
            total
        )

        if abs(
            difference
        ) < 1e-12:

            break

        if difference > 0:

            eligible = (

                weights
                <
                cfg.MAX_PROVISIONAL_WEIGHT
                -
                1e-12
            )

            if not eligible.any():

                break

            eligible_scores = scores[
                eligible
            ]

            addition = (

                difference

                *
                eligible_scores
                /
                eligible_scores.sum()
            )

            weights.loc[
                eligible
            ] += addition

            weights = weights.clip(

                upper=
                cfg.MAX_PROVISIONAL_WEIGHT
            )

        else:

            eligible = (

                weights
                >
                cfg.MIN_PROVISIONAL_WEIGHT
                +
                1e-12
            )

            if not eligible.any():

                break

            eligible_weights = weights[
                eligible
            ]

            reduction = (

                (-difference)

                *
                eligible_weights
                /
                eligible_weights.sum()
            )

            weights.loc[
                eligible
            ] -= reduction

            weights = weights.clip(

                lower=
                cfg.MIN_PROVISIONAL_WEIGHT
            )

    # Final defensive normalization.
    #
    # Under the present 10-stock / 25%-cap setup this should
    # remain feasible.

    if not np.isclose(

        weights.sum(),

        1.0,

        atol=1e-8,

    ):

        weights = (

            weights
            /
            weights.sum()
        )

    result[
        "risk_adjusted_weight"
    ] = weights.values

    result[
        "risk_adjusted_weight_percent"
    ] = (

        result[
            "risk_adjusted_weight"
        ]

        *
        100.0
    )

    print(
        result[
            [
                "agent3_rank",
                "symbol",
                "agent3_score",
                "risk_level",
                "risk_adjusted_weight_percent",
            ]
        ]
        .to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    print(
        f"Weight sum : "
        f"{result['risk_adjusted_weight'].sum():.10f}"
    )

    return result


# ============================================================
# AGENT-3 DECISION LABEL
# ============================================================

def create_risk_decision(
    data,
):

    section(
        "CREATING AGENT 3 RISK DECISIONS"
    )

    result = data.copy()

    # --------------------------------------------------------
    # This is NOT Buy/Sell/Hold.
    #
    # Agent 4 will make actual trading decisions.
    #
    # Agent 3 only describes whether a candidate is relatively
    # attractive after risk adjustment.
    # --------------------------------------------------------

    def decision(row):

        if (
            row[
                "trend_class"
            ]
            ==
            "TOP30"

            and

            row[
                "risk_level"
            ]
            ==
            "LOW"
        ):

            return "FAVORABLE"

        if (
            row[
                "trend_class"
            ]
            ==
            "TOP30"

            and

            row[
                "risk_level"
            ]
            ==
            "MEDIUM"
        ):

            return "ACCEPTABLE"

        if (
            row[
                "risk_level"
            ]
            ==
            "HIGH"
        ):

            return "CAUTION"

        return "WATCH"

    result[
        "risk_decision"
    ] = result.apply(

        decision,

        axis=1,
    )

    print(
        result[
            [
                "agent3_rank",
                "symbol",
                "trend_class",
                "risk_level",
                "risk_decision",
            ]
        ]
        .to_string(
            index=False
        )
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    data,
):

    section(
        "VALIDATING AGENT 3 RISK SCORING"
    )

    failures = []

    # --------------------------------------------------------
    # Score ranges
    # --------------------------------------------------------

    score_columns = [

        "volatility_score",

        "var_score",

        "sharpe_score",

        "paper_core_risk_score",

        "drawdown_score",

        "trend_score",

        "model_confidence_score",

        "agent3_score",

    ]

    score_range_valid = True

    for column in score_columns:

        valid = (

            data[
                column
            ]
            .between(
                0,
                1,
                inclusive="both",
            )
            .all()

            and

            np.isfinite(
                data[
                    column
                ]
            ).all()
        )

        if not valid:

            score_range_valid = False

            print(
                f"Invalid score column: "
                f"{column}"
            )

    print(
        f"All scores finite and [0,1] : "
        f"{score_range_valid}"
    )

    if not score_range_valid:

        failures.append(
            "score ranges"
        )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    expected_ranks = set(
        range(
            1,
            len(data)
            +
            1
        )
    )

    actual_ranks = set(

        data[
            "agent3_rank"
        ]
        .astype(int)
    )

    rank_valid = (

        expected_ranks
        ==
        actual_ranks
    )

    print(
        f"Agent 3 ranks complete       : "
        f"{rank_valid}"
    )

    if not rank_valid:

        failures.append(
            "Agent 3 ranking"
        )

    # --------------------------------------------------------
    # Descending score order
    # --------------------------------------------------------

    descending_valid = (

        data[
            "agent3_score"
        ]
        .is_monotonic_decreasing
    )

    print(
        f"Scores sorted descending     : "
        f"{descending_valid}"
    )

    if not descending_valid:

        failures.append(
            "score ordering"
        )

    # --------------------------------------------------------
    # Risk levels
    # --------------------------------------------------------

    allowed_levels = {

        "LOW",
        "MEDIUM",
        "HIGH",

    }

    levels_valid = (

        set(
            data[
                "risk_level"
            ]
        )
        .issubset(
            allowed_levels
        )
    )

    print(
        f"Valid risk levels            : "
        f"{levels_valid}"
    )

    if not levels_valid:

        failures.append(
            "risk levels"
        )

    # --------------------------------------------------------
    # Allocation
    # --------------------------------------------------------

    weights = (

        data[
            "risk_adjusted_weight"
        ]
    )

    weight_sum_valid = np.isclose(

        weights.sum(),

        1.0,

        atol=1e-8,
    )

    print(
        f"Allocation weights sum to 1  : "
        f"{weight_sum_valid}"
    )

    if not weight_sum_valid:

        failures.append(
            "allocation weight sum"
        )

    weights_nonnegative = (

        weights
        >=
        0
    ).all()

    print(
        f"Allocation non-negative      : "
        f"{weights_nonnegative}"
    )

    if not weights_nonnegative:

        failures.append(
            "negative allocation"
        )

    max_weight_valid = (

        weights.max()

        <=

        cfg.MAX_PROVISIONAL_WEIGHT
        +
        1e-8
    )

    print(
        f"Maximum allocation respected : "
        f"{max_weight_valid}"
    )

    if not max_weight_valid:

        failures.append(
            "maximum allocation"
        )

    # --------------------------------------------------------
    # No symbol loss
    # --------------------------------------------------------

    symbol_valid = (

        data[
            "symbol"
        ]
        .nunique()
        ==
        len(data)
    )

    print(
        f"Unique stocks                : "
        f"{symbol_valid}"
    )

    if not symbol_valid:

        failures.append(
            "duplicate symbols"
        )

    if failures:

        print()

        print(
            "VALIDATION RESULT: FAIL"
        )

        print()

        for failure in failures:

            print(
                f"  - {failure}"
            )

        raise ValueError(
            "Agent 3 risk scoring validation failed."
        )

    print()

    print(
        "VALIDATION RESULT: PASS"
    )

    return True


# ============================================================
# FINAL OUTPUT COLUMNS
# ============================================================

def build_final_output(
    data,
):

    section(
        "BUILDING FINAL AGENT 3 OUTPUT"
    )

    output_columns = [

        # Identity / timing
        "symbol",
        "prediction_date",

        # Agent 1/2 information
        "agent2_rank",
        "top30_probability",
        "ensemble_std",
        "trend_class",

        # Paper-core risk measures
        "historical_volatility_annual",
        "individual_var_percent",
        "sharpe_ratio",

        # Portfolio-risk decomposition
        "risk_contribution_percent",
        "component_var_percent",

        # Supporting downside diagnostics
        "max_drawdown_percent",
        "current_drawdown_percent",

        # Normalized paper-core scores
        "volatility_score",
        "var_score",
        "sharpe_score",
        "paper_core_risk_score",

        # Supporting normalized scores
        "drawdown_score",
        "model_confidence_score",
        "trend_score",

        # Agent 3 result
        "agent3_score",
        "agent3_rank",
        "risk_level",
        "risk_decision",

        # Provisional allocation
        "risk_adjusted_weight",
        "risk_adjusted_weight_percent",

    ]

    missing = [

        column

        for column
        in output_columns

        if column
        not in data.columns
    ]

    if missing:

        raise ValueError(

            "Unable to create final Agent 3 output. "
            "Missing columns:\n\n"

            +
            "\n".join(
                missing
            )
        )

    final_output = (

        data[
            output_columns
        ]
        .sort_values(
            "agent3_rank"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        final_output[
            [
                "agent3_rank",
                "symbol",
                "top30_probability",
                "sharpe_ratio",
                "historical_volatility_annual",
                "individual_var_percent",
                "agent3_score",
                "risk_level",
                "risk_decision",
                "risk_adjusted_weight_percent",
            ]
        ]
        .to_string(

            index=False,

            float_format=lambda x: f"{x:.4f}",
        )
    )

    return final_output


# ============================================================
# SAVE FINAL OUTPUT
# ============================================================

def save_results(
    final_output,
    inputs,
):

    section(
        "SAVING FINAL AGENT 3 OUTPUT"
    )

    final_output.to_csv(

        cfg.RISK_ASSESSMENT_CSV,

        index=False,
    )

    final_output.to_parquet(

        cfg.RISK_ASSESSMENT_PARQUET,

        index=False,
    )


    summary = {

        "agent":
            3,

        "agent_name":
            cfg.AGENT_NAME,

        "stocks":
            int(
                len(
                    final_output
                )
            ),

        "paper_core_metrics": [

            "Historical Volatility",

            "Value at Risk",

            "Sharpe Ratio",

        ],

        "implementation_enhancements": [

            "Drawdown diagnostic",

            "Percentile normalization",

            "Combined Agent-3 score",

            "Risk-level classification",

            "Risk-adjusted provisional allocation",

        ],

        "score_weights": {

            "trend":
                cfg.TREND_SCORE_WEIGHT,

            "paper_core_risk":
                cfg.CORE_RISK_SCORE_WEIGHT,

            "drawdown":
                cfg.DRAWDOWN_SCORE_WEIGHT,

        },

        "portfolio_metrics": {

            "var_percent":
                inputs[
                    "portfolio_var"
                ][
                    "portfolio_var_percent"
                ],

            "sharpe_ratio":
                inputs[
                    "portfolio_sharpe"
                ][
                    "sharpe_ratio"
                ],

            "max_drawdown_percent":
                inputs[
                    "portfolio_drawdown"
                ][
                    "max_drawdown_percent"
                ],

        },

        "best_agent3_stock":

            str(
                final_output
                .iloc[0][
                    "symbol"
                ]
            ),

    }


    with open(

        cfg.RISK_SUMMARY_JSON,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            summary,

            file,

            indent=4,
        )


    print(
        cfg.RISK_ASSESSMENT_CSV
    )

    print(
        cfg.RISK_ASSESSMENT_PARQUET
    )

    print(
        cfg.RISK_SUMMARY_JSON
    )


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_risk_scorer():

    section(
        "AGENT 3 - COMBINED RISK SCORING"
    )

    print(
        "Paper-core measurements:"
    )

    print()

    print(
        "1. Historical Volatility"
    )

    print(
        "2. Value at Risk"
    )

    print(
        "3. Sharpe Ratio"
    )

    print()

    print(
        "Implementation enhancement:"
    )

    print()

    print(
        "Transparent normalized scoring + "
        "Agent-2 trend integration + drawdown"
    )


    validate_input_files()


    inputs = (
        load_inputs()
    )


    metrics = (
        inputs[
            "metrics"
        ]
    )


    validate_required_columns(
        metrics
    )


    scored = (
        calculate_core_risk_scores(
            metrics
        )
    )


    scored = (
        calculate_supporting_scores(
            scored
        )
    )


    scored = (
        calculate_agent3_score(
            scored
        )
    )


    scored = (
        classify_risk_levels(
            scored
        )
    )


    scored = (
        calculate_provisional_allocation(
            scored
        )
    )


    scored = (
        create_risk_decision(
            scored
        )
    )


    validate_results(
        scored
    )


    final_output = (
        build_final_output(
            scored
        )
    )


    save_results(

        final_output,

        inputs,
    )


    section(
        "AGENT 3 RISK SCORING COMPLETE"
    )

    best = (
        final_output.iloc[0]
    )

    print(
        f"Stocks assessed : "
        f"{len(final_output)}"
    )

    print()

    print(
        f"Top Agent-3 candidate : "
        f"{best['symbol']}"
    )

    print(
        f"Agent-3 score         : "
        f"{best['agent3_score']:.4f}"
    )

    print(
        f"Risk level            : "
        f"{best['risk_level']}"
    )

    print(
        f"Risk decision         : "
        f"{best['risk_decision']}"
    )

    print(
        f"Provisional weight    : "
        f"{best['risk_adjusted_weight_percent']:.2f}%"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "These are provisional risk-adjusted weights."
    )

    print(
        "Final portfolio optimization belongs to Agent 5."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Agent 3 orchestration + integration validation"
    )

    return final_output


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_risk_scorer()