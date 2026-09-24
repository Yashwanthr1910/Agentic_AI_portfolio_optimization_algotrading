"""
Agent 1 -> Agent 2 Integration Validator
========================================

Purpose
-------
Validate that:

1. Agent 1 inference universe exists.
2. Agent 2 prediction files exist.
3. Every Agent 1 selected symbol received an Agent 2 prediction.
4. No unexpected symbols appeared.
5. Prediction probabilities are valid.
6. Ensemble ranks are valid.
7. Prediction dates are valid.
8. Required final output columns are present.

NO training.
NO prediction.
NO model selection.
"""

from pathlib import Path

import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

TREND_AGENT_ROOT = CURRENT_FILE.parents[1]

PROJECT_ROOT = TREND_AGENT_ROOT.parent


INFERENCE_UNIVERSE_FILE = (
    TREND_AGENT_ROOT
    / "data"
    / "processed"
    / "inference_universe.parquet"
)

PREDICTION_FILE = (
    TREND_AGENT_ROOT
    / "outputs"
    / "trend_predictions.parquet"
)

PREDICTION_CSV = (
    TREND_AGENT_ROOT
    / "outputs"
    / "trend_predictions.csv"
)

FINAL_MODEL_MANIFEST = (
    TREND_AGENT_ROOT
    / "reports"
    / "final_model"
    / "final_model_manifest.json"
)

PREDICTION_SUMMARY = (
    TREND_AGENT_ROOT
    / "reports"
    / "final_inference"
    / "prediction_summary.json"
)

VALIDATION_REPORT_DIR = (
    TREND_AGENT_ROOT
    / "reports"
    / "integration"
)

VALIDATION_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

VALIDATION_REPORT_FILE = (
    VALIDATION_REPORT_DIR
    / "agent1_agent2_validation.json"
)


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
# FILE VALIDATION
# ============================================================

def validate_files():

    section(
        "VALIDATING REQUIRED FILES"
    )

    required_files = [
        INFERENCE_UNIVERSE_FILE,
        PREDICTION_FILE,
        PREDICTION_CSV,
        FINAL_MODEL_MANIFEST,
        PREDICTION_SUMMARY,
    ]

    failures = []

    for path in required_files:

        exists = path.exists()

        print(
            f"{path.name:<40} : "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:
            failures.append(
                str(path)
            )

    if failures:

        raise FileNotFoundError(
            "Missing required files:\n\n"
            +
            "\n".join(
                failures
            )
        )

    return True


# ============================================================
# LOAD AGENT 1 SYMBOLS
# ============================================================

def load_agent1_symbols():

    section(
        "LOADING AGENT 1 OUTPUT"
    )

    data = pd.read_parquet(
        INFERENCE_UNIVERSE_FILE
    )

    if "symbol" not in data.columns:

        raise ValueError(
            "Agent 1 inference universe "
            "does not contain 'symbol'."
        )

    symbols = sorted(
        data[
            "symbol"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    print(
        f"Agent 1 selected stocks : "
        f"{len(symbols)}"
    )

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"{index:>2}. {symbol}"
        )

    return symbols


# ============================================================
# LOAD AGENT 2 PREDICTIONS
# ============================================================

def load_agent2_predictions():

    section(
        "LOADING AGENT 2 OUTPUT"
    )

    data = pd.read_parquet(
        PREDICTION_FILE
    )

    required_columns = [
        "symbol",
        "prediction_date",
        "top30_probability",
        "ensemble_std",
        "trend_class",
        "agent2_rank",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Agent 2 output is missing columns:\n"
            +
            "\n".join(
                missing
            )
        )

    data["symbol"] = (
        data["symbol"]
        .astype(str)
        .str.strip()
    )

    data["prediction_date"] = (
        pd.to_datetime(
            data["prediction_date"],
            errors="coerce",
        )
    )

    print(
        f"Predictions loaded : "
        f"{len(data)}"
    )

    print(
        f"Unique stocks      : "
        f"{data['symbol'].nunique()}"
    )

    print(
        f"Prediction date    : "
        f"{data['prediction_date'].min()} "
        f"-> "
        f"{data['prediction_date'].max()}"
    )

    return data


# ============================================================
# VALIDATE SYMBOL HANDOFF
# ============================================================

def validate_symbols(
    agent1_symbols,
    predictions,
):

    section(
        "VALIDATING AGENT 1 -> AGENT 2 SYMBOL HANDOFF"
    )

    agent2_symbols = sorted(
        predictions[
            "symbol"
        ]
        .unique()
        .tolist()
    )

    missing_in_agent2 = sorted(
        set(agent1_symbols)
        -
        set(agent2_symbols)
    )

    unexpected_in_agent2 = sorted(
        set(agent2_symbols)
        -
        set(agent1_symbols)
    )

    duplicates = (
        predictions[
            "symbol"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Agent 1 symbols       : "
        f"{len(agent1_symbols)}"
    )

    print(
        f"Agent 2 symbols       : "
        f"{len(agent2_symbols)}"
    )

    print(
        f"Missing in Agent 2    : "
        f"{len(missing_in_agent2)}"
    )

    print(
        f"Unexpected in Agent 2 : "
        f"{len(unexpected_in_agent2)}"
    )

    print(
        f"Duplicate predictions : "
        f"{duplicates}"
    )

    if missing_in_agent2:

        print()

        print(
            "Missing symbols:"
        )

        for symbol in missing_in_agent2:
            print(
                f"  - {symbol}"
            )

    if unexpected_in_agent2:

        print()

        print(
            "Unexpected symbols:"
        )

        for symbol in unexpected_in_agent2:
            print(
                f"  - {symbol}"
            )

    passed = (
        len(missing_in_agent2) == 0
        and
        len(unexpected_in_agent2) == 0
        and
        duplicates == 0
        and
        len(agent1_symbols)
        ==
        len(agent2_symbols)
    )

    return {
        "passed":
            bool(passed),

        "agent1_symbols":
            len(agent1_symbols),

        "agent2_symbols":
            len(agent2_symbols),

        "missing":
            missing_in_agent2,

        "unexpected":
            unexpected_in_agent2,

        "duplicate_predictions":
            int(duplicates),
    }


# ============================================================
# VALIDATE PROBABILITIES
# ============================================================

def validate_probabilities(
    predictions,
):

    section(
        "VALIDATING MODEL OUTPUTS"
    )

    probability = (
        predictions[
            "top30_probability"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    ensemble_std = (
        predictions[
            "ensemble_std"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    finite_probabilities = (
        np.isfinite(
            probability
        )
        .all()
    )

    probability_range_valid = (
        (
            probability >= 0
        ).all()
        and
        (
            probability <= 1
        ).all()
    )

    finite_std = (
        np.isfinite(
            ensemble_std
        )
        .all()
    )

    nonnegative_std = (
        ensemble_std >= 0
    ).all()

    allowed_classes = {
        "TOP30",
        "NOT_TOP30",
    }

    classes = set(
        predictions[
            "trend_class"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    classes_valid = (
        classes
        .issubset(
            allowed_classes
        )
    )

    class_matches_probability = (
        (
            (
                predictions[
                    "top30_probability"
                ]
                >=
                0.50
            )
            ==
            (
                predictions[
                    "trend_class"
                ]
                ==
                "TOP30"
            )
        )
        .all()
    )

    print(
        f"Finite probabilities      : "
        f"{finite_probabilities}"
    )

    print(
        f"Probability range [0,1]   : "
        f"{probability_range_valid}"
    )

    print(
        f"Finite ensemble std       : "
        f"{finite_std}"
    )

    print(
        f"Non-negative ensemble std : "
        f"{nonnegative_std}"
    )

    print(
        f"Valid classes             : "
        f"{classes_valid}"
    )

    print(
        f"Class matches threshold   : "
        f"{class_matches_probability}"
    )

    passed = all(
        [
            finite_probabilities,
            probability_range_valid,
            finite_std,
            nonnegative_std,
            classes_valid,
            class_matches_probability,
        ]
    )

    return {
        "passed":
            bool(passed),

        "finite_probabilities":
            bool(finite_probabilities),

        "probability_range_valid":
            bool(probability_range_valid),

        "finite_ensemble_std":
            bool(finite_std),

        "nonnegative_ensemble_std":
            bool(nonnegative_std),

        "classes_valid":
            bool(classes_valid),

        "class_matches_threshold":
            bool(
                class_matches_probability
            ),
    }


# ============================================================
# VALIDATE RANKING
# ============================================================

def validate_ranking(
    predictions,
):

    section(
        "VALIDATING AGENT 2 RANKING"
    )

    n = len(
        predictions
    )

    ranks = (
        predictions[
            "agent2_rank"
        ]
        .astype(int)
    )

    expected = set(
        range(
            1,
            n + 1,
        )
    )

    actual = set(
        ranks.tolist()
    )

    rank_complete = (
        actual
        ==
        expected
    )

    no_duplicate_ranks = (
        ranks.nunique()
        ==
        n
    )

    sorted_predictions = (
        predictions
        .sort_values(
            "agent2_rank"
        )
    )

    probabilities = (
        sorted_predictions[
            "top30_probability"
        ]
        .to_numpy()
    )

    descending_probability = (
        np.all(
            probabilities[:-1]
            >=
            probabilities[1:]
        )
        if len(probabilities) > 1
        else True
    )

    print(
        f"Ranks complete 1..N       : "
        f"{rank_complete}"
    )

    print(
        f"No duplicate ranks        : "
        f"{no_duplicate_ranks}"
    )

    print(
        f"Probability order correct : "
        f"{descending_probability}"
    )

    passed = all(
        [
            rank_complete,
            no_duplicate_ranks,
            descending_probability,
        ]
    )

    return {
        "passed":
            bool(passed),

        "rank_complete":
            bool(rank_complete),

        "no_duplicate_ranks":
            bool(no_duplicate_ranks),

        "descending_probability":
            bool(descending_probability),
    }


# ============================================================
# VALIDATE DATES
# ============================================================

def validate_dates(
    predictions,
):

    section(
        "VALIDATING PREDICTION DATES"
    )

    invalid_dates = int(
        predictions[
            "prediction_date"
        ]
        .isna()
        .sum()
    )

    unique_dates = int(
        predictions[
            "prediction_date"
        ]
        .nunique()
    )

    latest_date = (
        predictions[
            "prediction_date"
        ]
        .max()
    )

    earliest_date = (
        predictions[
            "prediction_date"
        ]
        .min()
    )

    print(
        f"Invalid dates : "
        f"{invalid_dates}"
    )

    print(
        f"Unique dates  : "
        f"{unique_dates}"
    )

    print(
        f"Earliest      : "
        f"{earliest_date}"
    )

    print(
        f"Latest        : "
        f"{latest_date}"
    )

    passed = (
        invalid_dates == 0
    )

    return {
        "passed":
            bool(passed),

        "invalid_dates":
            invalid_dates,

        "unique_prediction_dates":
            unique_dates,

        "earliest_prediction_date":
            str(
                earliest_date
            ),

        "latest_prediction_date":
            str(
                latest_date
            ),
    }


# ============================================================
# DISPLAY FINAL HANDOFF
# ============================================================

def display_final_output(
    predictions,
):

    section(
        "FINAL AGENT 1 -> AGENT 2 OUTPUT"
    )

    display = (
        predictions[
            [
                "agent2_rank",
                "symbol",
                "prediction_date",
                "top30_probability",
                "ensemble_std",
                "trend_class",
            ]
        ]
        .sort_values(
            "agent2_rank"
        )
        .copy()
    )

    display[
        "top30_probability"
    ] = (
        display[
            "top30_probability"
        ]
        .map(
            lambda value:
            f"{value:.4f}"
        )
    )

    display[
        "ensemble_std"
    ] = (
        display[
            "ensemble_std"
        ]
        .map(
            lambda value:
            f"{value:.4f}"
        )
    )

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def run_validation():

    section(
        "AGENT 1 -> AGENT 2 FINAL INTEGRATION VALIDATION"
    )

    validate_files()

    agent1_symbols = (
        load_agent1_symbols()
    )

    predictions = (
        load_agent2_predictions()
    )

    symbol_check = (
        validate_symbols(
            agent1_symbols,
            predictions,
        )
    )

    probability_check = (
        validate_probabilities(
            predictions
        )
    )

    ranking_check = (
        validate_ranking(
            predictions
        )
    )

    date_check = (
        validate_dates(
            predictions
        )
    )

    display_final_output(
        predictions
    )

    # ========================================================
    # OVERALL STATUS
    # ========================================================

    all_checks = {
        "file_check":
            True,

        "symbol_handoff":
            symbol_check[
                "passed"
            ],

        "probabilities":
            probability_check[
                "passed"
            ],

        "ranking":
            ranking_check[
                "passed"
            ],

        "dates":
            date_check[
                "passed"
            ],
    }

    overall_passed = all(
        all_checks.values()
    )

    report = {
        "integration":
            "Agent 1 -> Agent 2",

        "overall_passed":
            bool(
                overall_passed
            ),

        "checks":
            all_checks,

        "symbol_validation":
            symbol_check,

        "probability_validation":
            probability_check,

        "ranking_validation":
            ranking_check,

        "date_validation":
            date_check,

        "predictions_generated":
            int(
                len(
                    predictions
                )
            ),

        "top30_predictions":
            int(
                (
                    predictions[
                        "trend_class"
                    ]
                    ==
                    "TOP30"
                )
                .sum()
            ),

        "not_top30_predictions":
            int(
                (
                    predictions[
                        "trend_class"
                    ]
                    ==
                    "NOT_TOP30"
                )
                .sum()
            ),
    }

    with open(
        VALIDATION_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=4,
            default=str,
        )

    section(
        "INTEGRATION VALIDATION RESULT"
    )

    for check, passed in all_checks.items():

        print(
            f"{check:<25} : "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()

    if overall_passed:

        print(
            "OVERALL RESULT:"
        )

        print(
            "PASS"
        )

        print()

        print(
            "AGENT 1 -> AGENT 2 integration "
            "is valid."
        )

        print()

        print(
            "AGENT 2 STATUS:"
        )

        print(
            "COMPLETE"
        )

        print()

        print(
            "NEXT AGENT:"
        )

        print(
            "AGENT 3 - RISK MANAGEMENT"
        )

    else:

        print(
            "OVERALL RESULT:"
        )

        print(
            "FAIL"
        )

        print()

        print(
            "Do not start Agent 3 until "
            "the failed integration checks "
            "are corrected."
        )

    print()

    print(
        "Validation report:"
    )

    print(
        VALIDATION_REPORT_FILE
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    run_validation()