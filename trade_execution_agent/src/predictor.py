"""
Agent 4 - Final DQN Trade Predictor
===============================================================================

Purpose
-------
Generate final BUY / SELL / HOLD decisions for the stocks received from
Agent 3.

Final inference date:
    2025-12-01

DQN:
    7 state inputs
    64 hidden units
    64 hidden units
    3 Q-values

Actions:
    0 = HOLD
    1 = BUY
    2 = SELL

Important
---------
This file refuses to generate final predictions if training_summary.json
does not confirm a full 100-episode training run. This prevents an
accidental smoke-test model from being used for final Agent-4 output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# PATH SETUP
# =============================================================================

AGENT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(AGENT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(AGENT_ROOT),
    )


from config import config as cfg
from src.dqn_agent import DQNAgent


# =============================================================================
# PATHS
# =============================================================================

PREDICTION_REPORT_PATH = (
    cfg.REPORT_DIR
    / "prediction_summary.json"
)


# =============================================================================
# TRAINING CHECK
# =============================================================================

def validate_full_training() -> dict[str, Any]:
    """
    Ensure that final inference uses the full 100-episode model rather
    than a 2-episode smoke-test checkpoint.
    """

    summary_path = Path(
        cfg.TRAINING_SUMMARY_PATH
    )

    if not summary_path.exists():

        raise FileNotFoundError(
            "training_summary.json does not exist.\n"
            "Run full DQN training before final prediction."
        )

    with open(
        summary_path,
        "r",
        encoding="utf-8",
    ) as file:

        summary = json.load(
            file
        )

    episodes = int(
        summary.get(
            "episodes",
            0,
        )
    )

    learning_steps = int(
        summary.get(
            "learning_steps",
            0,
        )
    )

    target_updates = int(
        summary.get(
            "target_updates",
            0,
        )
    )

    if episodes != cfg.EPISODES:

        raise ValueError(
            "FINAL PREDICTION BLOCKED.\n\n"
            f"Saved training report has only {episodes} episodes.\n"
            f"Expected: {cfg.EPISODES}\n\n"
            "The saved model may be a smoke-test checkpoint.\n"
            "Run:\n"
            "python -c \"from trade_execution_agent.src.trainer "
            "import run_full_training; run_full_training()\""
        )

    if learning_steps <= 0:

        raise ValueError(
            "Training summary reports no DQN learning."
        )

    if target_updates <= 0:

        raise ValueError(
            "Training summary reports no target-network updates."
        )

    return summary


# =============================================================================
# INFERENCE DATA
# =============================================================================

def load_inference_data() -> pd.DataFrame:
    """
    Load final 2025 Agent-4 state/context data.
    """

    parquet_path = Path(
        cfg.DQN_INFERENCE_DATA_PATH
    )

    csv_path = Path(
        cfg.DQN_INFERENCE_DATA_CSV
    )

    if parquet_path.exists():

        try:

            df = pd.read_parquet(
                parquet_path
            )

            print(
                "Inference data format  : PARQUET"
            )

        except Exception as error:

            print(
                "Parquet loading failed."
            )

            print(
                f"Reason                 : "
                f"{type(error).__name__}: {error}"
            )

            if not csv_path.exists():

                raise

            df = pd.read_csv(
                csv_path,
                low_memory=False,
            )

            print(
                "Inference data format  : CSV"
            )

    elif csv_path.exists():

        df = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        print(
            "Inference data format  : CSV"
        )

    else:

        raise FileNotFoundError(
            "Agent-4 inference data was not found.\n"
            f"Expected:\n{parquet_path}\n"
            f"or:\n{csv_path}"
        )

    if df.empty:

        raise ValueError(
            "Agent-4 inference dataset is empty."
        )

    required_columns = {
        "symbol",
        "prediction_date",
        "market_state_date",
        *cfg.STATE_FEATURES,
    }

    missing = (
        required_columns
        -
        set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "Inference data missing columns:\n"
            +
            "\n".join(
                sorted(
                    missing
                )
            )
        )

    df = df.copy()

    df[
        "symbol"
    ] = (
        df[
            "symbol"
        ]
        .astype(
            str
        )
        .str.strip()
    )

    df[
        "prediction_date"
    ] = pd.to_datetime(
        df[
            "prediction_date"
        ],
        errors="coerce",
    )

    df[
        "market_state_date"
    ] = pd.to_datetime(
        df[
            "market_state_date"
        ],
        errors="coerce",
    )

    if df[
        "prediction_date"
    ].isna().any():

        raise ValueError(
            "Inference prediction dates are invalid."
        )

    if df[
        "market_state_date"
    ].isna().any():

        raise ValueError(
            "Inference market-state dates are invalid."
        )

    if (
        df[
            "symbol"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Inference data contains duplicate symbols."
        )

    for column in cfg.STATE_FEATURES:

        df[
            column
        ] = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )

    feature_matrix = (
        df[
            cfg.STATE_FEATURES
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    if not np.isfinite(
        feature_matrix
    ).all():

        raise ValueError(
            "Inference state features contain NaN or infinity."
        )

    if (
        df[
            "market_state_date"
        ]
        >
        df[
            "prediction_date"
        ]
    ).any():

        raise ValueError(
            "Future market data detected in inference dataset."
        )

    configured_date = pd.Timestamp(
        cfg.FINAL_INFERENCE_DATE
    )

    if not (
        df[
            "prediction_date"
        ]
        ==
        configured_date
    ).all():

        raise ValueError(
            "Inference prediction date does not match "
            f"FINAL_INFERENCE_DATE={cfg.FINAL_INFERENCE_DATE}"
        )

    return (
        df
        .sort_values(
            [
                "agent3_rank"
                if "agent3_rank" in df.columns
                else "symbol"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# MODEL
# =============================================================================

def load_final_agent() -> DQNAgent:
    """
    Load full trained DQN for greedy final inference.
    """

    weights_path = Path(
        cfg.DQN_WEIGHTS_PATH
    )

    if not weights_path.exists():

        raise FileNotFoundError(
            "DQN weights do not exist:\n"
            f"{weights_path}"
        )

    agent = DQNAgent()

    agent.load_weights(
        weights_path=weights_path,
        sync_target=True,
    )

    agent.set_evaluation_mode()

    if float(
        agent.epsilon
    ) != 0.0:

        raise ValueError(
            "Final DQN must use greedy inference with epsilon=0."
        )

    return agent


# =============================================================================
# PREDICT
# =============================================================================

def generate_trade_decisions(
    inference_df: pd.DataFrame,
    agent: DQNAgent,
) -> pd.DataFrame:

    records: list[
        dict[str, Any]
    ] = []

    for _, row in inference_df.iterrows():

        state = (
            row[
                cfg.STATE_FEATURES
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        if state.shape != (
            cfg.STATE_SIZE,
        ):

            raise ValueError(
                f"Invalid state shape for {row['symbol']}: "
                f"{state.shape}"
            )

        q_values = (
            agent.predict_q_values(
                state
            )
        )

        q_values = np.asarray(
            q_values,
            dtype=np.float32,
        ).reshape(
            -1
        )

        if q_values.shape != (
            cfg.NUM_ACTIONS,
        ):

            raise ValueError(
                f"Unexpected Q-value shape for "
                f"{row['symbol']}: {q_values.shape}"
            )

        if not np.isfinite(
            q_values
        ).all():

            raise ValueError(
                f"Invalid Q-values for {row['symbol']}."
            )

        action_id = int(
            np.argmax(
                q_values
            )
        )

        trade_action = (
            cfg.ACTION_NAMES[
                action_id
            ]
        )

        record = row.to_dict()

        record[
            "q_hold"
        ] = float(
            q_values[
                cfg.ACTION_HOLD
            ]
        )

        record[
            "q_buy"
        ] = float(
            q_values[
                cfg.ACTION_BUY
            ]
        )

        record[
            "q_sell"
        ] = float(
            q_values[
                cfg.ACTION_SELL
            ]
        )

        record[
            "action_id"
        ] = action_id

        record[
            "trade_action"
        ] = trade_action

        record[
            "selected_q_value"
        ] = float(
            q_values[
                action_id
            ]
        )

        records.append(
            record
        )

    result = pd.DataFrame(
        records
    )

    preferred_columns = [
        "symbol",
        "prediction_date",
        "market_state_date",
        "q_hold",
        "q_buy",
        "q_sell",
        "selected_q_value",
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

    existing_preferred = [
        column
        for column
        in preferred_columns
        if column
        in result.columns
    ]

    remaining_columns = [
        column
        for column
        in result.columns
        if column
        not in existing_preferred
    ]

    result = result[
        existing_preferred
        +
        remaining_columns
    ]

    return result


# =============================================================================
# SAVE
# =============================================================================

def save_predictions(
    predictions: pd.DataFrame,
    training_summary: dict[str, Any],
) -> None:

    cfg.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        cfg.TRADE_DECISIONS_CSV_PATH,
        index=False,
    )

    parquet_saved = False
    parquet_error = None

    try:

        predictions.to_parquet(
            cfg.TRADE_DECISIONS_PARQUET_PATH,
            index=False,
        )

        parquet_saved = True

    except Exception as error:

        parquet_error = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    action_counts = (
        predictions[
            "trade_action"
        ]
        .value_counts()
        .to_dict()
    )

    summary = {
        "status":
            "COMPLETE",

        "prediction_date":
            str(
                predictions[
                    "prediction_date"
                ]
                .iloc[
                    0
                ]
                .date()
            ),

        "stocks":
            int(
                len(
                    predictions
                )
            ),

        "symbols":
            predictions[
                "symbol"
            ].tolist(),

        "policy":
            "greedy",

        "epsilon":
            0.0,

        "training_episodes":
            int(
                training_summary[
                    "episodes"
                ]
            ),

        "training_learning_steps":
            int(
                training_summary[
                    "learning_steps"
                ]
            ),

        "training_target_updates":
            int(
                training_summary[
                    "target_updates"
                ]
            ),

        "action_counts": {
            "HOLD":
                int(
                    action_counts.get(
                        "HOLD",
                        0,
                    )
                ),

            "BUY":
                int(
                    action_counts.get(
                        "BUY",
                        0,
                    )
                ),

            "SELL":
                int(
                    action_counts.get(
                        "SELL",
                        0,
                    )
                ),
        },

        "csv_output":
            str(
                cfg.TRADE_DECISIONS_CSV_PATH
            ),

        "parquet_output":
            str(
                cfg.TRADE_DECISIONS_PARQUET_PATH
            ),

        "parquet_saved":
            parquet_saved,

        "parquet_error":
            parquet_error,

        "methodology_note":
            (
                "Final Agent-4 DQN inference uses the seven "
                "causal market-state features with greedy "
                "BUY/SELL/HOLD action selection. Agent-2 and "
                "Agent-3 values are retained as downstream "
                "context and are not silently added to the "
                "seven-node DQN state."
            ),
    }

    with open(
        PREDICTION_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_predictions(
    predictions: pd.DataFrame,
) -> dict[str, bool]:

    checks = {}

    checks[
        "predictions_not_empty"
    ] = (
        not predictions.empty
    )

    checks[
        "one_row_per_symbol"
    ] = (
        not predictions[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    checks[
        "state_count_matches_agent3"
    ] = (
        len(
            predictions
        )
        ==
        10
    )

    checks[
        "actions_valid"
    ] = bool(
        predictions[
            "action_id"
        ]
        .isin(
            [
                cfg.ACTION_HOLD,
                cfg.ACTION_BUY,
                cfg.ACTION_SELL,
            ]
        )
        .all()
    )

    checks[
        "action_names_valid"
    ] = bool(
        predictions[
            "trade_action"
        ]
        .isin(
            [
                "HOLD",
                "BUY",
                "SELL",
            ]
        )
        .all()
    )

    checks[
        "q_values_finite"
    ] = bool(
        np.isfinite(
            predictions[
                [
                    "q_hold",
                    "q_buy",
                    "q_sell",
                ]
            ]
            .to_numpy(
                dtype=float
            )
        ).all()
    )

    checks[
        "argmax_consistent"
    ] = bool(
        (
            predictions[
                "action_id"
            ]
            .to_numpy(
                dtype=int
            )
            ==
            np.argmax(
                predictions[
                    [
                        "q_hold",
                        "q_buy",
                        "q_sell",
                    ]
                ]
                .to_numpy(
                    dtype=float
                ),
                axis=1,
            )
        ).all()
    )

    checks[
        "prediction_date_valid"
    ] = bool(
        (
            pd.to_datetime(
                predictions[
                    "prediction_date"
                ]
            )
            ==
            pd.Timestamp(
                cfg.FINAL_INFERENCE_DATE
            )
        ).all()
    )

    checks[
        "no_future_market_state"
    ] = bool(
        (
            pd.to_datetime(
                predictions[
                    "market_state_date"
                ]
            )
            <=
            pd.to_datetime(
                predictions[
                    "prediction_date"
                ]
            )
        ).all()
    )

    checks[
        "csv_saved"
    ] = (
        Path(
            cfg.TRADE_DECISIONS_CSV_PATH
        ).exists()
    )

    return checks


# =============================================================================
# COMPLETE PREDICTOR
# =============================================================================

def run_prediction() -> pd.DataFrame:

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 4 - FINAL DQN TRADE PREDICTION"
    )

    print(
        "=" * 100
    )

    print()

    training_summary = (
        validate_full_training()
    )

    print(
        f"Confirmed training     : "
        f"{training_summary['episodes']} episodes"
    )

    print(
        f"Learning steps         : "
        f"{training_summary['learning_steps']:,}"
    )

    print(
        f"Target updates         : "
        f"{training_summary['target_updates']:,}"
    )

    inference_df = (
        load_inference_data()
    )

    print(
        f"Inference stocks       : "
        f"{len(inference_df)}"
    )

    print(
        f"Prediction date        : "
        f"{inference_df['prediction_date'].iloc[0].date()}"
    )

    agent = (
        load_final_agent()
    )

    predictions = (
        generate_trade_decisions(
            inference_df=inference_df,
            agent=agent,
        )
    )

    save_predictions(
        predictions,
        training_summary,
    )

    checks = validate_predictions(
        predictions
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL TRADE DECISIONS"
    )

    print(
        "=" * 100
    )

    print()

    display_columns = [
        "symbol",
        "q_hold",
        "q_buy",
        "q_sell",
        "trade_action",
    ]

    if "agent3_rank" in predictions.columns:

        display_columns.append(
            "agent3_rank"
        )

    if "risk_level" in predictions.columns:

        display_columns.append(
            "risk_level"
        )

    print(
        predictions[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )

    print(
        "PREDICTION VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    for name, passed in checks.items():

        print(
            f"{name:<50}: {passed}"
        )

    overall_pass = all(
        checks.values()
    )

    print()

    print(
        "VALIDATION RESULT: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    print()

    print(
        "Action distribution"
    )

    for action in [
        "BUY",
        "SELL",
        "HOLD",
    ]:

        count = int(
            (
                predictions[
                    "trade_action"
                ]
                ==
                action
            ).sum()
        )

        print(
            f"{action:<10}: {count}"
        )

    print()

    print(
        f"Trade decisions       : "
        f"{cfg.TRADE_DECISIONS_CSV_PATH}"
    )

    print(
        f"Prediction report     : "
        f"{PREDICTION_REPORT_PATH}"
    )

    if not overall_pass:

        failed = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        raise ValueError(
            "Agent-4 prediction validation failed:\n"
            +
            "\n".join(
                failed
            )
        )

    print()
    print(
        "=" * 100
    )

    print(
        "PREDICTOR STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Full training verified : COMPLETE"
    )

    print(
        "Inference data loading : COMPLETE"
    )

    print(
        "DQN model loading      : COMPLETE"
    )

    print(
        "Greedy Q prediction    : COMPLETE"
    )

    print(
        "BUY/SELL/HOLD output   : COMPLETE"
    )

    print(
        "Agent-3 context        : COMPLETE"
    )

    print(
        "Prediction validation  : COMPLETE"
    )

    print()

    print(
        "PREDICTOR STATUS: COMPLETE"
    )

    return predictions


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    run_prediction()