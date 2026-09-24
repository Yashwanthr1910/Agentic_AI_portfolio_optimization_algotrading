"""
Agent 3 -> Agent 4 Integration Validator
===============================================================================

Purpose
-------
Validate the interface between:

    Agent 3 - Risk Management
            ↓
    Agent 4 - DQN Trade Execution

Checks
------
- Same symbols
- One row per stock
- Same prediction date
- Agent-3 context preserved
- No missing Q-values
- Valid BUY / SELL / HOLD actions
- Action == argmax(Q)
- Full 100-episode training
- Agent-4 output generated
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

if str(
    AGENT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            AGENT_ROOT
        ),
    )


from config import config as cfg


# =============================================================================
# OUTPUT
# =============================================================================

VALIDATION_REPORT_PATH = (
    cfg.REPORT_DIR
    /
    "agent3_agent4_validation.json"
)


# =============================================================================
# READERS
# =============================================================================

def read_dataframe(
    parquet_path: Path,
    csv_path: Path,
    name: str,
) -> pd.DataFrame:

    if parquet_path.exists():

        try:

            df = pd.read_parquet(
                parquet_path
            )

            print(
                f"{name:<30}: PARQUET"
            )

            return df

        except Exception as error:

            print(
                f"{name}: parquet unavailable"
            )

            print(
                f"Reason: "
                f"{type(error).__name__}: {error}"
            )

    if csv_path.exists():

        df = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        print(
            f"{name:<30}: CSV"
        )

        return df

    raise FileNotFoundError(
        f"{name} was not found.\n"
        f"Parquet: {parquet_path}\n"
        f"CSV: {csv_path}"
    )


def load_agent3() -> pd.DataFrame:

    df = read_dataframe(
        parquet_path=Path(
            cfg.AGENT3_RISK_PARQUET_PATH
        ),
        csv_path=Path(
            cfg.AGENT3_RISK_CSV_PATH
        ),
        name="Agent-3 output",
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

    return df


def load_agent4() -> pd.DataFrame:

    parquet_path = Path(
        cfg.TRADE_DECISIONS_PARQUET_PATH
    )

    csv_path = Path(
        cfg.TRADE_DECISIONS_CSV_PATH
    )

    df = read_dataframe(
        parquet_path=parquet_path,
        csv_path=csv_path,
        name="Agent-4 output",
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

    return df


def load_training_summary() -> dict[str, Any]:

    path = Path(
        cfg.TRAINING_SUMMARY_PATH
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Training summary not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# =============================================================================
# COMPARISON HELPERS
# =============================================================================

def numeric_columns_match(
    merged: pd.DataFrame,
    column: str,
    atol: float = 1e-8,
) -> bool:

    left_column = (
        f"{column}_agent3"
    )

    right_column = (
        f"{column}_agent4"
    )

    if (
        left_column
        not in merged.columns
        or
        right_column
        not in merged.columns
    ):

        return False

    left = pd.to_numeric(
        merged[
            left_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    right = pd.to_numeric(
        merged[
            right_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    return bool(
        np.allclose(
            left,
            right,
            equal_nan=True,
            atol=atol,
            rtol=1e-7,
        )
    )


def text_columns_match(
    merged: pd.DataFrame,
    column: str,
) -> bool:

    left_column = (
        f"{column}_agent3"
    )

    right_column = (
        f"{column}_agent4"
    )

    if (
        left_column
        not in merged.columns
        or
        right_column
        not in merged.columns
    ):

        return False

    left = (
        merged[
            left_column
        ]
        .fillna(
            ""
        )
        .astype(
            str
        )
        .str.strip()
    )

    right = (
        merged[
            right_column
        ]
        .fillna(
            ""
        )
        .astype(
            str
        )
        .str.strip()
    )

    return bool(
        (
            left
            ==
            right
        )
        .all()
    )


# =============================================================================
# VALIDATOR
# =============================================================================

def validate_agent3_agent4() -> dict[str, Any]:

    print()
    print(
        "=" * 100
    )

    print(
        "AGENT 3 -> AGENT 4 INTEGRATION VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    agent3 = load_agent3()

    agent4 = load_agent4()

    training_summary = (
        load_training_summary()
    )

    print()
    print(
        f"Agent-3 rows          : "
        f"{len(agent3)}"
    )

    print(
        f"Agent-4 rows          : "
        f"{len(agent4)}"
    )

    print(
        f"Agent-3 stocks        : "
        f"{agent3['symbol'].nunique()}"
    )

    print(
        f"Agent-4 stocks        : "
        f"{agent4['symbol'].nunique()}"
    )

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    merged = agent3.merge(
        agent4,
        on="symbol",
        how="outer",
        suffixes=(
            "_agent3",
            "_agent4",
        ),
        indicator=True,
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # Q-values and actions
    # -------------------------------------------------------------------------

    q_matrix = (
        agent4[
            [
                "q_hold",
                "q_buy",
                "q_sell",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    predicted_action_ids = (
        np.argmax(
            q_matrix,
            axis=1,
        )
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    checks: dict[str, bool] = {}

    checks[
        "agent3_not_empty"
    ] = (
        not agent3.empty
    )

    checks[
        "agent4_not_empty"
    ] = (
        not agent4.empty
    )

    checks[
        "agent3_unique_symbols"
    ] = (
        not agent3[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    checks[
        "agent4_unique_symbols"
    ] = (
        not agent4[
            "symbol"
        ]
        .duplicated()
        .any()
    )

    checks[
        "same_row_count"
    ] = (
        len(
            agent3
        )
        ==
        len(
            agent4
        )
    )

    checks[
        "same_symbol_set"
    ] = (
        set(
            agent3[
                "symbol"
            ]
        )
        ==
        set(
            agent4[
                "symbol"
            ]
        )
    )

    checks[
        "one_to_one_merge"
    ] = bool(
        (
            merged[
                "_merge"
            ]
            ==
            "both"
        )
        .all()
    )

    checks[
        "prediction_dates_valid"
    ] = bool(
        agent3[
            "prediction_date"
        ]
        .notna()
        .all()
        and
        agent4[
            "prediction_date"
        ]
        .notna()
        .all()
    )

    if (
        "prediction_date_agent3"
        in merged.columns
        and
        "prediction_date_agent4"
        in merged.columns
    ):

        checks[
            "prediction_dates_match"
        ] = bool(
            (
                pd.to_datetime(
                    merged[
                        "prediction_date_agent3"
                    ]
                )
                ==
                pd.to_datetime(
                    merged[
                        "prediction_date_agent4"
                    ]
                )
            )
            .all()
        )

    else:

        checks[
            "prediction_dates_match"
        ] = False

    checks[
        "final_date_matches_config"
    ] = bool(
        (
            agent4[
                "prediction_date"
            ]
            ==
            pd.Timestamp(
                cfg.FINAL_INFERENCE_DATE
            )
        )
        .all()
    )

    # Agent 2 / Agent 3 numeric context
    for column in [
        "top30_probability",
        "agent2_rank",
        "agent3_score",
        "agent3_rank",
        "risk_adjusted_weight",
    ]:

        if (
            column in agent3.columns
            and
            column in agent4.columns
        ):

            checks[
                f"{column}_preserved"
            ] = numeric_columns_match(
                merged,
                column,
            )

    # Agent 3 text context
    for column in [
        "risk_level",
        "risk_decision",
        "trend_class",
    ]:

        if (
            column in agent3.columns
            and
            column in agent4.columns
        ):

            checks[
                f"{column}_preserved"
            ] = text_columns_match(
                merged,
                column,
            )

    checks[
        "q_values_finite"
    ] = bool(
        np.isfinite(
            q_matrix
        ).all()
    )

    checks[
        "action_ids_valid"
    ] = bool(
        agent4[
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
        agent4[
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
        "dqn_argmax_matches_action"
    ] = bool(
        (
            agent4[
                "action_id"
            ]
            .to_numpy(
                dtype=int
            )
            ==
            predicted_action_ids
        )
        .all()
    )

    action_name_from_id = (
        agent4[
            "action_id"
        ]
        .map(
            cfg.ACTION_NAMES
        )
    )

    checks[
        "action_id_name_consistent"
    ] = bool(
        (
            action_name_from_id
            ==
            agent4[
                "trade_action"
            ]
        )
        .all()
    )

    checks[
        "full_100_episode_training"
    ] = (
        int(
            training_summary.get(
                "episodes",
                0,
            )
        )
        ==
        cfg.EPISODES
    )

    checks[
        "dqn_learning_occurred"
    ] = (
        int(
            training_summary.get(
                "learning_steps",
                0,
            )
        )
        >
        0
    )

    checks[
        "target_network_updated"
    ] = (
        int(
            training_summary.get(
                "target_updates",
                0,
            )
        )
        >
        0
    )

    checks[
        "trade_output_exists"
    ] = (
        Path(
            cfg.TRADE_DECISIONS_CSV_PATH
        ).exists()
    )

    # -------------------------------------------------------------------------
    # Overall
    # -------------------------------------------------------------------------

    overall_pass = all(
        checks.values()
    )

    print()
    print(
        "=" * 100
    )

    print(
        "INTEGRATION CHECKS"
    )

    print(
        "=" * 100
    )

    print()

    for name, passed in checks.items():

        print(
            f"{name:<55}: "
            f"{passed}"
        )

    print()

    print(
        "VALIDATION RESULT: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    # -------------------------------------------------------------------------
    # Action distribution
    # -------------------------------------------------------------------------

    action_counts = (
        agent4[
            "trade_action"
        ]
        .value_counts()
        .to_dict()
    )

    report = {
        "status":
            (
                "PASS"
                if overall_pass
                else
                "FAIL"
            ),

        "integration":
            "Agent 3 -> Agent 4",

        "prediction_date":
            str(
                agent4[
                    "prediction_date"
                ]
                .iloc[
                    0
                ]
                .date()
            ),

        "agent3_rows":
            int(
                len(
                    agent3
                )
            ),

        "agent4_rows":
            int(
                len(
                    agent4
                )
            ),

        "symbols":
            sorted(
                agent4[
                    "symbol"
                ]
                .tolist()
            ),

        "training_episodes":
            int(
                training_summary.get(
                    "episodes",
                    0,
                )
            ),

        "learning_steps":
            int(
                training_summary.get(
                    "learning_steps",
                    0,
                )
            ),

        "target_updates":
            int(
                training_summary.get(
                    "target_updates",
                    0,
                )
            ),

        "action_distribution": {
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

            "HOLD":
                int(
                    action_counts.get(
                        "HOLD",
                        0,
                    )
                ),
        },

        "checks":
            {
                name:
                    bool(
                        passed
                    )
                for name, passed
                in checks.items()
            },

        "trade_decisions":
            str(
                cfg.TRADE_DECISIONS_CSV_PATH
            ),
    }

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        VALIDATION_REPORT_PATH,
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
        f"Validation report     : "
        f"{VALIDATION_REPORT_PATH}"
    )

    if not overall_pass:

        failed = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        raise ValueError(
            "Agent 3 -> Agent 4 integration failed:\n"
            +
            "\n".join(
                f"  - {name}"
                for name
                in failed
            )
        )

    print()
    print(
        "=" * 100
    )

    print(
        "INTEGRATION STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Agent-3 symbol transfer     : COMPLETE"
    )

    print(
        "Agent-3 risk context        : COMPLETE"
    )

    print(
        "DQN training verification  : COMPLETE"
    )

    print(
        "Q-value validation          : COMPLETE"
    )

    print(
        "Action validation           : COMPLETE"
    )

    print(
        "Agent-3 -> Agent-4 pipeline : COMPLETE"
    )

    print()

    print(
        "INTEGRATION STATUS: COMPLETE"
    )

    return {
        "checks":
            checks,

        "report":
            report,

        "agent3":
            agent3,

        "agent4":
            agent4,
    }


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    validate_agent3_agent4()