"""
Agent 4 - Trade Execution Agent
===============================================================================

Purpose
-------
Provide the orchestration layer for Agent 4.

Pipeline
--------
Agent 3 Risk Assessment
        ↓
Agent 4 inference dataset
        ↓
Fully trained DQN
        ↓
Q(HOLD), Q(BUY), Q(SELL)
        ↓
Greedy action selection
        ↓
BUY / SELL / HOLD
        ↓
trade_decisions.csv

This module DOES NOT retrain the DQN.

It verifies:
    - Full 100-episode training exists
    - Final inference data exists
    - Trained DQN weights exist
    - Greedy prediction succeeds
    - Final outputs are valid

The reference paper describes the DQN as the trade-execution agent
following stock selection, trend prediction and risk analysis.
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

PROJECT_ROOT = (
    AGENT_ROOT.parent
)

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

from src.predictor import (
    validate_full_training,
    load_inference_data,
    load_final_agent,
    generate_trade_decisions,
    save_predictions,
    validate_predictions,
)


# =============================================================================
# REPORT PATH
# =============================================================================

AGENT_REPORT_PATH = (
    cfg.REPORT_DIR
    /
    "agent_summary.json"
)


# =============================================================================
# TRADE EXECUTION AGENT
# =============================================================================

class TradeExecutionAgent:
    """
    High-level Agent-4 interface.

    The class performs final inference only.

    It intentionally does not train the DQN so that calling the
    production agent cannot accidentally overwrite the trained model.
    """

    def __init__(
        self,
        verbose: bool = True,
    ) -> None:

        self.verbose = bool(
            verbose
        )

        self.training_summary: dict[
            str,
            Any
        ] | None = None

        self.inference_data: (
            pd.DataFrame
            | None
        ) = None

        self.predictions: (
            pd.DataFrame
            | None
        ) = None

        self.validation_checks: dict[
            str,
            bool
        ] | None = None

        self.dqn_agent = None


    # =========================================================================
    # PREFLIGHT
    # =========================================================================

    def preflight_check(
        self,
    ) -> dict[str, bool]:
        """
        Check all required Agent-4 artifacts before inference.
        """

        checks = {
            "training_summary_exists":
                Path(
                    cfg.TRAINING_SUMMARY_PATH
                ).exists(),

            "dqn_weights_exist":
                Path(
                    cfg.DQN_WEIGHTS_PATH
                ).exists(),

            "dqn_model_exists":
                Path(
                    cfg.DQN_MODEL_PATH
                ).exists(),

            "inference_parquet_or_csv_exists":
                (
                    Path(
                        cfg.DQN_INFERENCE_DATA_PATH
                    ).exists()
                    or
                    Path(
                        cfg.DQN_INFERENCE_DATA_CSV
                    ).exists()
                ),

            "agent3_parquet_or_csv_exists":
                (
                    Path(
                        cfg.AGENT3_RISK_PARQUET_PATH
                    ).exists()
                    or
                    Path(
                        cfg.AGENT3_RISK_CSV_PATH
                    ).exists()
                ),
        }

        return checks


    # =========================================================================
    # RUN
    # =========================================================================

    def run(
        self,
    ) -> pd.DataFrame:
        """
        Execute complete Agent-4 final inference.
        """

        print()
        print(
            "=" * 100
        )

        print(
            "AGENT 4 - TRADE EXECUTION AGENT"
        )

        print(
            "=" * 100
        )

        print()

        # ---------------------------------------------------------------------
        # 1. Preflight checks
        # ---------------------------------------------------------------------

        preflight = (
            self.preflight_check()
        )

        print(
            "Preflight checks"
        )

        print(
            "-" * 100
        )

        for name, passed in preflight.items():

            print(
                f"{name:<55}: "
                f"{passed}"
            )

        if not all(
            preflight.values()
        ):

            failed = [
                name
                for name, passed
                in preflight.items()
                if not passed
            ]

            raise FileNotFoundError(
                "Agent-4 preflight validation failed:\n"
                +
                "\n".join(
                    f"  - {name}"
                    for name
                    in failed
                )
            )

        # ---------------------------------------------------------------------
        # 2. Verify full DQN training
        # ---------------------------------------------------------------------

        self.training_summary = (
            validate_full_training()
        )

        print()
        print(
            f"Training episodes      : "
            f"{self.training_summary['episodes']}"
        )

        print(
            f"Learning steps         : "
            f"{self.training_summary['learning_steps']:,}"
        )

        print(
            f"Target updates         : "
            f"{self.training_summary['target_updates']:,}"
        )

        # ---------------------------------------------------------------------
        # 3. Load final inference data
        # ---------------------------------------------------------------------

        self.inference_data = (
            load_inference_data()
        )

        print(
            f"Inference rows         : "
            f"{len(self.inference_data)}"
        )

        print(
            f"Inference stocks       : "
            f"{self.inference_data['symbol'].nunique()}"
        )

        print(
            f"Prediction date        : "
            f"{self.inference_data['prediction_date'].iloc[0].date()}"
        )

        # ---------------------------------------------------------------------
        # 4. Load trained DQN
        # ---------------------------------------------------------------------

        self.dqn_agent = (
            load_final_agent()
        )

        print(
            "DQN model              : LOADED"
        )

        print(
            f"Evaluation epsilon     : "
            f"{self.dqn_agent.epsilon}"
        )

        # ---------------------------------------------------------------------
        # 5. Generate final actions
        # ---------------------------------------------------------------------

        self.predictions = (
            generate_trade_decisions(
                inference_df=self.inference_data,
                agent=self.dqn_agent,
            )
        )

        # ---------------------------------------------------------------------
        # 6. Save
        # ---------------------------------------------------------------------

        save_predictions(
            predictions=self.predictions,
            training_summary=self.training_summary,
        )

        # ---------------------------------------------------------------------
        # 7. Validate
        # ---------------------------------------------------------------------

        self.validation_checks = (
            validate_predictions(
                self.predictions
            )
        )

        overall_pass = all(
            self.validation_checks.values()
        )

        # ---------------------------------------------------------------------
        # 8. Display
        # ---------------------------------------------------------------------

        print()
        print(
            "=" * 100
        )

        print(
            "AGENT 4 - FINAL DECISIONS"
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

        for optional_column in [
            "agent2_rank",
            "agent3_rank",
            "risk_level",
            "risk_decision",
        ]:

            if (
                optional_column
                in self.predictions.columns
            ):

                display_columns.append(
                    optional_column
                )

        print(
            self.predictions[
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
            "AGENT VALIDATION"
        )

        print(
            "=" * 100
        )

        print()

        for name, passed in (
            self.validation_checks.items()
        ):

            print(
                f"{name:<55}: "
                f"{passed}"
            )

        print()

        print(
            "VALIDATION RESULT: "
            f"{'PASS' if overall_pass else 'FAIL'}"
        )

        # ---------------------------------------------------------------------
        # 9. Save agent report
        # ---------------------------------------------------------------------

        self._save_agent_report(
            preflight=preflight,
            overall_pass=overall_pass,
        )

        if not overall_pass:

            failed = [
                name
                for name, passed
                in self.validation_checks.items()
                if not passed
            ]

            raise ValueError(
                "Agent-4 validation failed:\n"
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
            "TRADE EXECUTION AGENT STATUS"
        )

        print(
            "=" * 100
        )

        print()

        print(
            "Agent-3 input          : COMPLETE"
        )

        print(
            "Training verification  : COMPLETE"
        )

        print(
            "DQN loading            : COMPLETE"
        )

        print(
            "Q-value inference      : COMPLETE"
        )

        print(
            "BUY/SELL/HOLD decision : COMPLETE"
        )

        print(
            "Output persistence     : COMPLETE"
        )

        print(
            "Prediction validation  : COMPLETE"
        )

        print()

        print(
            "AGENT 4 STATUS: COMPLETE"
        )

        return self.predictions


    # =========================================================================
    # REPORT
    # =========================================================================

    def _save_agent_report(
        self,
        preflight: dict[str, bool],
        overall_pass: bool,
    ) -> None:

        if self.predictions is None:

            raise RuntimeError(
                "Predictions do not exist."
            )

        action_counts = (
            self.predictions[
                "trade_action"
            ]
            .value_counts()
            .to_dict()
        )

        report = {
            "agent":
                "Trade Execution Agent",

            "agent_number":
                4,

            "status":
                (
                    "COMPLETE"
                    if overall_pass
                    else
                    "FAILED"
                ),

            "prediction_date":
                str(
                    pd.to_datetime(
                        self.predictions[
                            "prediction_date"
                        ]
                        .iloc[
                            0
                        ]
                    )
                    .date()
                ),

            "stocks":
                int(
                    len(
                        self.predictions
                    )
                ),

            "training": {
                "episodes":
                    int(
                        self.training_summary[
                            "episodes"
                        ]
                    ),

                "learning_steps":
                    int(
                        self.training_summary[
                            "learning_steps"
                        ]
                    ),

                "target_updates":
                    int(
                        self.training_summary[
                            "target_updates"
                        ]
                    ),

                "learning_rate":
                    float(
                        self.training_summary[
                            "learning_rate"
                        ]
                    ),

                "gamma":
                    float(
                        self.training_summary[
                            "gamma"
                        ]
                    ),

                "batch_size":
                    int(
                        self.training_summary[
                            "batch_size"
                        ]
                    ),
            },

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

            "preflight":
                {
                    key:
                        bool(
                            value
                        )
                    for key, value
                    in preflight.items()
                },

            "prediction_validation":
                {
                    key:
                        bool(
                            value
                        )
                    for key, value
                    in self.validation_checks.items()
                },

            "output":
                str(
                    cfg.TRADE_DECISIONS_CSV_PATH
                ),

            "methodology_note":
                (
                    "Agent 4 follows the reference-paper DQN trade "
                    "execution architecture using seven market-state "
                    "inputs, two hidden layers of 64 units, and "
                    "BUY/SELL/HOLD actions. Exact state composition, "
                    "replay capacity, epsilon schedule, target-sync "
                    "frequency and step-level reward formula are "
                    "documented implementation reconstructions."
                ),
        }

        cfg.REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            AGENT_REPORT_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_trade_execution_agent() -> pd.DataFrame:

    agent = TradeExecutionAgent(
        verbose=True
    )

    return agent.run()


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    run_trade_execution_agent()