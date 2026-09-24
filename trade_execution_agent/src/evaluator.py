"""
Agent 4 - Paper-Aligned DQN Evaluator
===============================================================================

Default evaluation:
    2019-2023

This corresponds to the investing/backtesting period described by the
reference paper.

Optional robustness evaluation:
    2024

Usage
-----
Paper backtest:

    python trade_execution_agent/src/evaluator.py

Optional 2024 robustness test:

    from src.evaluator import run_robustness_evaluation
    run_robustness_evaluation()
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
from src.dqn_agent import DQNAgent
from src.trading_environment import TradingEnvironment


# =============================================================================
# LOAD DATA
# =============================================================================

def load_dataset(
    mode: str = "paper_backtest",
) -> pd.DataFrame:

    if mode == "paper_backtest":

        parquet_path = (
            cfg.DQN_INPUT_DATA_PATH
        )

        csv_path = (
            cfg.DQN_INPUT_DATA_CSV
        )

        expected_start = pd.Timestamp(
            cfg.EVALUATION_START_DATE
        )

        expected_end = pd.Timestamp(
            cfg.EVALUATION_END_DATE
        )

    elif mode == "robustness":

        parquet_path = (
            cfg.DQN_ROBUSTNESS_DATA_PATH
        )

        csv_path = (
            cfg.DQN_ROBUSTNESS_DATA_CSV
        )

        expected_start = pd.Timestamp(
            cfg.ROBUSTNESS_START_DATE
        )

        expected_end = pd.Timestamp(
            cfg.ROBUSTNESS_END_DATE
        )

    else:

        raise ValueError(
            f"Unknown evaluation mode: {mode}"
        )

    if parquet_path.exists():

        try:

            df = pd.read_parquet(
                parquet_path
            )

            source_format = (
                "PARQUET"
            )

        except Exception:

            df = pd.read_csv(
                csv_path,
                low_memory=False,
            )

            source_format = (
                "CSV"
            )

    elif csv_path.exists():

        df = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        source_format = (
            "CSV"
        )

    else:

        raise FileNotFoundError(
            f"Evaluation dataset not found for "
            f"mode={mode}"
        )

    df = df.copy()

    df[
        "date"
    ] = pd.to_datetime(
        df[
            "date"
        ],
        errors="coerce",
    )

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

    if df.empty:

        raise ValueError(
            "Evaluation dataset is empty."
        )

    if df[
        "date"
    ].min() < expected_start:

        raise ValueError(
            "Evaluation data starts before expected period."
        )

    if df[
        "date"
    ].max() > expected_end:

        raise ValueError(
            "Evaluation data extends beyond expected period."
        )

    if not np.isfinite(
        df[
            cfg.STATE_FEATURES
        ]
        .to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Evaluation state contains NaN/inf."
        )

    print(
        f"Evaluation mode        : {mode}"
    )

    print(
        f"Evaluation format      : {source_format}"
    )

    return (
        df
        .sort_values(
            [
                "symbol",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# ENVIRONMENT HELPERS
# =============================================================================

def create_environment(
    stock_df: pd.DataFrame,
) -> TradingEnvironment:

    try:

        return TradingEnvironment(
            stock_df.copy()
        )

    except TypeError as original_error:

        signature = inspect.signature(
            TradingEnvironment
        )

        for name in [
            "data",
            "stock_data",
            "market_data",
            "df",
        ]:

            if name in signature.parameters:

                return TradingEnvironment(
                    **{
                        name:
                            stock_df.copy()
                    }
                )

        raise original_error


def reset_environment(
    environment: TradingEnvironment,
) -> np.ndarray:

    result = environment.reset()

    if (
        isinstance(
            result,
            tuple,
        )
        and
        len(
            result
        )
        ==
        2
        and
        isinstance(
            result[
                1
            ],
            dict,
        )
    ):

        result = result[
            0
        ]

    return np.asarray(
        result,
        dtype=np.float32,
    )


def step_environment(
    environment: TradingEnvironment,
    action: int,
):

    result = environment.step(
        action
    )

    if len(
        result
    ) == 4:

        next_state, reward, done, info = (
            result
        )

    elif len(
        result
    ) == 5:

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = result

        done = bool(
            terminated
            or
            truncated
        )

    else:

        raise ValueError(
            "Unexpected environment.step output."
        )

    return (
        np.asarray(
            next_state,
            dtype=np.float32,
        ),

        float(
            reward
        ),

        bool(
            done
        ),

        {}
        if info is None
        else dict(
            info
        ),
    )


def get_portfolio_state(
    environment: TradingEnvironment,
) -> dict[str, Any]:

    if hasattr(
        environment,
        "get_portfolio_state",
    ):

        result = (
            environment
            .get_portfolio_state()
        )

        if isinstance(
            result,
            dict,
        ):

            return result

    return {}


def get_trade_count(
    state: dict[str, Any],
) -> int:

    for key in [
        "total_trades",
        "trade_count",
        "executed_trades",
    ]:

        if key in state:

            try:

                return int(
                    state[
                        key
                    ]
                )

            except Exception:

                pass

    return 0


# =============================================================================
# LOAD TRAINED DQN
# =============================================================================

def load_trained_agent() -> DQNAgent:

    if not Path(
        cfg.DQN_WEIGHTS_PATH
    ).exists():

        raise FileNotFoundError(
            "DQN weights missing. "
            "Run full training first."
        )

    agent = DQNAgent()

    agent.load_weights(
        cfg.DQN_WEIGHTS_PATH,
        sync_target=True,
    )

    agent.set_evaluation_mode()

    return agent


# =============================================================================
# SINGLE STOCK EVALUATION
# =============================================================================

def evaluate_stock(
    agent: DQNAgent,
    stock_df: pd.DataFrame,
    symbol: str,
):

    environment = create_environment(
        stock_df
    )

    state = reset_environment(
        environment
    )

    initial_portfolio = get_portfolio_state(
        environment
    )

    initial_value = float(
        initial_portfolio.get(
            "portfolio_value",
            cfg.INITIAL_CASH,
        )
    )

    initial_trade_count = get_trade_count(
        initial_portfolio
    )

    done = False
    steps = 0
    total_reward = 0.0

    hold_actions = 0
    buy_actions = 0
    sell_actions = 0

    q_hold = []
    q_buy = []
    q_sell = []

    while not done:

        q_values = agent.predict_q_values(
            state
        )

        action = int(
            np.argmax(
                q_values
            )
        )

        if action == cfg.ACTION_HOLD:

            hold_actions += 1

        elif action == cfg.ACTION_BUY:

            buy_actions += 1

        elif action == cfg.ACTION_SELL:

            sell_actions += 1

        (
            next_state,
            reward,
            done,
            info,
        ) = step_environment(
            environment,
            action,
        )

        total_reward += reward
        steps += 1

        q_hold.append(
            float(
                q_values[
                    cfg.ACTION_HOLD
                ]
            )
        )

        q_buy.append(
            float(
                q_values[
                    cfg.ACTION_BUY
                ]
            )
        )

        q_sell.append(
            float(
                q_values[
                    cfg.ACTION_SELL
                ]
            )
        )

        state = (
            next_state
        )

    final_portfolio = get_portfolio_state(
        environment
    )

    final_value = float(
        final_portfolio.get(
            "portfolio_value",
            np.nan,
        )
    )

    final_trade_count = get_trade_count(
        final_portfolio
    )

    executed_trades = max(
        0,
        final_trade_count
        -
        initial_trade_count,
    )

    if (
        np.isfinite(
            initial_value
        )
        and
        initial_value
        >
        0
        and
        np.isfinite(
            final_value
        )
    ):

        portfolio_return = (
            final_value
            /
            initial_value
            -
            1.0
        )

    else:

        portfolio_return = (
            np.nan
        )

    max_drawdown = float(
        final_portfolio.get(
            "max_drawdown",
            np.nan,
        )
    )

    transaction_costs = float(
        final_portfolio.get(
            "transaction_costs",
            final_portfolio.get(
                "total_transaction_costs",
                np.nan,
            ),
        )
    )

    return {
        "symbol":
            symbol,

        "start_date":
            str(
                stock_df[
                    "date"
                ]
                .min()
                .date()
            ),

        "end_date":
            str(
                stock_df[
                    "date"
                ]
                .max()
                .date()
            ),

        "steps":
            steps,

        "initial_portfolio_value":
            initial_value,

        "final_portfolio_value":
            final_value,

        "portfolio_return":
            portfolio_return,

        "portfolio_return_percent":
            portfolio_return
            *
            100.0,

        "total_reward":
            total_reward,

        "hold_actions":
            hold_actions,

        "buy_actions":
            buy_actions,

        "sell_actions":
            sell_actions,

        "executed_trades":
            executed_trades,

        "max_drawdown":
            max_drawdown,

        "max_drawdown_percent":
            max_drawdown
            *
            100.0,

        "transaction_costs":
            transaction_costs,

        "mean_q_hold":
            float(
                np.mean(
                    q_hold
                )
            ),

        "mean_q_buy":
            float(
                np.mean(
                    q_buy
                )
            ),

        "mean_q_sell":
            float(
                np.mean(
                    q_sell
                )
            ),
    }


# =============================================================================
# EVALUATOR
# =============================================================================

class DQNEvaluator:

    def __init__(
        self,
        mode: str = "paper_backtest",
    ):

        self.mode = (
            mode
        )

        self.data = load_dataset(
            mode
        )

        self.agent = (
            load_trained_agent()
        )

        self.symbols = sorted(
            self.data[
                "symbol"
            ]
            .unique()
            .tolist()
        )


    def evaluate(
        self,
    ) -> pd.DataFrame:

        print()
        print(
            "=" * 100
        )

        if self.mode == "paper_backtest":

            print(
                "AGENT 4 - PAPER BACKTEST EVALUATION"
            )

        else:

            print(
                "AGENT 4 - 2024 ROBUSTNESS EVALUATION"
            )

        print(
            "=" * 100
        )

        print()

        print(
            f"Date range            : "
            f"{self.data['date'].min().date()} "
            f"-> "
            f"{self.data['date'].max().date()}"
        )

        print(
            f"Stocks                : "
            f"{len(self.symbols)}"
        )

        print(
            "Policy                : GREEDY"
        )

        print(
            "Exploration           : 0.0"
        )

        results = []

        for index, symbol in enumerate(
            self.symbols,
            start=1,
        ):

            stock_df = (
                self.data[
                    self.data[
                        "symbol"
                    ]
                    ==
                    symbol
                ]
                .sort_values(
                    "date"
                )
                .reset_index(
                    drop=True
                )
            )

            result = evaluate_stock(
                self.agent,
                stock_df,
                symbol,
            )

            results.append(
                result
            )

            print(
                f"{index:02d}/{len(self.symbols):02d} | "
                f"{symbol:<15} | "
                f"return="
                f"{result['portfolio_return_percent']:8.3f}% | "
                f"maxDD="
                f"{result['max_drawdown_percent']:8.3f}% | "
                f"trades="
                f"{result['executed_trades']:4d}"
            )

        return pd.DataFrame(
            results
        )


# =============================================================================
# REPORT + VALIDATION
# =============================================================================

def save_and_validate(
    evaluator: DQNEvaluator,
    results: pd.DataFrame,
):

    if evaluator.mode == "paper_backtest":

        report_path = (
            cfg.EVALUATION_REPORT_PATH
        )

        results_path = (
            cfg.REPORT_DIR
            /
            "evaluation_stock_results.csv"
        )

    else:

        report_path = (
            cfg.ROBUSTNESS_REPORT_PATH
        )

        results_path = (
            cfg.REPORT_DIR
            /
            "robustness_stock_results.csv"
        )

    results.to_csv(
        results_path,
        index=False,
    )

    changed_value = (
        results[
            "portfolio_return"
        ]
        .abs()
        >
        1e-10
    )

    trade_count_consistent = bool(
        (
            (~changed_value)
            |
            (
                results[
                    "executed_trades"
                ]
                >
                0
            )
        )
        .all()
    )

    checks = {
        "results_not_empty":
            not results.empty,

        "all_symbols_evaluated":
            set(
                results[
                    "symbol"
                ]
            )
            ==
            set(
                evaluator.symbols
            ),

        "returns_finite":
            bool(
                np.isfinite(
                    results[
                        "portfolio_return"
                    ]
                ).all()
            ),

        "drawdown_nonpositive":
            bool(
                (
                    results[
                        "max_drawdown"
                    ]
                    <=
                    cfg.FLOAT_TOLERANCE
                )
                .all()
            ),

        "action_counts_equal_steps":
            bool(
                (
                    results[
                        "hold_actions"
                    ]
                    +
                    results[
                        "buy_actions"
                    ]
                    +
                    results[
                        "sell_actions"
                    ]
                    ==
                    results[
                        "steps"
                    ]
                )
                .all()
            ),

        "trade_counter_consistent":
            trade_count_consistent,

        "greedy_policy":
            evaluator.agent.epsilon
            ==
            0.0,
    }

    summary = {
        "status":
            (
                "PASS"
                if all(
                    checks.values()
                )
                else
                "FAIL"
            ),

        "mode":
            evaluator.mode,

        "start":
            str(
                evaluator.data[
                    "date"
                ]
                .min()
                .date()
            ),

        "end":
            str(
                evaluator.data[
                    "date"
                ]
                .max()
                .date()
            ),

        "stocks":
            len(
                results
            ),

        "mean_return_percent":
            float(
                results[
                    "portfolio_return_percent"
                ].mean()
            ),

        "median_return_percent":
            float(
                results[
                    "portfolio_return_percent"
                ].median()
            ),

        "positive_stocks":
            int(
                (
                    results[
                        "portfolio_return"
                    ]
                    >
                    0
                )
                .sum()
            ),

        "negative_stocks":
            int(
                (
                    results[
                        "portfolio_return"
                    ]
                    <
                    0
                )
                .sum()
            ),

        "mean_max_drawdown_percent":
            float(
                results[
                    "max_drawdown_percent"
                ].mean()
            ),

        "total_executed_trades":
            int(
                results[
                    "executed_trades"
                ].sum()
            ),

        "checks":
            checks,

        "results_path":
            str(
                results_path
            ),

        "methodology_note":
            (
                "2019-2023 is treated as the paper-aligned "
                "investing/backtesting period. 2024 is retained "
                "only as an additional robustness test."
            ),
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print()
    print(
        "=" * 100
    )

    print(
        "EVALUATION VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    for name, passed in checks.items():

        print(
            f"{name:<50}: {passed}"
        )

    overall = all(
        checks.values()
    )

    print()

    print(
        "VALIDATION RESULT: "
        f"{'PASS' if overall else 'FAIL'}"
    )

    print()

    print(
        f"Mean return           : "
        f"{results['portfolio_return_percent'].mean():.4f}%"
    )

    print(
        f"Median return         : "
        f"{results['portfolio_return_percent'].median():.4f}%"
    )

    print(
        f"Executed trades       : "
        f"{int(results['executed_trades'].sum()):,}"
    )

    if not overall:

        raise ValueError(
            "Agent-4 evaluation validation failed."
        )

    return {
        "results":
            results,

        "checks":
            checks,

        "summary":
            summary,
    }


# =============================================================================
# PAPER BACKTEST
# =============================================================================

def run_evaluation():

    evaluator = DQNEvaluator(
        mode="paper_backtest"
    )

    results = (
        evaluator
        .evaluate()
    )

    output = save_and_validate(
        evaluator,
        results,
    )

    print()
    print(
        "PAPER BACKTEST STATUS: COMPLETE"
    )

    return output


# =============================================================================
# OPTIONAL 2024 TEST
# =============================================================================

def run_robustness_evaluation():

    evaluator = DQNEvaluator(
        mode="robustness"
    )

    results = (
        evaluator
        .evaluate()
    )

    output = save_and_validate(
        evaluator,
        results,
    )

    print()
    print(
        "ROBUSTNESS EVALUATION STATUS: COMPLETE"
    )

    return output


if __name__ == "__main__":

    run_evaluation()