"""
Agent 4 - Paper-Aligned DQN Trainer
===============================================================================

Training period:
    2015-2018

This approximates the paper's historical-calculation period because the
current project dataset does not contain 2000-2014.

Paper DQN settings:
    learning rate = 0.0001
    gamma         = 0.95
    batch size    = 32
    episodes      = 100
    hidden layers = 64 -> 64
    actions       = HOLD / BUY / SELL

Direct execution runs only the smoke test.

Full training:
    run_full_training()
"""

from __future__ import annotations

import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# IMPORT PATH
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
from src.dqn_agent import DQNAgent
from src.trading_environment import TradingEnvironment


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_training_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )


# =============================================================================
# LOAD TRAINING DATA
# =============================================================================

def load_training_data() -> pd.DataFrame:

    if cfg.DQN_TRAINING_DATA_PATH.exists():

        try:

            df = pd.read_parquet(
                cfg.DQN_TRAINING_DATA_PATH
            )

            print(
                "Training data format   : PARQUET"
            )

        except Exception:

            df = pd.read_csv(
                cfg.DQN_TRAINING_DATA_CSV,
                low_memory=False,
            )

            print(
                "Training data format   : CSV"
            )

    elif cfg.DQN_TRAINING_DATA_CSV.exists():

        df = pd.read_csv(
            cfg.DQN_TRAINING_DATA_CSV,
            low_memory=False,
        )

        print(
            "Training data format   : CSV"
        )

    else:

        raise FileNotFoundError(
            "DQN training dataset not found. "
            "Run data_loader.py first."
        )

    if df.empty:

        raise ValueError(
            "DQN training data is empty."
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

    required_columns = {
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
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
            f"Training columns missing: "
            f"{sorted(missing)}"
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

    if not np.isfinite(
        df[
            cfg.STATE_FEATURES
        ].to_numpy(
            dtype=float
        )
    ).all():

        raise ValueError(
            "Training state contains NaN/inf."
        )

    if df[
        "date"
    ].min() < pd.Timestamp(
        cfg.TRAIN_START_DATE
    ):

        raise ValueError(
            "Training data starts before configured training period."
        )

    if df[
        "date"
    ].max() > pd.Timestamp(
        cfg.TRAIN_END_DATE
    ):

        raise ValueError(
            "Training data extends into paper backtest period."
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
    portfolio_state: dict[str, Any],
) -> int:
    """
    Read the actual environment trade counter.
    """

    for key in [
        "total_trades",
        "trade_count",
        "executed_trades",
    ]:

        if key in portfolio_state:

            try:

                return int(
                    portfolio_state[
                        key
                    ]
                )

            except Exception:

                pass

    return 0


# =============================================================================
# TRAINER
# =============================================================================

class DQNTrainer:

    def __init__(
        self,
        training_data: pd.DataFrame | None = None,
        agent: DQNAgent | None = None,
        episodes: int | None = None,
        max_steps_per_episode: int | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ):

        self.seed = (
            cfg.RANDOM_SEED
            if seed is None
            else int(
                seed
            )
        )

        set_training_seed(
            self.seed
        )

        self.rng = np.random.default_rng(
            self.seed
        )

        self.training_data = (
            load_training_data()
            if training_data is None
            else training_data.copy()
        )

        self.agent = (
            DQNAgent()
            if agent is None
            else agent
        )

        self.episodes = (
            cfg.EPISODES
            if episodes is None
            else int(
                episodes
            )
        )

        self.max_steps_per_episode = (
            None
            if max_steps_per_episode is None
            else int(
                max_steps_per_episode
            )
        )

        self.verbose = bool(
            verbose
        )

        self.symbols = sorted(
            self.training_data[
                "symbol"
            ]
            .unique()
            .tolist()
        )

        self.stock_data = {
            symbol:
                (
                    self.training_data[
                        self.training_data[
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
            for symbol
            in self.symbols
        }

        self.history = []


    def build_episode_sequence(
        self,
    ) -> list[str]:

        sequence = []

        while len(
            sequence
        ) < self.episodes:

            cycle = (
                self.symbols.copy()
            )

            self.rng.shuffle(
                cycle
            )

            sequence.extend(
                cycle
            )

        return sequence[
            :self.episodes
        ]


    def train_episode(
        self,
        episode_number: int,
        symbol: str,
    ) -> dict[str, Any]:

        environment = create_environment(
            self.stock_data[
                symbol
            ]
        )

        state = reset_environment(
            environment
        )

        initial_portfolio = get_portfolio_state(
            environment
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

        losses = []

        start_time = time.perf_counter()

        while not done:

            action = self.agent.select_action(
                state,
                training=True,
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
                environment_done,
                info,
            ) = step_environment(
                environment,
                action,
            )

            steps += 1

            forced_done = (
                self.max_steps_per_episode
                is not None
                and
                steps
                >=
                self.max_steps_per_episode
            )

            done = bool(
                environment_done
                or
                forced_done
            )

            self.agent.remember(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            training_result = (
                self.agent
                .train_step()
            )

            if training_result is not None:

                losses.append(
                    float(
                        training_result[
                            "loss"
                        ]
                    )
                )

            total_reward += reward

            state = (
                next_state
            )

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        final_portfolio = get_portfolio_state(
            environment
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

        final_value = float(
            final_portfolio.get(
                "portfolio_value",
                np.nan,
            )
        )

        max_drawdown = float(
            final_portfolio.get(
                "max_drawdown",
                np.nan,
            )
        )

        result = {
            "episode":
                episode_number,

            "symbol":
                symbol,

            "steps":
                steps,

            "total_reward":
                total_reward,

            "mean_loss":
                (
                    float(
                        np.mean(
                            losses
                        )
                    )
                    if losses
                    else np.nan
                ),

            "epsilon":
                float(
                    self.agent.epsilon
                ),

            "replay_size":
                len(
                    self.agent.replay_buffer
                ),

            "learning_steps":
                self.agent.learning_steps,

            "target_updates":
                self.agent.target_updates,

            "hold_actions":
                hold_actions,

            "buy_actions":
                buy_actions,

            "sell_actions":
                sell_actions,

            "executed_trades":
                executed_trades,

            "final_portfolio_value":
                final_value,

            "max_drawdown":
                max_drawdown,

            "episode_seconds":
                elapsed,
        }

        return result


    def train(
        self,
        save_model: bool = True,
        save_reports: bool = True,
    ) -> pd.DataFrame:

        print()
        print(
            "=" * 100
        )

        print(
            "AGENT 4 - PAPER-ALIGNED DQN TRAINING"
        )

        print(
            "=" * 100
        )

        print()

        print(
            f"Training period       : "
            f"{self.training_data['date'].min().date()} "
            f"-> "
            f"{self.training_data['date'].max().date()}"
        )

        print(
            f"Training rows         : "
            f"{len(self.training_data):,}"
        )

        print(
            f"Stocks                : "
            f"{len(self.symbols)}"
        )

        print(
            f"Episodes              : "
            f"{self.episodes}"
        )

        print(
            f"Learning rate         : "
            f"{cfg.LEARNING_RATE}"
        )

        print(
            f"Gamma                 : "
            f"{cfg.GAMMA}"
        )

        print(
            f"Batch size            : "
            f"{cfg.BATCH_SIZE}"
        )

        print()

        start = time.perf_counter()

        sequence = self.build_episode_sequence()

        for index, symbol in enumerate(
            sequence,
            start=1,
        ):

            result = self.train_episode(
                index,
                symbol,
            )

            self.history.append(
                result
            )

            if self.verbose:

                loss_text = (
                    "N/A"
                    if pd.isna(
                        result[
                            "mean_loss"
                        ]
                    )
                    else
                    f"{result['mean_loss']:.6f}"
                )

                print(
                    f"Episode "
                    f"{index:03d}/{self.episodes:03d} | "
                    f"{symbol:<15} | "
                    f"steps={result['steps']:4d} | "
                    f"reward={result['total_reward']: .6f} | "
                    f"loss={loss_text:<10} | "
                    f"epsilon={result['epsilon']:.4f} | "
                    f"trades={result['executed_trades']:4d}"
                )

        duration = (
            time.perf_counter()
            -
            start
        )

        history_df = pd.DataFrame(
            self.history
        )

        if save_model:

            self.agent.save()

        if save_reports:

            cfg.REPORT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            history_df.to_csv(
                cfg.TRAINING_HISTORY_PATH,
                index=False,
            )

            valid_losses = (
                history_df[
                    "mean_loss"
                ]
                .dropna()
            )

            summary = {
                "status":
                    "COMPLETE",

                "training_start":
                    str(
                        self.training_data[
                            "date"
                        ]
                        .min()
                        .date()
                    ),

                "training_end":
                    str(
                        self.training_data[
                            "date"
                        ]
                        .max()
                        .date()
                    ),

                "rows":
                    len(
                        self.training_data
                    ),

                "stocks":
                    len(
                        self.symbols
                    ),

                "episodes":
                    self.episodes,

                "learning_rate":
                    cfg.LEARNING_RATE,

                "gamma":
                    cfg.GAMMA,

                "batch_size":
                    cfg.BATCH_SIZE,

                "learning_steps":
                    self.agent.learning_steps,

                "target_updates":
                    self.agent.target_updates,

                "epsilon_final":
                    self.agent.epsilon,

                "total_environment_steps":
                    int(
                        history_df[
                            "steps"
                        ].sum()
                    ),

                "total_executed_trades":
                    int(
                        history_df[
                            "executed_trades"
                        ].sum()
                    ),

                "mean_reward":
                    float(
                        history_df[
                            "total_reward"
                        ].mean()
                    ),

                "mean_loss":
                    (
                        float(
                            valid_losses.mean()
                        )
                        if not valid_losses.empty
                        else None
                    ),

                "duration_seconds":
                    duration,

                "methodology_note":
                    (
                        "2015-2018 is used as the historical DQN "
                        "training period because the available source "
                        "data begins in 2015. The reference paper uses "
                        "2000-2018 historical calculations."
                    ),
            }

            with open(
                cfg.TRAINING_SUMMARY_PATH,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    summary,
                    file,
                    indent=4,
                )

        return history_df


# =============================================================================
# SMOKE TEST
# =============================================================================

def run_trainer_smoke_test():

    training_data = (
        load_training_data()
    )

    trainer = DQNTrainer(
        training_data=training_data,
        episodes=2,
        max_steps_per_episode=600,
        seed=cfg.RANDOM_SEED,
        verbose=True,
    )

    history = trainer.train(
        save_model=False,
        save_reports=False,
    )

    checks = {
        "episodes_complete":
            len(
                history
            )
            ==
            2,

        "replay_ready":
            len(
                trainer.agent.replay_buffer
            )
            >=
            cfg.MIN_REPLAY_SIZE,

        "learning_occurred":
            trainer.agent.learning_steps
            >
            0,

        "epsilon_reduced":
            trainer.agent.epsilon
            <
            cfg.EPSILON_START,

        "rewards_finite":
            bool(
                np.isfinite(
                    history[
                        "total_reward"
                    ]
                ).all()
            ),
    }

    print()
    print(
        "=" * 100
    )

    print(
        "TRAINER VALIDATION"
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

    if not overall:

        raise ValueError(
            "Trainer smoke test failed."
        )

    print()
    print(
        "TRAINER STATUS: COMPLETE"
    )

    return history


# =============================================================================
# FULL PAPER-ALIGNED TRAINING
# =============================================================================

def run_full_training():

    trainer = DQNTrainer(
        episodes=cfg.EPISODES,
        max_steps_per_episode=None,
        seed=cfg.RANDOM_SEED,
        verbose=True,
    )

    history = trainer.train(
        save_model=True,
        save_reports=True,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FULL DQN TRAINING COMPLETE"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Episodes             : "
        f"{len(history)}"
    )

    print(
        f"Learning steps       : "
        f"{trainer.agent.learning_steps:,}"
    )

    print(
        f"Target updates       : "
        f"{trainer.agent.target_updates:,}"
    )

    print(
        f"Final epsilon        : "
        f"{trainer.agent.epsilon:.6f}"
    )

    return history


if __name__ == "__main__":

    run_trainer_smoke_test()