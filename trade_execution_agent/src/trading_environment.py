"""
Agent 4 - DQN Trading Environment
======================================================================

Purpose
-------
Provide a sequential reinforcement-learning environment for the
Agent-4 Deep Q-Network.

At every step:

    S_t
      |
      v
    action
      |
      v
    portfolio transition
      |
      v
    reward
      |
      v
    S_(t+1)

Actions
-------
0 = HOLD
1 = BUY
2 = SELL

Paper alignment
---------------
The paper defines three trading operations:

    Buy
    Sell
    Hold

and describes the environment transition using:

    (S_t, A_t, R_t, S_(t+1))

The paper also states that maximum drawdown is used in the reward
formulation.

Implementation choices
----------------------
The exact portfolio accounting, transaction-cost treatment,
trade fraction, and step-wise reward equation are not fully
specified in the paper.

This implementation therefore uses:

    reward =
        portfolio_return
        -
        drawdown_penalty

while explicitly tracking maximum drawdown.

No short selling is used by default.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ======================================================================
# PROJECT IMPORT PATH
# ======================================================================

AGENT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(AGENT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(AGENT_ROOT),
    )


from config import config as cfg


# ======================================================================
# DISPLAY
# ======================================================================

SEPARATOR = "=" * 100


def print_header(
    title: str,
) -> None:

    print()

    print(
        SEPARATOR
    )

    print(
        title
    )

    print(
        SEPARATOR
    )

    print()


# ======================================================================
# ENVIRONMENT
# ======================================================================

class TradingEnvironment:
    """
    Single-stock sequential trading environment.

    One environment instance represents one stock trajectory.

    State
    -----
    Seven market-derived features defined in cfg.STATE_FEATURES.

    Actions
    -------
    0 -> HOLD
    1 -> BUY
    2 -> SELL

    Portfolio state
    ---------------
    cash
    shares
    portfolio value
    peak portfolio value
    current drawdown
    maximum drawdown
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float | None = None,
        transaction_cost_rate: float | None = None,
        trade_fraction: float | None = None,
        max_position_fraction: float | None = None,
        allow_short_selling: bool | None = None,
        allow_fractional_shares: bool | None = None,
    ) -> None:

        self.data = (
            data
            .copy()
            .sort_values(
                "date"
            )
            .reset_index(
                drop=True
            )
        )

        self.initial_cash = float(
            (
                cfg.INITIAL_CASH
                if initial_cash is None
                else initial_cash
            )
        )

        self.transaction_cost_rate = float(
            (
                cfg.TRANSACTION_COST_RATE
                if transaction_cost_rate is None
                else transaction_cost_rate
            )
        )

        self.trade_fraction = float(
            (
                cfg.DEFAULT_TRADE_FRACTION
                if trade_fraction is None
                else trade_fraction
            )
        )

        self.max_position_fraction = float(
            (
                cfg.MAX_POSITION_FRACTION
                if max_position_fraction is None
                else max_position_fraction
            )
        )

        self.allow_short_selling = bool(
            (
                cfg.ALLOW_SHORT_SELLING
                if allow_short_selling is None
                else allow_short_selling
            )
        )

        self.allow_fractional_shares = bool(
            (
                cfg.ALLOW_FRACTIONAL_SHARES
                if allow_fractional_shares is None
                else allow_fractional_shares
            )
        )

        self._validate_data()

        self.symbol = str(
            self.data[
                "symbol"
            ].iloc[
                0
            ]
        )

        self.reset()


    # ==================================================================
    # VALIDATION
    # ==================================================================

    def _validate_data(
        self,
    ) -> None:

        required_columns = [

            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",

            *cfg.STATE_FEATURES,

        ]

        missing_columns = [

            column
            for column in required_columns
            if column not in self.data.columns

        ]

        if missing_columns:

            raise ValueError(
                "TradingEnvironment input is missing columns:\n"
                +
                "\n".join(
                    f"  - {column}"
                    for column in missing_columns
                )
            )

        if self.data.empty:

            raise ValueError(
                "TradingEnvironment received an empty dataset."
            )

        if (
            self.data[
                "symbol"
            ]
            .nunique()
            !=
            1
        ):

            raise ValueError(
                "TradingEnvironment expects data for exactly one stock."
            )

        if (
            self.data[
                cfg.STATE_FEATURES
            ]
            .isna()
            .any()
            .any()
        ):

            raise ValueError(
                "State features contain missing values."
            )

        if (
            self.data[
                "close"
            ]
            <=
            0
        ).any():

            raise ValueError(
                "Close prices must be strictly positive."
            )

        if len(
            self.data
        ) < 2:

            raise ValueError(
                "TradingEnvironment requires at least two observations."
            )


    # ==================================================================
    # RESET
    # ==================================================================

    def reset(
        self,
    ) -> np.ndarray:
        """
        Reset environment to the first state.
        """

        self.current_step = 0

        self.cash = float(
            self.initial_cash
        )

        self.shares = 0.0

        self.total_transaction_cost = 0.0

        self.total_trades = 0

        self.buy_count = 0

        self.sell_count = 0

        self.hold_count = 0

        self.portfolio_value = float(
            self.initial_cash
        )

        self.previous_portfolio_value = float(
            self.initial_cash
        )

        self.peak_portfolio_value = float(
            self.initial_cash
        )

        self.current_drawdown = 0.0

        self.max_drawdown = 0.0

        self.done = False

        self.history: list[dict[str, Any]] = []

        return self._get_state()


    # ==================================================================
    # STATE
    # ==================================================================

    def _get_state(
        self,
    ) -> np.ndarray:
        """
        Return current DQN state vector.
        """

        row = self.data.iloc[
            self.current_step
        ]

        state = (
            row[
                cfg.STATE_FEATURES
            ]
            .astype(
                np.float32
            )
            .to_numpy()
        )

        if len(
            state
        ) != cfg.STATE_SIZE:

            raise ValueError(
                f"State size mismatch. "
                f"Expected {cfg.STATE_SIZE}, "
                f"got {len(state)}."
            )

        if not np.isfinite(
            state
        ).all():

            raise ValueError(
                "Non-finite value detected in state."
            )

        return state


    # ==================================================================
    # PRICE
    # ==================================================================

    def _current_price(
        self,
    ) -> float:

        return float(
            self.data[
                "close"
            ].iloc[
                self.current_step
            ]
        )


    def _next_price(
        self,
    ) -> float:

        next_index = min(
            self.current_step + 1,
            len(
                self.data
            )
            -
            1,
        )

        return float(
            self.data[
                "close"
            ].iloc[
                next_index
            ]
        )


    # ==================================================================
    # PORTFOLIO VALUE
    # ==================================================================

    def _calculate_portfolio_value(
        self,
        price: float,
    ) -> float:

        return float(
            self.cash
            +
            self.shares
            *
            price
        )


    # ==================================================================
    # BUY
    # ==================================================================

    def _execute_buy(
        self,
        price: float,
    ) -> dict:
        """
        Buy using a fraction of available cash while respecting
        the maximum position constraint.
        """

        current_portfolio_value = (
            self._calculate_portfolio_value(
                price
            )
        )

        current_position_value = (
            self.shares
            *
            price
        )

        maximum_position_value = (
            current_portfolio_value
            *
            self.max_position_fraction
        )

        available_position_capacity = max(
            0.0,
            maximum_position_value
            -
            current_position_value,
        )

        cash_budget = (
            self.cash
            *
            self.trade_fraction
        )

        trade_budget = min(
            cash_budget,
            available_position_capacity,
        )

        if trade_budget <= 0:

            return {
                "executed": False,
                "shares": 0.0,
                "gross_value": 0.0,
                "transaction_cost": 0.0,
            }

        effective_cost_per_share = (
            price
            *
            (
                1.0
                +
                self.transaction_cost_rate
                +
                cfg.SLIPPAGE_RATE
            )
        )

        shares_to_buy = (
            trade_budget
            /
            effective_cost_per_share
        )

        if not self.allow_fractional_shares:

            shares_to_buy = float(
                np.floor(
                    shares_to_buy
                )
            )

        if shares_to_buy <= 0:

            return {
                "executed": False,
                "shares": 0.0,
                "gross_value": 0.0,
                "transaction_cost": 0.0,
            }

        gross_value = (
            shares_to_buy
            *
            price
        )

        transaction_cost = (
            gross_value
            *
            self.transaction_cost_rate
        )

        slippage_cost = (
            gross_value
            *
            cfg.SLIPPAGE_RATE
        )

        total_cost = (
            gross_value
            +
            transaction_cost
            +
            slippage_cost
        )

        if total_cost > self.cash:

            shares_to_buy = (
                self.cash
                /
                (
                    price
                    *
                    (
                        1.0
                        +
                        self.transaction_cost_rate
                        +
                        cfg.SLIPPAGE_RATE
                    )
                )
            )

            if not self.allow_fractional_shares:

                shares_to_buy = float(
                    np.floor(
                        shares_to_buy
                    )
                )

            gross_value = (
                shares_to_buy
                *
                price
            )

            transaction_cost = (
                gross_value
                *
                self.transaction_cost_rate
            )

            slippage_cost = (
                gross_value
                *
                cfg.SLIPPAGE_RATE
            )

            total_cost = (
                gross_value
                +
                transaction_cost
                +
                slippage_cost
            )

        if shares_to_buy <= 0:

            return {
                "executed": False,
                "shares": 0.0,
                "gross_value": 0.0,
                "transaction_cost": 0.0,
            }

        self.cash -= total_cost

        self.shares += shares_to_buy

        self.total_transaction_cost += (
            transaction_cost
            +
            slippage_cost
        )

        self.total_trades += 1

        self.buy_count += 1

        return {
            "executed": True,
            "shares": float(
                shares_to_buy
            ),
            "gross_value": float(
                gross_value
            ),
            "transaction_cost": float(
                transaction_cost
                +
                slippage_cost
            ),
        }


    # ==================================================================
    # SELL
    # ==================================================================

    def _execute_sell(
        self,
        price: float,
    ) -> dict:
        """
        Sell a fraction of the current position.

        Short selling is disabled by default.
        """

        if (
            self.shares
            <=
            0
            and
            not self.allow_short_selling
        ):

            return {
                "executed": False,
                "shares": 0.0,
                "gross_value": 0.0,
                "transaction_cost": 0.0,
            }

        if self.shares > 0:

            shares_to_sell = (
                self.shares
                *
                self.trade_fraction
            )

            if not self.allow_fractional_shares:

                shares_to_sell = float(
                    np.floor(
                        shares_to_sell
                    )
                )

                if (
                    shares_to_sell
                    <
                    1
                    and
                    self.shares
                    >=
                    1
                ):

                    shares_to_sell = 1.0

            shares_to_sell = min(
                shares_to_sell,
                self.shares,
            )

        else:

            shares_to_sell = 0.0

        if shares_to_sell <= 0:

            return {
                "executed": False,
                "shares": 0.0,
                "gross_value": 0.0,
                "transaction_cost": 0.0,
            }

        gross_value = (
            shares_to_sell
            *
            price
        )

        transaction_cost = (
            gross_value
            *
            self.transaction_cost_rate
        )

        slippage_cost = (
            gross_value
            *
            cfg.SLIPPAGE_RATE
        )

        net_proceeds = (
            gross_value
            -
            transaction_cost
            -
            slippage_cost
        )

        self.cash += net_proceeds

        self.shares -= shares_to_sell

        self.total_transaction_cost += (
            transaction_cost
            +
            slippage_cost
        )

        self.total_trades += 1

        self.sell_count += 1

        return {
            "executed": True,
            "shares": float(
                shares_to_sell
            ),
            "gross_value": float(
                gross_value
            ),
            "transaction_cost": float(
                transaction_cost
                +
                slippage_cost
            ),
        }


    # ==================================================================
    # HOLD
    # ==================================================================

    def _execute_hold(
        self,
    ) -> dict:

        self.hold_count += 1

        return {
            "executed": True,
            "shares": 0.0,
            "gross_value": 0.0,
            "transaction_cost": 0.0,
        }


    # ==================================================================
    # ACTION
    # ==================================================================

    def _execute_action(
        self,
        action: int,
        price: float,
    ) -> dict:

        if action == cfg.ACTION_HOLD:

            return self._execute_hold()

        if action == cfg.ACTION_BUY:

            return self._execute_buy(
                price
            )

        if action == cfg.ACTION_SELL:

            return self._execute_sell(
                price
            )

        raise ValueError(
            f"Invalid action {action}. "
            f"Valid actions: {cfg.ACTION_NAMES}"
        )


    # ==================================================================
    # DRAWDOWN
    # ==================================================================

    def _update_drawdown(
        self,
        portfolio_value: float,
    ) -> None:

        self.peak_portfolio_value = max(
            self.peak_portfolio_value,
            portfolio_value,
        )

        if self.peak_portfolio_value > 0:

            self.current_drawdown = (
                portfolio_value
                -
                self.peak_portfolio_value
            ) / self.peak_portfolio_value

        else:

            self.current_drawdown = 0.0

        self.max_drawdown = min(
            self.max_drawdown,
            self.current_drawdown,
        )


    # ==================================================================
    # REWARD
    # ==================================================================

    def _calculate_reward(
        self,
        previous_value: float,
        current_value: float,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        """
        Reward combines portfolio return and drawdown penalty.

        reward =
            RETURN_REWARD_WEIGHT * portfolio_return
            -
            DRAWDOWN_PENALTY_WEIGHT * abs(current_drawdown)

        The exact step-level reward formulation is an
        implementation choice.
        """

        if previous_value <= 0:

            portfolio_return = 0.0

        else:

            portfolio_return = (
                current_value
                /
                previous_value
            ) - 1.0

        drawdown_penalty = abs(
            min(
                0.0,
                self.current_drawdown,
            )
        )

        raw_reward = (
            cfg.RETURN_REWARD_WEIGHT
            *
            portfolio_return
            -
            cfg.DRAWDOWN_PENALTY_WEIGHT
            *
            drawdown_penalty
        )

        clipped_reward = float(
            np.clip(
                raw_reward,
                cfg.REWARD_CLIP_MIN,
                cfg.REWARD_CLIP_MAX,
            )
        )

        return (
            clipped_reward,
            float(
                portfolio_return
            ),
            float(
                drawdown_penalty
            ),
        )


    # ==================================================================
    # STEP
    # ==================================================================

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        dict,
    ]:
        """
        Execute one environment transition.

        Returns
        -------
        next_state
        reward
        done
        info
        """

        if self.done:

            raise RuntimeError(
                "Environment episode has already finished. "
                "Call reset() before step()."
            )

        if action not in cfg.ACTION_NAMES:

            raise ValueError(
                f"Invalid action: {action}"
            )

        current_row = self.data.iloc[
            self.current_step
        ]

        current_price = float(
            current_row[
                "close"
            ]
        )

        previous_value = (
            self._calculate_portfolio_value(
                current_price
            )
        )

        self.previous_portfolio_value = (
            previous_value
        )

        trade_info = (
            self._execute_action(
                action,
                current_price,
            )
        )

        # --------------------------------------------------------------
        # Move market to next day
        # --------------------------------------------------------------

        next_step = (
            self.current_step
            +
            1
        )

        if (
            cfg.MAX_STEPS_PER_EPISODE
            is not None
            and
            next_step
            >=
            cfg.MAX_STEPS_PER_EPISODE
        ):

            next_step = min(
                next_step,
                len(
                    self.data
                )
                -
                1,
            )

            self.done = True

        if next_step >= len(
            self.data
        ):

            next_step = (
                len(
                    self.data
                )
                -
                1
            )

            self.done = True

        self.current_step = (
            next_step
        )

        next_row = self.data.iloc[
            self.current_step
        ]

        next_price = float(
            next_row[
                "close"
            ]
        )

        current_value = (
            self._calculate_portfolio_value(
                next_price
            )
        )

        self.portfolio_value = (
            current_value
        )

        # --------------------------------------------------------------
        # Update drawdown before reward
        # --------------------------------------------------------------

        self._update_drawdown(
            current_value
        )

        reward, portfolio_return, drawdown_penalty = (
            self._calculate_reward(
                previous_value=previous_value,
                current_value=current_value,
            )
        )

        # --------------------------------------------------------------
        # End when final row has been reached
        # --------------------------------------------------------------

        if (
            self.current_step
            >=
            len(
                self.data
            )
            -
            1
        ):

            self.done = True

        next_state = (
            self._get_state()
        )

        action_name = (
            cfg.ACTION_NAMES[
                action
            ]
        )

        info = {

            "symbol":
                self.symbol,

            "date":
                pd.Timestamp(
                    next_row[
                        "date"
                    ]
                ),

            "action":
                int(
                    action
                ),

            "action_name":
                action_name,

            "action_executed":
                bool(
                    trade_info[
                        "executed"
                    ]
                ),

            "trade_shares":
                float(
                    trade_info[
                        "shares"
                    ]
                ),

            "trade_value":
                float(
                    trade_info[
                        "gross_value"
                    ]
                ),

            "transaction_cost":
                float(
                    trade_info[
                        "transaction_cost"
                    ]
                ),

            "cash":
                float(
                    self.cash
                ),

            "shares":
                float(
                    self.shares
                ),

            "price":
                float(
                    next_price
                ),

            "portfolio_value":
                float(
                    current_value
                ),

            "portfolio_return":
                float(
                    portfolio_return
                ),

            "current_drawdown":
                float(
                    self.current_drawdown
                ),

            "max_drawdown":
                float(
                    self.max_drawdown
                ),

            "drawdown_penalty":
                float(
                    drawdown_penalty
                ),

            "reward":
                float(
                    reward
                ),

            "done":
                bool(
                    self.done
                ),

        }

        self.history.append(
            info.copy()
        )

        return (
            next_state,
            reward,
            self.done,
            info,
        )


    # ==================================================================
    # PORTFOLIO STATE
    # ==================================================================

    def get_portfolio_state(
        self,
    ) -> dict:

        current_price = (
            self._current_price()
        )

        current_value = (
            self._calculate_portfolio_value(
                current_price
            )
        )

        return {

            "symbol":
                self.symbol,

            "step":
                int(
                    self.current_step
                ),

            "date":
                pd.Timestamp(
                    self.data[
                        "date"
                    ].iloc[
                        self.current_step
                    ]
                ),

            "cash":
                float(
                    self.cash
                ),

            "shares":
                float(
                    self.shares
                ),

            "price":
                float(
                    current_price
                ),

            "portfolio_value":
                float(
                    current_value
                ),

            "peak_portfolio_value":
                float(
                    self.peak_portfolio_value
                ),

            "current_drawdown":
                float(
                    self.current_drawdown
                ),

            "max_drawdown":
                float(
                    self.max_drawdown
                ),

            "total_transaction_cost":
                float(
                    self.total_transaction_cost
                ),

            "total_trades":
                int(
                    self.total_trades
                ),

            "buy_count":
                int(
                    self.buy_count
                ),

            "sell_count":
                int(
                    self.sell_count
                ),

            "hold_count":
                int(
                    self.hold_count
                ),

        }


    # ==================================================================
    # HISTORY
    # ==================================================================

    def get_history(
        self,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            self.history
        )


# ======================================================================
# ENVIRONMENT FACTORY
# ======================================================================

def build_stock_environment(
    dataset: pd.DataFrame,
    symbol: str,
) -> TradingEnvironment:
    """
    Build one environment for one requested stock.
    """

    symbol = str(
        symbol
    ).strip().upper()

    stock_data = dataset[
        dataset[
            "symbol"
        ]
        .astype(
            str
        )
        .str.upper()
        ==
        symbol
    ].copy()

    if stock_data.empty:

        raise ValueError(
            f"No rows found for symbol: {symbol}"
        )

    return TradingEnvironment(
        stock_data
    )


# ======================================================================
# RANDOM POLICY SMOKE TEST
# ======================================================================

def run_environment_smoke_test() -> dict:
    """
    Run a short random-action episode to validate environment mechanics.

    This is NOT DQN training.
    """

    print_header(
        "AGENT 4 - TRADING ENVIRONMENT SMOKE TEST"
    )

    if not cfg.DQN_TRAINING_DATA_PARQUET.exists():

        raise FileNotFoundError(
            "Training data not found:\n"
            f"{cfg.DQN_TRAINING_DATA_PARQUET}\n\n"
            "Run data_loader.py first."
        )

    training_df = pd.read_parquet(
        cfg.DQN_TRAINING_DATA_PARQUET
    )

    symbols = sorted(
        training_df[
            "symbol"
        ]
        .astype(
            str
        )
        .unique()
        .tolist()
    )

    if not symbols:

        raise ValueError(
            "No symbols found in DQN training data."
        )

    test_symbol = symbols[
        0
    ]

    print(
        f"Test symbol       : {test_symbol}"
    )

    env = build_stock_environment(
        dataset=training_df,
        symbol=test_symbol,
    )

    state = env.reset()

    print(
        f"Rows              : {len(env.data):,}"
    )

    print(
        f"State shape       : {state.shape}"
    )

    print(
        f"Initial cash      : {env.cash:,.2f}"
    )

    print(
        f"Initial value     : {env.portfolio_value:,.2f}"
    )

    rng = np.random.default_rng(
        cfg.RANDOM_SEED
    )

    max_test_steps = min(
        100,
        len(
            env.data
        )
        -
        1,
    )

    rewards = []

    for _ in range(
        max_test_steps
    ):

        action = int(
            rng.integers(
                low=0,
                high=cfg.NUM_ACTIONS,
            )
        )

        _, reward, done, _ = (
            env.step(
                action
            )
        )

        rewards.append(
            reward
        )

        if done:

            break

    portfolio_state = (
        env.get_portfolio_state()
    )

    history = (
        env.get_history()
    )

    checks = {

        "state_size_valid":
            state.shape
            ==
            (
                cfg.STATE_SIZE,
            ),

        "history_created":
            len(
                history
            )
            >
            0,

        "portfolio_value_positive":
            portfolio_state[
                "portfolio_value"
            ]
            >
            0,

        "cash_nonnegative":
            portfolio_state[
                "cash"
            ]
            >=
            -cfg.FLOAT_TOLERANCE,

        "shares_nonnegative":
            (
                portfolio_state[
                    "shares"
                ]
                >=
                -cfg.FLOAT_TOLERANCE
                if
                not cfg.ALLOW_SHORT_SELLING
                else
                True
            ),

        "drawdown_nonpositive":
            portfolio_state[
                "current_drawdown"
            ]
            <=
            cfg.FLOAT_TOLERANCE,

        "max_drawdown_nonpositive":
            portfolio_state[
                "max_drawdown"
            ]
            <=
            cfg.FLOAT_TOLERANCE,

        "rewards_finite":
            bool(
                np.isfinite(
                    np.asarray(
                        rewards,
                        dtype=float,
                    )
                ).all()
            ),

    }

    print()

    print_header(
        "TRADING ENVIRONMENT VALIDATION"
    )

    for name, value in checks.items():

        print(
            f"{name:<40}: {value}"
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
        f"Steps executed      : {len(history):,}"
    )

    print(
        f"Final portfolio     : "
        f"{portfolio_state['portfolio_value']:,.2f}"
    )

    print(
        f"Final cash          : "
        f"{portfolio_state['cash']:,.2f}"
    )

    print(
        f"Shares held         : "
        f"{portfolio_state['shares']:,.2f}"
    )

    print(
        f"Total trades        : "
        f"{portfolio_state['total_trades']}"
    )

    print(
        f"Buy count           : "
        f"{portfolio_state['buy_count']}"
    )

    print(
        f"Sell count          : "
        f"{portfolio_state['sell_count']}"
    )

    print(
        f"Hold count          : "
        f"{portfolio_state['hold_count']}"
    )

    print(
        f"Current drawdown    : "
        f"{portfolio_state['current_drawdown']:.4%}"
    )

    print(
        f"Maximum drawdown    : "
        f"{portfolio_state['max_drawdown']:.4%}"
    )

    print(
        f"Transaction costs   : "
        f"{portfolio_state['total_transaction_cost']:,.2f}"
    )

    if not overall_pass:

        failed = [

            name
            for name, passed
            in checks.items()
            if not passed

        ]

        raise ValueError(
            "Trading environment validation failed:\n"
            +
            "\n".join(
                f"  - {name}"
                for name in failed
            )
        )

    print_header(
        "TRADING ENVIRONMENT STATUS"
    )

    print(
        "Environment mechanics : COMPLETE"
    )

    print(
        "BUY action            : COMPLETE"
    )

    print(
        "SELL action           : COMPLETE"
    )

    print(
        "HOLD action           : COMPLETE"
    )

    print(
        "Portfolio accounting  : COMPLETE"
    )

    print(
        "Drawdown tracking     : COMPLETE"
    )

    print(
        "Reward calculation    : COMPLETE"
    )

    print()

    print(
        "ENVIRONMENT STATUS: COMPLETE"
    )

    return {

        "checks":
            checks,

        "portfolio_state":
            portfolio_state,

        "history":
            history,

    }


# ======================================================================
# DIRECT EXECUTION
# ======================================================================

if __name__ == "__main__":

    run_environment_smoke_test()