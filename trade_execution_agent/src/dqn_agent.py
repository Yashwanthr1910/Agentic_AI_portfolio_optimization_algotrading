"""
Agent 4 - Deep Q-Learning Agent
======================================================================

Purpose
-------
Implements the learning logic for the trade-execution DQN.

Main responsibilities
---------------------
1. Build evaluation and target Q-networks.
2. Select HOLD / BUY / SELL actions using epsilon-greedy policy.
3. Store experiences in replay memory.
4. Sample replay mini-batches.
5. Compute Bellman Q-learning targets.
6. Train the evaluation network.
7. Periodically synchronize the target network.
8. Decay epsilon over time.
9. Save and load learned model weights.

Action mapping
--------------
0 = HOLD
1 = BUY
2 = SELL

Bellman target
--------------
For non-terminal transitions:

    y = r + gamma * max_a Q_target(s_next, a)

For terminal transitions:

    y = r

Only the Q-value corresponding to the action actually taken is replaced
with the Bellman target.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


# ======================================================================
# PROJECT IMPORT PATH
# ======================================================================

AGENT_ROOT = Path(__file__).resolve().parents[1]

if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))


from config import config as cfg
from src.dqn_model import (
    build_dqn_network_pair,
    update_target_network,
    set_random_seeds,
)
from src.replay_buffer import ReplayBuffer


# ======================================================================
# DQN AGENT
# ======================================================================

class DQNAgent:
    """
    Deep Q-Network trading agent.
    """

    def __init__(
        self,
        state_size: int | None = None,
        num_actions: int | None = None,
        seed: int | None = None,
    ) -> None:

        self.state_size = int(
            cfg.STATE_SIZE
            if state_size is None
            else state_size
        )

        self.num_actions = int(
            cfg.NUM_ACTIONS
            if num_actions is None
            else num_actions
        )

        self.seed = int(
            cfg.RANDOM_SEED
            if seed is None
            else seed
        )

        if self.state_size <= 0:
            raise ValueError(
                "state_size must be positive."
            )

        if self.num_actions != 3:
            raise ValueError(
                "Agent-4 expects exactly three actions: "
                "HOLD, BUY, SELL."
            )

        # --------------------------------------------------------------
        # Reproducibility
        # --------------------------------------------------------------

        set_random_seeds(
            self.seed
        )

        self.rng = np.random.default_rng(
            self.seed
        )

        random.seed(
            self.seed
        )

        # --------------------------------------------------------------
        # Hyperparameters
        # --------------------------------------------------------------

        self.gamma = float(
            cfg.GAMMA
        )

        self.batch_size = int(
            cfg.BATCH_SIZE
        )

        self.epsilon = float(
            cfg.EPSILON_START
        )

        self.epsilon_min = float(
            cfg.EPSILON_MIN
        )

        self.epsilon_decay = float(
            cfg.EPSILON_DECAY
        )

        self.target_update_frequency = int(
            cfg.TARGET_UPDATE_FREQUENCY
        )

        self.min_replay_size = int(
            cfg.MIN_REPLAY_SIZE
        )

        # --------------------------------------------------------------
        # Networks
        # --------------------------------------------------------------

        (
            self.evaluation_network,
            self.target_network,
        ) = build_dqn_network_pair(
            state_size=self.state_size,
            num_actions=self.num_actions,
        )

        # --------------------------------------------------------------
        # Replay buffer
        # --------------------------------------------------------------
        self.replay_buffer = ReplayBuffer(
         capacity=cfg.REPLAY_BUFFER_CAPACITY,
         state_size=self.state_size,
        )

        # --------------------------------------------------------------
        # Counters
        # --------------------------------------------------------------

        self.environment_steps = 0
        self.learning_steps = 0
        self.target_updates = 0

        # --------------------------------------------------------------
        # Training history
        # --------------------------------------------------------------

        self.loss_history: list[float] = []
        self.epsilon_history: list[float] = []

    # ==================================================================
    # STATE VALIDATION
    # ==================================================================

    def _prepare_state(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a state into a validated float32 vector.
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.ndim != 1:
            raise ValueError(
                "State must be one-dimensional."
            )

        if state.shape[0] != self.state_size:
            raise ValueError(
                f"Expected state size {self.state_size}, "
                f"received {state.shape[0]}."
            )

        if not np.isfinite(state).all():
            raise ValueError(
                "State contains NaN or infinite values."
            )

        return state

    # ==================================================================
    # Q-VALUE PREDICTION
    # ==================================================================

    def predict_q_values(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Return Q-values from the evaluation network.
        """

        state = self._prepare_state(
            state
        )

        state_batch = np.expand_dims(
            state,
            axis=0,
        )

        q_values = self.evaluation_network(
            state_batch,
            training=False,
        ).numpy()[0]

        if q_values.shape != (
            self.num_actions,
        ):
            raise ValueError(
                "Unexpected Q-value shape: "
                f"{q_values.shape}"
            )

        if not np.isfinite(q_values).all():
            raise ValueError(
                "Q-values contain NaN or infinite values."
            )

        return q_values.astype(
            np.float32
        )

    # ==================================================================
    # ACTION SELECTION
    # ==================================================================

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
    ) -> int:
        """
        Select an action using epsilon-greedy policy.

        During training:
            random action with probability epsilon
            greedy action otherwise

        During evaluation/inference:
            always greedy
        """

        state = self._prepare_state(
            state
        )

        if training:

            random_value = float(
                self.rng.random()
            )

            if random_value < self.epsilon:

                action = int(
                    self.rng.integers(
                        0,
                        self.num_actions,
                    )
                )

                return action

        q_values = self.predict_q_values(
            state
        )

        action = int(
            np.argmax(
                q_values
            )
        )

        return action

    # ==================================================================
    # EXPERIENCE STORAGE
    # ==================================================================

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store one transition in replay memory.
        """

        state = self._prepare_state(
            state
        )

        next_state = self._prepare_state(
            next_state
        )

        self.replay_buffer.add(
            state=state,
            action=int(action),
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
        )

        self.environment_steps += 1

    # ==================================================================
    # BELLMAN TARGETS
    # ==================================================================

    def compute_training_targets(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Compute the DQN training targets.

        Returns
        -------
        current_q_values:
            Evaluation-network predictions before replacement.

        target_q_values:
            Copy of current predictions where only the selected action's
            Q-value is replaced with the Bellman target.
        """

        states = np.asarray(
            states,
            dtype=np.float32,
        )

        next_states = np.asarray(
            next_states,
            dtype=np.float32,
        )

        actions = np.asarray(
            actions,
            dtype=np.int64,
        )

        rewards = np.asarray(
            rewards,
            dtype=np.float32,
        )

        dones = np.asarray(
            dones,
            dtype=np.float32,
        )

        batch_size = states.shape[0]

        if states.shape != (
            batch_size,
            self.state_size,
        ):
            raise ValueError(
                "Invalid states batch shape."
            )

        if next_states.shape != (
            batch_size,
            self.state_size,
        ):
            raise ValueError(
                "Invalid next_states batch shape."
            )

        if actions.shape != (
            batch_size,
        ):
            raise ValueError(
                "Invalid actions batch shape."
            )

        if rewards.shape != (
            batch_size,
        ):
            raise ValueError(
                "Invalid rewards batch shape."
            )

        if dones.shape != (
            batch_size,
        ):
            raise ValueError(
                "Invalid dones batch shape."
            )

        if not np.isfinite(states).all():
            raise ValueError(
                "states contains invalid values."
            )

        if not np.isfinite(next_states).all():
            raise ValueError(
                "next_states contains invalid values."
            )

        if not np.isfinite(rewards).all():
            raise ValueError(
                "rewards contains invalid values."
            )

        if not np.isfinite(dones).all():
            raise ValueError(
                "dones contains invalid values."
            )

        if np.any(actions < 0) or np.any(
            actions >= self.num_actions
        ):
            raise ValueError(
                "Action batch contains invalid actions."
            )

        # --------------------------------------------------------------
        # Current Q(s, a)
        # --------------------------------------------------------------

        current_q_values = (
            self.evaluation_network(
                states,
                training=False,
            )
            .numpy()
            .astype(
                np.float32
            )
        )

        # --------------------------------------------------------------
        # Q_target(s_next, a)
        # --------------------------------------------------------------

        next_q_values = (
            self.target_network(
                next_states,
                training=False,
            )
            .numpy()
            .astype(
                np.float32
            )
        )

        max_next_q = np.max(
            next_q_values,
            axis=1,
        )

        # --------------------------------------------------------------
        # Bellman equation
        #
        # target =
        # reward + gamma * max(Q_next) * (1 - done)
        # --------------------------------------------------------------

        bellman_targets = (
            rewards
            +
            self.gamma
            *
            max_next_q
            *
            (
                1.0
                -
                dones
            )
        )

        target_q_values = (
            current_q_values.copy()
        )

        row_indices = np.arange(
            batch_size
        )

        target_q_values[
            row_indices,
            actions,
        ] = bellman_targets

        return (
            current_q_values,
            target_q_values,
        )

    # ==================================================================
    # LEARNING
    # ==================================================================

    def train_step(
        self,
    ) -> dict[str, Any] | None:
        """
        Perform one replay-learning step.

        Returns None if the replay buffer has not yet reached the
        configured minimum replay size.
        """

        if len(
            self.replay_buffer
        ) < self.min_replay_size:

            return None

        if not self.replay_buffer.can_sample(
            self.batch_size
        ):

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(
            self.batch_size
        )

        (
            current_q_values,
            target_q_values,
        ) = self.compute_training_targets(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

        loss = self.evaluation_network.train_on_batch(
            states,
            target_q_values,
        )

        loss = float(
            np.asarray(
                loss
            ).reshape(
                -1
            )[0]
        )

        if not np.isfinite(loss):
            raise ValueError(
                "Training produced a non-finite loss."
            )

        self.learning_steps += 1

        self.loss_history.append(
            loss
        )

        # --------------------------------------------------------------
        # Target network synchronization
        # --------------------------------------------------------------

        target_updated = False

        if (
            self.learning_steps
            %
            self.target_update_frequency
            ==
            0
        ):

            self.sync_target_network()

            target_updated = True

        # --------------------------------------------------------------
        # Epsilon decay
        # --------------------------------------------------------------

        old_epsilon = float(
            self.epsilon
        )

        self.decay_epsilon()

        self.epsilon_history.append(
            self.epsilon
        )

        return {
            "loss":
                loss,

            "learning_step":
                int(
                    self.learning_steps
                ),

            "environment_steps":
                int(
                    self.environment_steps
                ),

            "replay_size":
                int(
                    len(
                        self.replay_buffer
                    )
                ),

            "epsilon_before":
                old_epsilon,

            "epsilon_after":
                float(
                    self.epsilon
                ),

            "target_updated":
                bool(
                    target_updated
                ),

            "mean_current_q":
                float(
                    np.mean(
                        current_q_values
                    )
                ),

            "mean_target_q":
                float(
                    np.mean(
                        target_q_values
                    )
                ),
        }

    # ==================================================================
    # TARGET NETWORK SYNC
    # ==================================================================

    def sync_target_network(
        self,
    ) -> None:
        """
        Hard-copy evaluation-network weights into target network.
        """

        update_target_network(
            evaluation_network=self.evaluation_network,
            target_network=self.target_network,
        )

        self.target_updates += 1

    # ==================================================================
    # EPSILON DECAY
    # ==================================================================

    def decay_epsilon(
        self,
    ) -> float:
        """
        Apply multiplicative epsilon decay.
        """

        if self.epsilon > self.epsilon_min:

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon
                *
                self.epsilon_decay,
            )

        return float(
            self.epsilon
        )

    # ==================================================================
    # FORCE GREEDY MODE
    # ==================================================================

    def set_evaluation_mode(
        self,
    ) -> None:
        """
        Disable exploration.
        """

        self.epsilon = 0.0

    # ==================================================================
    # SAVE
    # ==================================================================

    def save(
        self,
    ) -> None:
        """
        Save evaluation and target networks.
        """

        cfg.MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.evaluation_network.save(
            cfg.DQN_MODEL_PATH
        )

        self.evaluation_network.save_weights(
            cfg.DQN_WEIGHTS_PATH
        )

        self.target_network.save(
            cfg.TARGET_MODEL_PATH
        )

    # ==================================================================
    # LOAD WEIGHTS
    # ==================================================================

    def load_weights(
        self,
        weights_path: str | Path | None = None,
        sync_target: bool = True,
    ) -> None:
        """
        Load evaluation-network weights.
        """

        path = Path(
            cfg.DQN_WEIGHTS_PATH
            if weights_path is None
            else weights_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"DQN weights not found: {path}"
            )

        self.evaluation_network.load_weights(
            path
        )

        if sync_target:

            self.sync_target_network()

    # ==================================================================
    # METADATA
    # ==================================================================

    def get_status(
        self,
    ) -> dict[str, Any]:
        """
        Return current DQN-agent status.
        """

        return {
            "state_size":
                int(
                    self.state_size
                ),

            "num_actions":
                int(
                    self.num_actions
                ),

            "gamma":
                float(
                    self.gamma
                ),

            "batch_size":
                int(
                    self.batch_size
                ),

            "epsilon":
                float(
                    self.epsilon
                ),

            "epsilon_min":
                float(
                    self.epsilon_min
                ),

            "epsilon_decay":
                float(
                    self.epsilon_decay
                ),

            "replay_size":
                int(
                    len(
                        self.replay_buffer
                    )
                ),

            "environment_steps":
                int(
                    self.environment_steps
                ),

            "learning_steps":
                int(
                    self.learning_steps
                ),

            "target_updates":
                int(
                    self.target_updates
                ),
        }


# ======================================================================
# SMOKE TEST
# ======================================================================

def run_dqn_agent_smoke_test() -> dict[str, Any]:
    """
    Validate core DQN-agent functionality.
    """

    print()
    print("=" * 100)
    print("AGENT 4 - DQN AGENT SMOKE TEST")
    print("=" * 100)
    print()

    set_random_seeds(
        cfg.RANDOM_SEED
    )

    agent = DQNAgent()

    print(
        f"State size             : {agent.state_size}"
    )

    print(
        f"Number of actions      : {agent.num_actions}"
    )

    print(
        f"Gamma                  : {agent.gamma}"
    )

    print(
        f"Batch size             : {agent.batch_size}"
    )

    print(
        f"Initial epsilon        : {agent.epsilon}"
    )

    print(
        f"Replay capacity        : {cfg.REPLAY_BUFFER_CAPACITY:,}"
    )

    print(
        f"Minimum replay size    : {agent.min_replay_size:,}"
    )

    print(
        f"Target update freq     : {agent.target_update_frequency}"
    )

    print()

    # ------------------------------------------------------------------
    # State + Q prediction
    # ------------------------------------------------------------------

    rng = np.random.default_rng(
        cfg.RANDOM_SEED
    )

    sample_state = rng.normal(
        size=agent.state_size
    ).astype(
        np.float32
    )

    q_values = agent.predict_q_values(
        sample_state
    )

    greedy_action = agent.select_action(
        sample_state,
        training=False,
    )

    checks = {}

    checks[
        "q_value_shape_valid"
    ] = (
        q_values.shape
        ==
        (
            agent.num_actions,
        )
    )

    checks[
        "q_values_finite"
    ] = bool(
        np.isfinite(
            q_values
        ).all()
    )

    checks[
        "greedy_action_valid"
    ] = (
        greedy_action
        in cfg.ACTION_NAMES
    )

    checks[
        "greedy_action_matches_argmax"
    ] = (
        greedy_action
        ==
        int(
            np.argmax(
                q_values
            )
        )
    )

    # ------------------------------------------------------------------
    # Epsilon random action check
    # ------------------------------------------------------------------

    agent.epsilon = 1.0

    random_actions = [
        agent.select_action(
            sample_state,
            training=True,
        )
        for _ in range(
            100
        )
    ]

    checks[
        "exploration_actions_valid"
    ] = all(
        action
        in cfg.ACTION_NAMES
        for action
        in random_actions
    )

    checks[
        "exploration_multiple_actions"
    ] = (
        len(
            set(
                random_actions
            )
        )
        >
        1
    )

    # Restore epsilon
    agent.epsilon = float(
        cfg.EPSILON_START
    )

    # ------------------------------------------------------------------
    # Fill replay buffer
    # ------------------------------------------------------------------

    required_samples = max(
        agent.min_replay_size,
        agent.batch_size,
    )

    for index in range(
        required_samples
    ):

        state = rng.normal(
            size=agent.state_size
        ).astype(
            np.float32
        )

        next_state = (
            state
            +
            rng.normal(
                loc=0.0,
                scale=0.05,
                size=agent.state_size,
            ).astype(
                np.float32
            )
        )

        action = int(
            rng.integers(
                0,
                agent.num_actions,
            )
        )

        reward = float(
            rng.normal(
                loc=0.0,
                scale=0.01,
            )
        )

        done = bool(
            index
            %
            100
            ==
            0
        )

        agent.remember(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    checks[
        "replay_minimum_reached"
    ] = (
        len(
            agent.replay_buffer
        )
        >=
        agent.min_replay_size
    )

    # ------------------------------------------------------------------
    # Bellman target test
    # ------------------------------------------------------------------

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
    ) = agent.replay_buffer.sample(
        agent.batch_size
    )

    (
        current_q,
        target_q,
    ) = agent.compute_training_targets(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )

    checks[
        "current_q_shape_valid"
    ] = (
        current_q.shape
        ==
        (
            agent.batch_size,
            agent.num_actions,
        )
    )

    checks[
        "target_q_shape_valid"
    ] = (
        target_q.shape
        ==
        (
            agent.batch_size,
            agent.num_actions,
        )
    )

    checks[
        "target_q_finite"
    ] = bool(
        np.isfinite(
            target_q
        ).all()
    )

    # Verify unselected Q-values were not modified.
    unchanged_mask = np.ones_like(
        current_q,
        dtype=bool,
    )

    unchanged_mask[
        np.arange(
            agent.batch_size
        ),
        actions,
    ] = False

    checks[
        "unselected_q_values_preserved"
    ] = bool(
        np.allclose(
            current_q[
                unchanged_mask
            ],
            target_q[
                unchanged_mask
            ],
        )
    )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    epsilon_before = float(
        agent.epsilon
    )

    training_result = agent.train_step()

    checks[
        "training_result_created"
    ] = (
        training_result
        is not None
    )

    if training_result is not None:

        checks[
            "loss_finite"
        ] = bool(
            np.isfinite(
                training_result[
                    "loss"
                ]
            )
        )

        checks[
            "learning_step_incremented"
        ] = (
            agent.learning_steps
            ==
            1
        )

        checks[
            "epsilon_decayed"
        ] = (
            agent.epsilon
            <
            epsilon_before
        )

    else:

        checks[
            "loss_finite"
        ] = False

        checks[
            "learning_step_incremented"
        ] = False

        checks[
            "epsilon_decayed"
        ] = False

    # ------------------------------------------------------------------
    # Target sync test
    # ------------------------------------------------------------------

    eval_weights = (
        agent.evaluation_network.get_weights()
    )

    target_weights_before = (
        agent.target_network.get_weights()
    )

    weights_different_before_sync = any(
        not np.allclose(
            eval_weight,
            target_weight,
        )
        for eval_weight, target_weight
        in zip(
            eval_weights,
            target_weights_before,
        )
    )

    agent.sync_target_network()

    target_weights_after = (
        agent.target_network.get_weights()
    )

    weights_same_after_sync = all(
        np.allclose(
            eval_weight,
            target_weight,
        )
        for eval_weight, target_weight
        in zip(
            eval_weights,
            target_weights_after,
        )
    )

    checks[
        "weights_changed_after_training"
    ] = bool(
        weights_different_before_sync
    )

    checks[
        "target_sync_valid"
    ] = bool(
        weights_same_after_sync
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    print("=" * 100)
    print("DQN AGENT VALIDATION")
    print("=" * 100)
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
        f"Q-values              : {q_values}"
    )

    print(
        f"Greedy action         : "
        f"{greedy_action} "
        f"({cfg.ACTION_NAMES[greedy_action]})"
    )

    if training_result is not None:

        print(
            f"Training loss         : "
            f"{training_result['loss']:.8f}"
        )

        print(
            f"Epsilon after step    : "
            f"{training_result['epsilon_after']:.6f}"
        )

    print(
        f"Replay size           : "
        f"{len(agent.replay_buffer):,}"
    )

    print(
        f"Learning steps        : "
        f"{agent.learning_steps}"
    )

    print(
        f"Target updates        : "
        f"{agent.target_updates}"
    )

    if not overall_pass:

        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise ValueError(
            "DQN-agent validation failed:\n"
            +
            "\n".join(
                f"  - {item}"
                for item in failed
            )
        )

    # ------------------------------------------------------------------
    # Save smoke-test report
    # ------------------------------------------------------------------

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        cfg.REPORT_DIR
        /
        "dqn_agent_summary.json"
    )

    report = {
        "status":
            agent.get_status(),

        "checks":
            {
                key:
                    bool(
                        value
                    )
                for key, value
                in checks.items()
            },

        "sample_q_values":
            [
                float(
                    value
                )
                for value
                in q_values
            ],

        "greedy_action":
            int(
                greedy_action
            ),

        "greedy_action_name":
            cfg.ACTION_NAMES[
                greedy_action
            ],

        "training_result":
            training_result,
    }

    with open(
        report_path,
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
        f"Agent report saved    : {report_path}"
    )

    print()

    print("=" * 100)
    print("DQN AGENT STATUS")
    print("=" * 100)
    print()

    print(
        "Q-value prediction     : COMPLETE"
    )

    print(
        "Epsilon-greedy policy  : COMPLETE"
    )

    print(
        "Replay integration     : COMPLETE"
    )

    print(
        "Bellman targets        : COMPLETE"
    )

    print(
        "Network training       : COMPLETE"
    )

    print(
        "Target synchronization : COMPLETE"
    )

    print(
        "Epsilon decay          : COMPLETE"
    )

    print()

    print(
        "DQN AGENT STATUS: COMPLETE"
    )

    return {
        "agent":
            agent,

        "checks":
            checks,

        "report_path":
            report_path,
    }


# ======================================================================
# DIRECT EXECUTION
# ======================================================================

if __name__ == "__main__":

    run_dqn_agent_smoke_test()