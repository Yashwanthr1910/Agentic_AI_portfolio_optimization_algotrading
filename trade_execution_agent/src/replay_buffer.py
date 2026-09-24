"""
Agent 4 - DQN Experience Replay Buffer
======================================================================

Purpose
-------
Store and sample DQN experience tuples:

    (state, action, reward, next_state, done)

Paper alignment
---------------
The paper explicitly states that the DQN interacts with the
environment and stores experience tuples in experience replay.

Implementation choices
----------------------
The exact replay-buffer capacity and sampling implementation
are not clearly specified in the paper.

This implementation uses:
- fixed-capacity deque
- uniform random sampling
- reproducible NumPy random generator
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Deque, NamedTuple

import numpy as np


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
# EXPERIENCE STRUCTURE
# ======================================================================

class Experience(NamedTuple):
    """
    One DQN transition.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# ======================================================================
# REPLAY BUFFER
# ======================================================================

class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored.

    state_size:
        Expected state vector length.

    random_seed:
        Seed used for reproducible sampling.
    """

    def __init__(
        self,
        capacity: int | None = None,
        state_size: int | None = None,
        random_seed: int | None = None,
    ) -> None:

        self.capacity = int(
            cfg.REPLAY_BUFFER_CAPACITY
            if capacity is None
            else capacity
        )

        self.state_size = int(
            cfg.STATE_SIZE
            if state_size is None
            else state_size
        )

        self.random_seed = int(
            cfg.RANDOM_SEED
            if random_seed is None
            else random_seed
        )

        if self.capacity <= 0:
            raise ValueError(
                "Replay buffer capacity must be positive."
            )

        if self.state_size <= 0:
            raise ValueError(
                "Replay buffer state_size must be positive."
            )

        self.buffer: Deque[Experience] = deque(
            maxlen=self.capacity
        )

        self.rng = np.random.default_rng(
            self.random_seed
        )


    # ==================================================================
    # LENGTH
    # ==================================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.buffer
        )


    # ==================================================================
    # CLEAR
    # ==================================================================

    def clear(
        self,
    ) -> None:

        self.buffer.clear()


    # ==================================================================
    # STATE VALIDATION
    # ==================================================================

    def _validate_state(
        self,
        state: np.ndarray,
        name: str,
    ) -> np.ndarray:

        state_array = np.asarray(
            state,
            dtype=np.float32,
        )

        if state_array.ndim != 1:

            raise ValueError(
                f"{name} must be a 1D array. "
                f"Got shape {state_array.shape}."
            )

        if state_array.shape[0] != self.state_size:

            raise ValueError(
                f"{name} size mismatch. "
                f"Expected {self.state_size}, "
                f"got {state_array.shape[0]}."
            )

        if not np.isfinite(
            state_array
        ).all():

            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        return state_array


    # ==================================================================
    # ADD EXPERIENCE
    # ==================================================================

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:

        state_array = self._validate_state(
            state,
            "state",
        )

        next_state_array = self._validate_state(
            next_state,
            "next_state",
        )

        action = int(
            action
        )

        if action not in cfg.ACTION_NAMES:

            raise ValueError(
                f"Invalid action {action}. "
                f"Valid actions are "
                f"{list(cfg.ACTION_NAMES.keys())}."
            )

        reward = float(
            reward
        )

        if not np.isfinite(
            reward
        ):

            raise ValueError(
                "Reward must be finite."
            )

        experience = Experience(
            state=state_array.copy(),
            action=action,
            reward=reward,
            next_state=next_state_array.copy(),
            done=bool(done),
        )

        self.buffer.append(
            experience
        )


    # ==================================================================
    # CAN SAMPLE
    # ==================================================================

    def can_sample(
        self,
        batch_size: int | None = None,
    ) -> bool:

        batch_size = int(
            cfg.BATCH_SIZE
            if batch_size is None
            else batch_size
        )

        return len(
            self.buffer
        ) >= batch_size


    # ==================================================================
    # SAMPLE
    # ==================================================================

    def sample(
        self,
        batch_size: int | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Uniformly sample a mini-batch.

        Returns
        -------
        states:
            shape = (batch_size, state_size)

        actions:
            shape = (batch_size,)

        rewards:
            shape = (batch_size,)

        next_states:
            shape = (batch_size, state_size)

        dones:
            shape = (batch_size,)
        """

        batch_size = int(
            cfg.BATCH_SIZE
            if batch_size is None
            else batch_size
        )

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be positive."
            )

        if len(
            self.buffer
        ) < batch_size:

            raise ValueError(
                f"Not enough replay experiences. "
                f"Requested {batch_size}, "
                f"available {len(self.buffer)}."
            )

        indices = self.rng.choice(
            len(
                self.buffer
            ),
            size=batch_size,
            replace=False,
        )

        batch = [
            self.buffer[
                int(index)
            ]
            for index in indices
        ]

        states = np.stack(
            [
                experience.state
                for experience in batch
            ]
        ).astype(
            np.float32
        )

        actions = np.asarray(
            [
                experience.action
                for experience in batch
            ],
            dtype=np.int64,
        )

        rewards = np.asarray(
            [
                experience.reward
                for experience in batch
            ],
            dtype=np.float32,
        )

        next_states = np.stack(
            [
                experience.next_state
                for experience in batch
            ]
        ).astype(
            np.float32
        )

        dones = np.asarray(
            [
                experience.done
                for experience in batch
            ],
            dtype=np.float32,
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )


    # ==================================================================
    # GET ALL
    # ==================================================================

    def get_all(
        self,
    ) -> list[Experience]:

        return list(
            self.buffer
        )


# ======================================================================
# SMOKE TEST
# ======================================================================

def run_replay_buffer_smoke_test() -> dict:
    """
    Validate replay-memory mechanics.

    This does not perform DQN training.
    """

    print()
    print(
        "=" * 100
    )
    print(
        "AGENT 4 - REPLAY BUFFER SMOKE TEST"
    )
    print(
        "=" * 100
    )
    print()

    test_capacity = 100

    buffer = ReplayBuffer(
        capacity=test_capacity,
        state_size=cfg.STATE_SIZE,
        random_seed=cfg.RANDOM_SEED,
    )

    rng = np.random.default_rng(
        cfg.RANDOM_SEED
    )

    transitions_to_add = 80

    for index in range(
        transitions_to_add
    ):

        state = rng.normal(
            size=cfg.STATE_SIZE
        ).astype(
            np.float32
        )

        next_state = rng.normal(
            size=cfg.STATE_SIZE
        ).astype(
            np.float32
        )

        action = int(
            index
            %
            cfg.NUM_ACTIONS
        )

        reward = float(
            rng.normal()
        )

        done = bool(
            index
            %
            17
            ==
            0
        )

        buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    sample_size = min(
        cfg.BATCH_SIZE,
        len(
            buffer
        ),
    )

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
    ) = buffer.sample(
        sample_size
    )

    checks = {

        "buffer_length_valid":
            len(
                buffer
            )
            ==
            transitions_to_add,

        "capacity_valid":
            buffer.capacity
            ==
            test_capacity,

        "states_shape_valid":
            states.shape
            ==
            (
                sample_size,
                cfg.STATE_SIZE,
            ),

        "actions_shape_valid":
            actions.shape
            ==
            (
                sample_size,
            ),

        "rewards_shape_valid":
            rewards.shape
            ==
            (
                sample_size,
            ),

        "next_states_shape_valid":
            next_states.shape
            ==
            (
                sample_size,
                cfg.STATE_SIZE,
            ),

        "dones_shape_valid":
            dones.shape
            ==
            (
                sample_size,
            ),

        "states_finite":
            bool(
                np.isfinite(
                    states
                ).all()
            ),

        "rewards_finite":
            bool(
                np.isfinite(
                    rewards
                ).all()
            ),

        "next_states_finite":
            bool(
                np.isfinite(
                    next_states
                ).all()
            ),

        "actions_valid":
            bool(
                np.isin(
                    actions,
                    list(
                        cfg.ACTION_NAMES.keys()
                    ),
                ).all()
            ),

    }

    print(
        f"Capacity          : {buffer.capacity:,}"
    )

    print(
        f"Experiences added : {len(buffer):,}"
    )

    print(
        f"Sample size       : {sample_size:,}"
    )

    print(
        f"States shape      : {states.shape}"
    )

    print(
        f"Actions shape     : {actions.shape}"
    )

    print(
        f"Rewards shape     : {rewards.shape}"
    )

    print(
        f"Next states shape : {next_states.shape}"
    )

    print(
        f"Dones shape       : {dones.shape}"
    )

    print()

    print(
        "=" * 100
    )

    print(
        "REPLAY BUFFER VALIDATION"
    )

    print(
        "=" * 100
    )

    print()

    for name, passed in checks.items():

        print(
            f"{name:<40}: {passed}"
        )

    overall_pass = all(
        checks.values()
    )

    print()

    print(
        "VALIDATION RESULT: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    if not overall_pass:

        failed = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        raise ValueError(
            "Replay buffer validation failed:\n"
            +
            "\n".join(
                f"  - {name}"
                for name in failed
            )
        )

    # ----------------------------------------------------------
    # Capacity rollover test
    # ----------------------------------------------------------

    rollover_buffer = ReplayBuffer(
        capacity=10,
        state_size=cfg.STATE_SIZE,
        random_seed=cfg.RANDOM_SEED,
    )

    for index in range(
        25
    ):

        state = np.full(
            cfg.STATE_SIZE,
            fill_value=float(
                index
            ),
            dtype=np.float32,
        )

        rollover_buffer.add(
            state=state,
            action=cfg.ACTION_HOLD,
            reward=float(
                index
            ),
            next_state=state,
            done=False,
        )

    rollover_valid = (
        len(
            rollover_buffer
        )
        ==
        10
    )

    print()

    print(
        f"Capacity rollover valid                 : "
        f"{rollover_valid}"
    )

    if not rollover_valid:

        raise ValueError(
            "Replay buffer capacity rollover test failed."
        )

    print()

    print(
        "=" * 100
    )

    print(
        "REPLAY BUFFER STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Experience storage      : COMPLETE"
    )

    print(
        "Uniform sampling       : COMPLETE"
    )

    print(
        "Batch construction     : COMPLETE"
    )

    print(
        "Capacity rollover      : COMPLETE"
    )

    print(
        "State validation       : COMPLETE"
    )

    print()

    print(
        "REPLAY BUFFER STATUS: COMPLETE"
    )

    return {
        "checks": checks,
        "sample_size": sample_size,
        "buffer_length": len(
            buffer
        ),
    }


# ======================================================================
# DIRECT EXECUTION
# ======================================================================

if __name__ == "__main__":

    run_replay_buffer_smoke_test()