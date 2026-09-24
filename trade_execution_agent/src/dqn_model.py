"""
Agent 4 - Deep Q-Network Model
======================================================================

Purpose
-------
Define the neural-network architecture used by the DQN trade
execution agent.

Architecture
------------
Input:
    7 state features

Hidden layer 1:
    Dense(64, ReLU)

Hidden layer 2:
    Dense(64, ReLU)

Output:
    Dense(3, Linear)

The three output values correspond to:

    Q(HOLD)
    Q(BUY)
    Q(SELL)

Paper alignment
---------------
The paper specifies:

    - two hidden layers
    - 64 neurons per hidden layer
    - Adam optimizer
    - learning rate = 0.0001

The paper also describes BUY, SELL, and HOLD as the available
actions.

Implementation note
-------------------
One section of the paper mentions seven output neurons, which is
inconsistent with the explicit three-action definition.

This implementation therefore uses three output neurons because
a DQN requires one Q-value for each discrete action.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


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
# REPRODUCIBILITY
# ======================================================================

def set_random_seeds(
    seed: int | None = None,
) -> int:
    """
    Set Python, NumPy, and TensorFlow random seeds.
    """

    seed = int(
        cfg.RANDOM_SEED
        if seed is None
        else seed
    )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    tf.random.set_seed(
        seed
    )

    return seed


# ======================================================================
# OPTIMIZER
# ======================================================================

def build_optimizer() -> tf.keras.optimizers.Optimizer:
    """
    Build Adam optimizer using the configured learning rate.
    """

    optimizer_name = str(
        cfg.OPTIMIZER_NAME
    ).strip().lower()

    if optimizer_name != "adam":

        raise ValueError(
            "Agent-4 DQN currently expects Adam optimizer. "
            f"Received: {cfg.OPTIMIZER_NAME}"
        )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.LEARNING_RATE
    )

    return optimizer


# ======================================================================
# LOSS
# ======================================================================

def get_loss():
    """
    Return configured DQN loss function.
    """

    loss_name = str(
        cfg.LOSS_FUNCTION
    ).strip().lower()

    if loss_name == "huber":

        return tf.keras.losses.Huber()

    if loss_name in {
        "mse",
        "mean_squared_error",
    }:

        return tf.keras.losses.MeanSquaredError()

    raise ValueError(
        "Unsupported loss function: "
        f"{cfg.LOSS_FUNCTION}"
    )


# ======================================================================
# BUILD DQN
# ======================================================================

def build_dqn_model(
    state_size: int | None = None,
    num_actions: int | None = None,
    model_name: str = "dqn_network",
) -> tf.keras.Model:
    """
    Build the paper-aligned DQN neural network.

    Parameters
    ----------
    state_size:
        Number of input state variables.

    num_actions:
        Number of discrete trading actions.

    model_name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
    """

    state_size = int(
        cfg.STATE_SIZE
        if state_size is None
        else state_size
    )

    num_actions = int(
        cfg.NUM_ACTIONS
        if num_actions is None
        else num_actions
    )

    if state_size <= 0:

        raise ValueError(
            "state_size must be positive."
        )

    if num_actions <= 0:

        raise ValueError(
            "num_actions must be positive."
        )

    if num_actions != 3:

        raise ValueError(
            "Agent-4 currently expects exactly three "
            "actions: HOLD, BUY, SELL."
        )

    if len(
        cfg.HIDDEN_UNITS
    ) != 2:

        raise ValueError(
            "Paper-aligned DQN requires exactly "
            "two hidden layers."
        )

    inputs = tf.keras.Input(
        shape=(
            state_size,
        ),
        name="state_input",
    )

    x = tf.keras.layers.Dense(
        units=int(
            cfg.HIDDEN_UNITS[
                0
            ]
        ),
        activation=cfg.HIDDEN_ACTIVATION,
        name="hidden_dense_1",
    )(
        inputs
    )

    x = tf.keras.layers.Dense(
        units=int(
            cfg.HIDDEN_UNITS[
                1
            ]
        ),
        activation=cfg.HIDDEN_ACTIVATION,
        name="hidden_dense_2",
    )(
        x
    )

    outputs = tf.keras.layers.Dense(
        units=num_actions,
        activation=cfg.OUTPUT_ACTIVATION,
        name="q_values",
    )(
        x
    )

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name,
    )

    model.compile(
        optimizer=build_optimizer(),
        loss=get_loss(),
    )

    return model


# ======================================================================
# BUILD EVALUATION + TARGET NETWORKS
# ======================================================================

def build_dqn_network_pair(
    state_size: int | None = None,
    num_actions: int | None = None,
) -> tuple[
    tf.keras.Model,
    tf.keras.Model,
]:
    """
    Build evaluation and target networks.

    Both networks start with identical weights.
    """

    set_random_seeds()

    evaluation_network = build_dqn_model(
        state_size=state_size,
        num_actions=num_actions,
        model_name="evaluation_network",
    )

    target_network = build_dqn_model(
        state_size=state_size,
        num_actions=num_actions,
        model_name="target_network",
    )

    target_network.set_weights(
        evaluation_network.get_weights()
    )

    return (
        evaluation_network,
        target_network,
    )


# ======================================================================
# TARGET NETWORK UPDATE
# ======================================================================

def update_target_network(
    evaluation_network: tf.keras.Model,
    target_network: tf.keras.Model,
) -> None:
    """
    Copy evaluation-network weights into target network.
    """

    target_network.set_weights(
        evaluation_network.get_weights()
    )


# ======================================================================
# MODEL VALIDATION
# ======================================================================

def validate_model(
    model: tf.keras.Model,
    expected_state_size: int | None = None,
    expected_num_actions: int | None = None,
) -> dict:
    """
    Validate architecture and model output.
    """

    expected_state_size = int(
        cfg.STATE_SIZE
        if expected_state_size is None
        else expected_state_size
    )

    expected_num_actions = int(
        cfg.NUM_ACTIONS
        if expected_num_actions is None
        else expected_num_actions
    )

    dense_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Dense,
        )
    ]

    checks = {}

    checks[
        "input_shape_valid"
    ] = (
        model.input_shape
        ==
        (
            None,
            expected_state_size,
        )
    )

    checks[
        "output_shape_valid"
    ] = (
        model.output_shape
        ==
        (
            None,
            expected_num_actions,
        )
    )

    checks[
        "dense_layer_count_valid"
    ] = (
        len(
            dense_layers
        )
        ==
        3
    )

    if len(
        dense_layers
    ) >= 3:

        checks[
            "hidden_layer_1_units_valid"
        ] = (
            dense_layers[
                0
            ].units
            ==
            cfg.HIDDEN_UNITS[
                0
            ]
        )

        checks[
            "hidden_layer_2_units_valid"
        ] = (
            dense_layers[
                1
            ].units
            ==
            cfg.HIDDEN_UNITS[
                1
            ]
        )

        checks[
            "output_units_valid"
        ] = (
            dense_layers[
                2
            ].units
            ==
            expected_num_actions
        )

    else:

        checks[
            "hidden_layer_1_units_valid"
        ] = False

        checks[
            "hidden_layer_2_units_valid"
        ] = False

        checks[
            "output_units_valid"
        ] = False

    sample_state = np.zeros(
        (
            1,
            expected_state_size,
        ),
        dtype=np.float32,
    )

    sample_q_values = model(
        sample_state,
        training=False,
    ).numpy()

    checks[
        "sample_output_shape_valid"
    ] = (
        sample_q_values.shape
        ==
        (
            1,
            expected_num_actions,
        )
    )

    checks[
        "sample_output_finite"
    ] = bool(
        np.isfinite(
            sample_q_values
        ).all()
    )

    return checks


# ======================================================================
# MODEL SUMMARY DATA
# ======================================================================

def get_model_summary_dict(
    model: tf.keras.Model,
) -> dict:
    """
    Return architecture metadata for reporting.
    """

    dense_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Dense,
        )
    ]

    return {

        "model_name":
            model.name,

        "state_size":
            cfg.STATE_SIZE,

        "state_features":
            list(
                cfg.STATE_FEATURES
            ),

        "num_actions":
            cfg.NUM_ACTIONS,

        "actions":
            {
                str(
                    key
                ):
                    value
                for key, value
                in cfg.ACTION_NAMES.items()
            },

        "hidden_units":
            [
                int(
                    unit
                )
                for unit
                in cfg.HIDDEN_UNITS
            ],

        "hidden_activation":
            cfg.HIDDEN_ACTIVATION,

        "output_activation":
            cfg.OUTPUT_ACTIVATION,

        "learning_rate":
            cfg.LEARNING_RATE,

        "optimizer":
            cfg.OPTIMIZER_NAME,

        "loss_function":
            cfg.LOSS_FUNCTION,

        "parameter_count":
            int(
                model.count_params()
            ),

        "dense_layers":
            [
                {
                    "name":
                        layer.name,

                    "units":
                        int(
                            layer.units
                        ),

                    "activation":
                        layer.activation.__name__,
                }
                for layer in dense_layers
            ],

    }


# ======================================================================
# MODEL SMOKE TEST
# ======================================================================

def run_dqn_model_smoke_test() -> dict:
    """
    Build and validate both DQN networks.
    """

    print()

    print(
        "=" * 100
    )

    print(
        "AGENT 4 - DQN MODEL SMOKE TEST"
    )

    print(
        "=" * 100
    )

    print()

    seed = set_random_seeds()

    print(
        f"TensorFlow version     : {tf.__version__}"
    )

    print(
        f"Random seed            : {seed}"
    )

    print(
        f"State size             : {cfg.STATE_SIZE}"
    )

    print(
        f"Number of actions      : {cfg.NUM_ACTIONS}"
    )

    print(
        f"Hidden units           : {cfg.HIDDEN_UNITS}"
    )

    print(
        f"Learning rate          : {cfg.LEARNING_RATE}"
    )

    print(
        f"Optimizer              : {cfg.OPTIMIZER_NAME}"
    )

    print(
        f"Loss                   : {cfg.LOSS_FUNCTION}"
    )

    print()

    # ------------------------------------------------------------------
    # Build network pair
    # ------------------------------------------------------------------

    (
        evaluation_network,
        target_network,
    ) = build_dqn_network_pair()

    print(
        "=" * 100
    )

    print(
        "EVALUATION NETWORK"
    )

    print(
        "=" * 100
    )

    print()

    evaluation_network.summary()

    # ------------------------------------------------------------------
    # Validate evaluation model
    # ------------------------------------------------------------------

    evaluation_checks = validate_model(
        evaluation_network
    )

    # ------------------------------------------------------------------
    # Validate target model
    # ------------------------------------------------------------------

    target_checks = validate_model(
        target_network
    )

    # ------------------------------------------------------------------
    # Verify initial weight equality
    # ------------------------------------------------------------------

    evaluation_weights = (
        evaluation_network.get_weights()
    )

    target_weights = (
        target_network.get_weights()
    )

    initial_weights_identical = all(

        np.array_equal(
            eval_weight,
            target_weight,
        )

        for eval_weight, target_weight
        in zip(
            evaluation_weights,
            target_weights,
        )

    )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    rng = np.random.default_rng(
        cfg.RANDOM_SEED
    )

    test_states = rng.normal(
        size=(
            cfg.BATCH_SIZE,
            cfg.STATE_SIZE,
        )
    ).astype(
        np.float32
    )

    eval_q_values = (
        evaluation_network(
            test_states,
            training=False,
        )
        .numpy()
    )

    target_q_values = (
        target_network(
            test_states,
            training=False,
        )
        .numpy()
    )

    forward_pass_same = bool(
        np.allclose(
            eval_q_values,
            target_q_values,
            rtol=0.0,
            atol=1e-7,
        )
    )

    checks = {

        **{
            f"evaluation_{key}":
                value
            for key, value
            in evaluation_checks.items()
        },

        **{
            f"target_{key}":
                value
            for key, value
            in target_checks.items()
        },

        "initial_weights_identical":
            initial_weights_identical,

        "forward_pass_same":
            forward_pass_same,

        "batch_output_shape_valid":
            eval_q_values.shape
            ==
            (
                cfg.BATCH_SIZE,
                cfg.NUM_ACTIONS,
            ),

        "batch_output_finite":
            bool(
                np.isfinite(
                    eval_q_values
                ).all()
            ),

    }

    print()

    print(
        "=" * 100
    )

    print(
        "DQN MODEL VALIDATION"
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
        f"Input shape            : "
        f"{evaluation_network.input_shape}"
    )

    print(
        f"Output shape           : "
        f"{evaluation_network.output_shape}"
    )

    print(
        f"Trainable parameters   : "
        f"{evaluation_network.count_params():,}"
    )

    print()

    print(
        "Sample Q-values:"
    )

    print(
        eval_q_values[
            :5
        ]
    )

    if not overall_pass:

        failed_checks = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        raise ValueError(
            "DQN model validation failed:\n"
            +
            "\n".join(
                f"  - {name}"
                for name in failed_checks
            )
        )

    # ------------------------------------------------------------------
    # Save architecture report
    # ------------------------------------------------------------------

    cfg.REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        cfg.REPORT_DIR
        /
        "dqn_model_summary.json"
    )

    report = {

        "evaluation_network":
            get_model_summary_dict(
                evaluation_network
            ),

        "target_network":
            get_model_summary_dict(
                target_network
            ),

        "checks":
            {
                key:
                    bool(
                        value
                    )
                for key, value
                in checks.items()
            },

        "paper_alignment_note":
            (
                "Two hidden layers with 64 neurons each, "
                "Adam optimizer, and learning rate 0.0001 "
                "follow the paper. Three output Q-values are "
                "used for HOLD, BUY, and SELL because the paper's "
                "separate mention of seven output nodes conflicts "
                "with its explicit three-action definition."
            ),

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
        f"Model report saved     : {report_path}"
    )

    print()

    print(
        "=" * 100
    )

    print(
        "DQN MODEL STATUS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        "Evaluation network     : COMPLETE"
    )

    print(
        "Target network         : COMPLETE"
    )

    print(
        "7 -> 64 -> 64 -> 3     : COMPLETE"
    )

    print(
        "Adam optimizer         : COMPLETE"
    )

    print(
        "Target synchronization : COMPLETE"
    )

    print()

    print(
        "DQN MODEL STATUS: COMPLETE"
    )

    return {

        "evaluation_network":
            evaluation_network,

        "target_network":
            target_network,

        "checks":
            checks,

        "report_path":
            report_path,

    }


# ======================================================================
# DIRECT EXECUTION
# ======================================================================

if __name__ == "__main__":

    run_dqn_model_smoke_test()