"""
Progressive Attention component for Agent 2 - Trend Prediction Agent.

Receives Transformer sequence output and learns temporal importance
weights across the sequence.

Input:
    (batch, sequence_length, model_dim)

Output:
    (batch, model_dim)

The paper specifies progressive attention but does not provide enough
implementation detail to reproduce the authors' exact source code.
This is therefore a reproducible implementation of progressive
temporal attention consistent with the described architecture.
"""

from pathlib import Path
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


CURRENT_FILE = Path(__file__).resolve()
TREND_AGENT_ROOT = CURRENT_FILE.parents[1]

if str(TREND_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TREND_AGENT_ROOT))

from config import config as cfg


SEQUENCE_LENGTH = int(getattr(cfg, "SEQUENCE_LENGTH", 60))
MODEL_DIM = int(getattr(cfg, "LSTM_UNITS", 64))

ATTENTION_HIDDEN_DIM = int(
    getattr(cfg, "ATTENTION_HIDDEN_DIM", 64)
)

ATTENTION_DROPOUT = float(
    getattr(cfg, "ATTENTION_DROPOUT", 0.10)
)

RANDOM_SEED = int(
    getattr(cfg, "RANDOM_SEED", 42)
)

tf.random.set_seed(RANDOM_SEED)


@keras.utils.register_keras_serializable(
    package="TrendPredictionAgent",
    name="ProgressiveAttention",
)
class ProgressiveAttention(layers.Layer):

    def __init__(
        self,
        hidden_dim=64,
        dropout_rate=0.10,
        return_attention=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.return_attention = bool(return_attention)

        self.attention_hidden = layers.Dense(
            self.hidden_dim,
            activation="tanh",
            name="attention_hidden",
        )

        self.attention_dropout = layers.Dropout(
            self.dropout_rate,
            name="attention_dropout",
        )

        self.attention_score = layers.Dense(
            1,
            activation=None,
            name="attention_score",
        )


    def build(self, input_shape):

        self.progressive_strength = self.add_weight(
            name="progressive_strength",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
        )

        super().build(input_shape)


    def call(
        self,
        inputs,
        training=None,
        mask=None,
    ):

        if inputs.shape.rank != 3:
            raise ValueError(
                "ProgressiveAttention expects "
                "(batch, sequence, features)."
            )

        hidden = self.attention_hidden(inputs)

        hidden = self.attention_dropout(
            hidden,
            training=training,
        )

        scores = self.attention_score(hidden)

        scores = tf.squeeze(
            scores,
            axis=-1,
        )

        sequence_length = tf.shape(inputs)[1]

        progression = tf.linspace(
            tf.cast(0.0, inputs.dtype),
            tf.cast(1.0, inputs.dtype),
            sequence_length,
        )

        progression = tf.reshape(
            progression,
            (1, -1),
        )

        strength = tf.sigmoid(
            self.progressive_strength
        )

        scores = (
            scores
            + strength * progression
        )

        if mask is not None:

            mask = tf.cast(
                mask,
                tf.bool,
            )

            scores = tf.where(
                mask,
                scores,
                tf.cast(
                    -1e9,
                    scores.dtype,
                ),
            )

        attention_weights = tf.nn.softmax(
            scores,
            axis=1,
        )

        expanded_weights = tf.expand_dims(
            attention_weights,
            axis=-1,
        )

        weighted_sequence = (
            inputs * expanded_weights
        )

        context_vector = tf.reduce_sum(
            weighted_sequence,
            axis=1,
        )

        if self.return_attention:

            return (
                context_vector,
                attention_weights,
            )

        return context_vector


    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
                "return_attention": self.return_attention,
            }
        )

        return config


def build_progressive_attention_component(
    return_attention=False,
):

    inputs = keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            MODEL_DIM,
        ),
        name="transformer_sequence",
    )

    outputs = ProgressiveAttention(
        hidden_dim=ATTENTION_HIDDEN_DIM,
        dropout_rate=ATTENTION_DROPOUT,
        return_attention=return_attention,
        name="progressive_attention",
    )(inputs)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="progressive_attention_component",
    )


def validate_component(
    test_input,
    context_vector,
    attention_weights,
):

    batch_size = int(test_input.shape[0])

    expected_context = (
        batch_size,
        MODEL_DIM,
    )

    expected_attention = (
        batch_size,
        SEQUENCE_LENGTH,
    )

    if tuple(context_vector.shape) != expected_context:

        raise ValueError(
            f"Context shape mismatch. "
            f"Expected {expected_context}, "
            f"got {tuple(context_vector.shape)}"
        )

    if tuple(attention_weights.shape) != expected_attention:

        raise ValueError(
            f"Attention shape mismatch. "
            f"Expected {expected_attention}, "
            f"got {tuple(attention_weights.shape)}"
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(context_vector)
        ).numpy()
    ):
        raise ValueError(
            "Context contains NaN or infinity."
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(attention_weights)
        ).numpy()
    ):
        raise ValueError(
            "Attention contains NaN or infinity."
        )

    attention_sums = tf.reduce_sum(
        attention_weights,
        axis=1,
    )

    tf.debugging.assert_near(
        attention_sums,
        tf.ones_like(attention_sums),
        atol=1e-5,
    )


def run_component_test():

    print()
    print("=" * 75)
    print(
        "AGENT 2 - PROGRESSIVE ATTENTION COMPONENT TEST"
    )
    print("=" * 75)
    print()

    print(
        f"Sequence length      : {SEQUENCE_LENGTH}"
    )

    print(
        f"Model dimension      : {MODEL_DIM}"
    )

    print(
        f"Attention hidden dim : {ATTENTION_HIDDEN_DIM}"
    )

    print(
        f"Attention dropout    : {ATTENTION_DROPOUT}"
    )

    print()

    model = build_progressive_attention_component(
        return_attention=True
    )

    model.summary()

    batch_size = 4

    test_input = tf.random.normal(
        (
            batch_size,
            SEQUENCE_LENGTH,
            MODEL_DIM,
        ),
        seed=RANDOM_SEED,
    )

    context_vector, attention_weights = model(
        test_input,
        training=False,
    )

    print()

    print(
        f"Test input shape       : {test_input.shape}"
    )

    print(
        f"Context vector shape   : {context_vector.shape}"
    )

    print(
        f"Attention weight shape : {attention_weights.shape}"
    )

    validate_component(
        test_input,
        context_vector,
        attention_weights,
    )

    attention_sums = tf.reduce_sum(
        attention_weights,
        axis=1,
    )

    print()
    print("Attention sums:")
    print(attention_sums.numpy())

    attention_layer = model.get_layer(
        "progressive_attention"
    )

    progressive_strength = tf.sigmoid(
        attention_layer.progressive_strength
    )

    print()

    print(
        "Initial progressive strength : "
        f"{float(progressive_strength.numpy()):.6f}"
    )

    print()
    print(
        "Progressive Attention forward pass PASSED."
    )

    print()
    print("=" * 75)
    print(
        "PROGRESSIVE ATTENTION COMPONENT READY"
    )
    print("=" * 75)

    print()
    print("NEXT STAGE: trend_model.py")
    print()


if __name__ == "__main__":
    run_component_test()
