"""
Complete Trend Prediction Model
Agent 2 - Trend Prediction Agent
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

from src.dilated_lstm import DilatedLSTMEncoder
from src.transformer import TransformerEncoder
from src.progressive_attention import ProgressiveAttention


SEQUENCE_LENGTH = int(
    getattr(cfg, "SEQUENCE_LENGTH", 60)
)

FEATURE_COLUMNS = list(
    getattr(cfg, "TREND_FEATURE_COLUMNS", [])
)

NUM_FEATURES = len(FEATURE_COLUMNS)

if NUM_FEATURES == 0:
    NUM_FEATURES = 30


MODEL_DIM = int(
    getattr(cfg, "LSTM_UNITS", 64)
)

LSTM_DROPOUT = float(
    getattr(cfg, "LSTM_DROPOUT", 0.20)
)

LSTM_RECURRENT_DROPOUT = float(
    getattr(cfg, "LSTM_RECURRENT_DROPOUT", 0.0)
)

DILATION_RATES = list(
    getattr(cfg, "DILATION_RATES", [1, 2, 4])
)


TRANSFORMER_NUM_HEADS = int(
    getattr(cfg, "TRANSFORMER_NUM_HEADS", 4)
)

TRANSFORMER_KEY_DIM = int(
    getattr(cfg, "TRANSFORMER_KEY_DIM", 32)
)

TRANSFORMER_FF_DIM = int(
    getattr(cfg, "TRANSFORMER_FF_DIM", 128)
)

TRANSFORMER_DROPOUT = float(
    getattr(cfg, "TRANSFORMER_DROPOUT", 0.20)
)

TRANSFORMER_BLOCKS = int(
    getattr(cfg, "TRANSFORMER_BLOCKS", 2)
)


ATTENTION_HIDDEN_DIM = int(
    getattr(cfg, "ATTENTION_HIDDEN_DIM", 64)
)

ATTENTION_DROPOUT = float(
    getattr(cfg, "ATTENTION_DROPOUT", 0.10)
)

LEARNING_RATE = float(
    getattr(cfg, "LEARNING_RATE", 0.001)
)

RANDOM_SEED = int(
    getattr(cfg, "RANDOM_SEED", 42)
)

NUM_CLASSES = 2

tf.random.set_seed(RANDOM_SEED)


def build_trend_model(
    compile_model=True,
):

    inputs = keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            NUM_FEATURES,
        ),
        name="trend_sequence_input",
    )

    # Continuous-feature embedding/projection
    x = layers.Dense(
        MODEL_DIM,
        activation=None,
        name="feature_embedding",
    )(inputs)

    # Dilated LSTM
    x = DilatedLSTMEncoder(
        units=MODEL_DIM,
        dilation_rates=DILATION_RATES,
        dropout=LSTM_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        name="dilated_lstm_encoder",
    )(x)

    # Transformer
    x = TransformerEncoder(
        model_dim=MODEL_DIM,
        num_heads=TRANSFORMER_NUM_HEADS,
        key_dim=TRANSFORMER_KEY_DIM,
        ff_dim=TRANSFORMER_FF_DIM,
        dropout_rate=TRANSFORMER_DROPOUT,
        num_blocks=TRANSFORMER_BLOCKS,
        name="transformer_encoder",
    )(x)

    # Progressive attention
    context = ProgressiveAttention(
        hidden_dim=ATTENTION_HIDDEN_DIM,
        dropout_rate=ATTENTION_DROPOUT,
        return_attention=False,
        name="progressive_attention",
    )(x)

    # Linear classification layer
    logits = layers.Dense(
        NUM_CLASSES,
        activation=None,
        name="trend_logits",
    )(context)

    # Softmax
    outputs = layers.Softmax(
        name="trend_probability",
    )(logits)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="dilated_lstm_transformer_trend_model",
    )

    if compile_model:

        optimizer = keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(
                    name="accuracy"
                )
            ],
        )

    return model


def validate_model_output(
    test_input,
    predictions,
):

    batch_size = int(
        test_input.shape[0]
    )

    expected_shape = (
        batch_size,
        NUM_CLASSES,
    )

    if tuple(predictions.shape) != expected_shape:

        raise ValueError(
            f"Output shape mismatch. "
            f"Expected {expected_shape}, "
            f"got {tuple(predictions.shape)}"
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(predictions)
        ).numpy()
    ):
        raise ValueError(
            "Predictions contain NaN or infinity."
        )

    if not bool(
        tf.reduce_all(
            predictions >= 0
        ).numpy()
    ):
        raise ValueError(
            "Probability below zero detected."
        )

    if not bool(
        tf.reduce_all(
            predictions <= 1
        ).numpy()
    ):
        raise ValueError(
            "Probability above one detected."
        )

    probability_sums = tf.reduce_sum(
        predictions,
        axis=1,
    )

    tf.debugging.assert_near(
        probability_sums,
        tf.ones_like(probability_sums),
        atol=1e-5,
    )


def run_model_test():

    print()
    print("=" * 78)
    print(
        "AGENT 2 - COMPLETE TREND MODEL TEST"
    )
    print("=" * 78)
    print()

    print(
        f"Sequence length      : {SEQUENCE_LENGTH}"
    )

    print(
        f"Number of features   : {NUM_FEATURES}"
    )

    print(
        f"Model dimension      : {MODEL_DIM}"
    )

    print(
        f"Dilation rates       : {DILATION_RATES}"
    )

    print(
        f"Transformer heads    : {TRANSFORMER_NUM_HEADS}"
    )

    print(
        f"Transformer blocks   : {TRANSFORMER_BLOCKS}"
    )

    print(
        f"Attention hidden dim : {ATTENTION_HIDDEN_DIM}"
    )

    print(
        f"Number of classes    : {NUM_CLASSES}"
    )

    print(
        f"Learning rate        : {LEARNING_RATE}"
    )

    print()

    model = build_trend_model(
        compile_model=True
    )

    model.summary()

    batch_size = 4

    test_input = tf.random.normal(
        (
            batch_size,
            SEQUENCE_LENGTH,
            NUM_FEATURES,
        ),
        seed=RANDOM_SEED,
    )

    predictions = model(
        test_input,
        training=False,
    )

    print()

    print(
        f"Test input shape  : {test_input.shape}"
    )

    print(
        f"Prediction shape  : {predictions.shape}"
    )

    print()
    print(
        "Predicted probabilities:"
    )

    print(
        predictions.numpy()
    )

    validate_model_output(
        test_input,
        predictions,
    )

    probability_sums = tf.reduce_sum(
        predictions,
        axis=1,
    )

    print()
    print(
        "Probability sums:"
    )

    print(
        probability_sums.numpy()
    )

    predicted_classes = tf.argmax(
        predictions,
        axis=1,
    )

    print()
    print(
        "Predicted classes:"
    )

    print(
        predicted_classes.numpy()
    )

    print()

    print(
        f"Model input shape : {model.input_shape}"
    )

    print(
        f"Model output shape: {model.output_shape}"
    )

    print()
    print(
        "Complete Trend Model forward pass PASSED."
    )

    print()
    print("=" * 78)
    print(
        "TREND MODEL READY"
    )
    print("=" * 78)

    print()
    print(
        "NEXT STAGE: trainer.py"
    )
    print()


if __name__ == "__main__":
    run_model_test()
