"""
Transformer encoder component for Agent 2 - Trend Prediction Agent.

Paper-aligned role:
- Receives sequence representations produced by the Dilated LSTM.
- Applies multi-head self-attention.
- Uses residual Add & Norm.
- Applies a feed-forward network with ReLU.
- Uses a second residual Add & Norm.

The paper does not provide every implementation hyperparameter,
so configurable defaults are used from config.py.
"""

from pathlib import Path
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------
# Make project imports work when this file is executed directly.
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
TREND_AGENT_ROOT = CURRENT_FILE.parents[1]

if str(TREND_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(TREND_AGENT_ROOT))

from config import config as cfg


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL_DIM = int(getattr(cfg, "LSTM_UNITS", 64))
NUM_HEADS = int(getattr(cfg, "TRANSFORMER_NUM_HEADS", 4))
KEY_DIM = int(getattr(cfg, "TRANSFORMER_KEY_DIM", 32))
FF_DIM = int(getattr(cfg, "TRANSFORMER_FF_DIM", 128))
DROPOUT_RATE = float(getattr(cfg, "TRANSFORMER_DROPOUT", 0.20))
NUM_BLOCKS = int(getattr(cfg, "TRANSFORMER_BLOCKS", 2))
SEQUENCE_LENGTH = int(getattr(cfg, "SEQUENCE_LENGTH", 60))
RANDOM_SEED = int(getattr(cfg, "RANDOM_SEED", 42))

tf.random.set_seed(RANDOM_SEED)


@keras.utils.register_keras_serializable(
    package="TrendPredictionAgent",
    name="TransformerEncoderBlock",
)
class TransformerEncoderBlock(layers.Layer):
    """
    Transformer encoder block:

        Input
          |
          v
    Multi-Head Self Attention
          |
        Dropout
          |
    Add + LayerNorm
          |
          v
    Feed-Forward Network
          |
        Dropout
          |
    Add + LayerNorm
          |
        Output
    """

    def __init__(
        self,
        model_dim=64,
        num_heads=4,
        key_dim=32,
        ff_dim=128,
        dropout_rate=0.20,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.ff_dim = int(ff_dim)
        self.dropout_rate = float(dropout_rate)

        self.self_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            name="multi_head_self_attention",
        )

        self.attention_dropout = layers.Dropout(
            self.dropout_rate,
            name="attention_dropout",
        )

        self.attention_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name="attention_add_norm",
        )

        self.feed_forward = keras.Sequential(
            [
                layers.Dense(
                    self.ff_dim,
                    activation="relu",
                    name="ffn_dense_relu",
                ),
                layers.Dropout(
                    self.dropout_rate,
                    name="ffn_internal_dropout",
                ),
                layers.Dense(
                    self.model_dim,
                    name="ffn_projection",
                ),
            ],
            name="feed_forward_network",
        )

        self.ffn_dropout = layers.Dropout(
            self.dropout_rate,
            name="ffn_output_dropout",
        )

        self.ffn_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name="ffn_add_norm",
        )

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass.

        Expected input:
            (batch_size, sequence_length, model_dim)

        Output:
            same shape as input
        """

        attention_output = self.self_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=mask,
            training=training,
        )

        attention_output = self.attention_dropout(
            attention_output,
            training=training,
        )

        # First residual connection: Add & Norm
        x = self.attention_norm(inputs + attention_output)

        ffn_output = self.feed_forward(
            x,
            training=training,
        )

        ffn_output = self.ffn_dropout(
            ffn_output,
            training=training,
        )

        # Second residual connection: Add & Norm
        output = self.ffn_norm(x + ffn_output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@keras.utils.register_keras_serializable(
    package="TrendPredictionAgent",
    name="TransformerEncoder",
)
class TransformerEncoder(layers.Layer):
    """
    Stack of Transformer encoder blocks.
    """

    def __init__(
        self,
        model_dim=64,
        num_heads=4,
        key_dim=32,
        ff_dim=128,
        dropout_rate=0.20,
        num_blocks=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.ff_dim = int(ff_dim)
        self.dropout_rate = float(dropout_rate)
        self.num_blocks = int(num_blocks)

        self.blocks = [
            TransformerEncoderBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                key_dim=self.key_dim,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                name=f"transformer_block_{i + 1}",
            )
            for i in range(self.num_blocks)
        ]

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for block in self.blocks:
            x = block(
                x,
                training=training,
                mask=mask,
            )

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "num_blocks": self.num_blocks,
            }
        )
        return config


def build_transformer_component():
    """
    Standalone Transformer component model for testing.

    Expected input from Dilated LSTM:
        (batch, 60, 64)

    Expected output:
        (batch, 60, 64)
    """

    inputs = keras.Input(
        shape=(SEQUENCE_LENGTH, MODEL_DIM),
        name="dilated_lstm_sequence",
    )

    outputs = TransformerEncoder(
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        key_dim=KEY_DIM,
        ff_dim=FF_DIM,
        dropout_rate=DROPOUT_RATE,
        num_blocks=NUM_BLOCKS,
        name="transformer_encoder",
    )(inputs)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="transformer_component",
    )


def run_component_test():
    print()
    print("=" * 75)
    print("AGENT 2 - TRANSFORMER COMPONENT TEST")
    print("=" * 75)
    print()

    print(f"Sequence length     : {SEQUENCE_LENGTH}")
    print(f"Model dimension     : {MODEL_DIM}")
    print(f"Attention heads     : {NUM_HEADS}")
    print(f"Key dimension       : {KEY_DIM}")
    print(f"Feed-forward size   : {FF_DIM}")
    print(f"Transformer blocks  : {NUM_BLOCKS}")
    print(f"Dropout rate        : {DROPOUT_RATE}")
    print()

    model = build_transformer_component()
    model.summary()

    batch_size = 4

    test_input = tf.random.normal(
        shape=(
            batch_size,
            SEQUENCE_LENGTH,
            MODEL_DIM,
        )
    )

    test_output = model(
        test_input,
        training=False,
    )

    print()
    print(f"Test input shape  : {test_input.shape}")
    print(f"Test output shape : {test_output.shape}")

    expected_shape = (
        batch_size,
        SEQUENCE_LENGTH,
        MODEL_DIM,
    )

    assert tuple(test_output.shape) == expected_shape, (
        f"Unexpected Transformer output shape. "
        f"Expected {expected_shape}, got {tuple(test_output.shape)}"
    )

    assert bool(
        tf.reduce_all(tf.math.is_finite(test_output)).numpy()
    ), "Transformer output contains NaN or infinity."

    print()
    print("Transformer forward pass PASSED.")

    print()
    print("=" * 75)
    print("TRANSFORMER COMPONENT READY")
    print("=" * 75)
    print()
    print("NEXT STAGE: progressive_attention.py")
    print()


if __name__ == "__main__":
    run_component_test()
