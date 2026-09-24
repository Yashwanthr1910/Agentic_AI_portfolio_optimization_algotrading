"""
dilated_lstm.py

Agent 2 - Trend Prediction Agent

This module defines the Dilated LSTM component used by the
Trend Prediction model.

Important
---------
This file does NOT load training data directly.

Data loading is handled later by trainer.py.

Pipeline:

sequences.npz
    ↓
trainer.py
    ↓
trend_model.py
    ↓
DilatedLSTMEncoder
"""

from pathlib import Path
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# PROJECT SETUP
# ============================================================

TREND_AGENT_ROOT = Path(__file__).resolve().parents[1]

if str(TREND_AGENT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(TREND_AGENT_ROOT),
    )


# ============================================================
# CONFIG
# ============================================================

from config import config as cfg


SEQUENCE_LENGTH = getattr(
    cfg,
    "SEQUENCE_LENGTH",
    60,
)

TREND_FEATURE_COLUMNS = getattr(
    cfg,
    "TREND_FEATURE_COLUMNS",
    [],
)

LSTM_UNITS = getattr(
    cfg,
    "LSTM_UNITS",
    64,
)

LSTM_DROPOUT = getattr(
    cfg,
    "LSTM_DROPOUT",
    0.20,
)

LSTM_RECURRENT_DROPOUT = getattr(
    cfg,
    "LSTM_RECURRENT_DROPOUT",
    0.0,
)

DILATION_RATES = getattr(
    cfg,
    "DILATION_RATES",
    [1, 2, 4],
)

RANDOM_SEED = getattr(
    cfg,
    "RANDOM_SEED",
    42,
)


tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# DILATED LSTM LAYER
# ============================================================

@keras.utils.register_keras_serializable(
    package="TrendPredictionAgent"
)
class DilatedLSTMLayer(layers.Layer):
    """
    Dilated LSTM layer.

    Dilation determines the temporal skip distance.

    Example:
        dilation = 1
            0,1,2,3,4,...

        dilation = 2
            0,2,4,...
            1,3,5,...

        dilation = 4
            0,4,8,...
            1,5,9,...
            2,6,10,...
            3,7,11,...
    """

    def __init__(
        self,
        units,
        dilation_rate=1,
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=True,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        self.units = int(
            units
        )

        self.dilation_rate = int(
            dilation_rate
        )

        self.dropout = float(
            dropout
        )

        self.recurrent_dropout = float(
            recurrent_dropout
        )

        self.return_sequences = bool(
            return_sequences
        )

        if self.units <= 0:

            raise ValueError(
                "units must be greater than 0."
            )

        if self.dilation_rate <= 0:

            raise ValueError(
                "dilation_rate must be greater than 0."
            )

        self.lstm = layers.LSTM(
            units=self.units,
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            name=f"lstm_dilation_{self.dilation_rate}",
        )


    def build(
        self,
        input_shape,
    ):

        if len(input_shape) != 3:

            raise ValueError(
                "DilatedLSTMLayer expects input shape "
                "(batch, timesteps, features)."
            )

        timesteps = input_shape[1]

        if timesteps is None:

            raise ValueError(
                "Timesteps must be known."
            )

        if (
            int(timesteps)
            % self.dilation_rate
            != 0
        ):

            raise ValueError(
                f"Timesteps {timesteps} must be divisible by "
                f"dilation rate {self.dilation_rate}."
            )

        super().build(
            input_shape
        )


    def call(
        self,
        inputs,
        training=None,
    ):

        dilation = self.dilation_rate

        # ----------------------------------------------------
        # Standard LSTM
        # ----------------------------------------------------

        if dilation == 1:

            output = self.lstm(
                inputs,
                training=training,
            )

        else:

            # ------------------------------------------------
            # Create interleaved subsequences
            # ------------------------------------------------

            subsequences = []

            for offset in range(
                dilation
            ):

                subsequence = inputs[
                    :,
                    offset::dilation,
                    :,
                ]

                subsequences.append(
                    subsequence
                )

            # ------------------------------------------------
            # Shape:
            # batch, dilation, sub_length, features
            # ------------------------------------------------

            stacked = tf.stack(
                subsequences,
                axis=1,
            )

            shape = tf.shape(
                stacked
            )

            batch_size = shape[0]
            sub_length = shape[2]
            feature_count = shape[3]

            # ------------------------------------------------
            # Merge batch and dilation dimensions
            # ------------------------------------------------

            reshaped = tf.reshape(
                stacked,
                (
                    batch_size * dilation,
                    sub_length,
                    feature_count,
                ),
            )

            # ------------------------------------------------
            # Process each dilated subsequence
            # ------------------------------------------------

            lstm_output = self.lstm(
                reshaped,
                training=training,
            )

            # ------------------------------------------------
            # Restore dimensions
            # ------------------------------------------------

            restored = tf.reshape(
                lstm_output,
                (
                    batch_size,
                    dilation,
                    sub_length,
                    self.units,
                ),
            )

            # ------------------------------------------------
            # Reorder back into chronological order
            # ------------------------------------------------

            restored = tf.transpose(
                restored,
                perm=[
                    0,
                    2,
                    1,
                    3,
                ],
            )

            output = tf.reshape(
                restored,
                (
                    batch_size,
                    sub_length * dilation,
                    self.units,
                ),
            )

        if not self.return_sequences:

            output = output[
                :,
                -1,
                :,
            ]

        return output


    def compute_output_shape(
        self,
        input_shape,
    ):

        if self.return_sequences:

            return (
                input_shape[0],
                input_shape[1],
                self.units,
            )

        return (
            input_shape[0],
            self.units,
        )


    def get_config(
        self,
    ):

        config = super().get_config()

        config.update(
            {
                "units": self.units,
                "dilation_rate": self.dilation_rate,
                "dropout": self.dropout,
                "recurrent_dropout": self.recurrent_dropout,
                "return_sequences": self.return_sequences,
            }
        )

        return config


# ============================================================
# DILATED LSTM ENCODER
# ============================================================

@keras.utils.register_keras_serializable(
    package="TrendPredictionAgent"
)
class DilatedLSTMEncoder(layers.Layer):
    """
    Stack multiple dilated LSTM layers.

    Default dilation rates:
        [1, 2, 4]
    """

    def __init__(
        self,
        units=LSTM_UNITS,
        dilation_rates=None,
        dropout=LSTM_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        residual_connections=True,
        use_layer_norm=True,
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        self.units = int(
            units
        )

        if dilation_rates is None:

            dilation_rates = DILATION_RATES

        self.dilation_rates = [
            int(rate)
            for rate in dilation_rates
        ]

        self.dropout = float(
            dropout
        )

        self.recurrent_dropout = float(
            recurrent_dropout
        )

        self.residual_connections = bool(
            residual_connections
        )

        self.use_layer_norm = bool(
            use_layer_norm
        )

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        self.input_projection = layers.Dense(
            self.units,
            name="input_projection",
        )

        # ----------------------------------------------------
        # Dilated LSTM stack
        # ----------------------------------------------------

        self.dilated_layers = []

        self.normalization_layers = []

        for index, rate in enumerate(
            self.dilation_rates
        ):

            self.dilated_layers.append(
                DilatedLSTMLayer(
                    units=self.units,
                    dilation_rate=rate,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True,
                    name=(
                        f"dilated_lstm_"
                        f"{index + 1}_"
                        f"d{rate}"
                    ),
                )
            )

            if self.use_layer_norm:

                self.normalization_layers.append(
                    layers.LayerNormalization(
                        epsilon=1e-6,
                        name=(
                            f"dilated_norm_"
                            f"{index + 1}"
                        ),
                    )
                )

            else:

                self.normalization_layers.append(
                    None
                )


    def call(
        self,
        inputs,
        training=None,
    ):

        # ----------------------------------------------------
        # Convert 30 features → LSTM hidden dimension
        # ----------------------------------------------------

        x = self.input_projection(
            inputs
        )

        # ----------------------------------------------------
        # Dilated stack
        # ----------------------------------------------------

        for (
            dilated_layer,
            norm_layer,
        ) in zip(
            self.dilated_layers,
            self.normalization_layers,
        ):

            residual = x

            x = dilated_layer(
                x,
                training=training,
            )

            if self.residual_connections:

                x = x + residual

            if norm_layer is not None:

                x = norm_layer(
                    x
                )

        return x


    def compute_output_shape(
        self,
        input_shape,
    ):

        return (
            input_shape[0],
            input_shape[1],
            self.units,
        )


    def get_config(
        self,
    ):

        config = super().get_config()

        config.update(
            {
                "units": self.units,
                "dilation_rates": self.dilation_rates,
                "dropout": self.dropout,
                "recurrent_dropout": self.recurrent_dropout,
                "residual_connections": self.residual_connections,
                "use_layer_norm": self.use_layer_norm,
            }
        )

        return config


# ============================================================
# MODEL BUILDER
# ============================================================

def build_dilated_lstm_model(
    sequence_length=None,
    feature_count=None,
):
    """
    Build standalone Dilated LSTM model for testing.
    """

    if sequence_length is None:

        sequence_length = SEQUENCE_LENGTH

    if feature_count is None:

        feature_count = len(
            TREND_FEATURE_COLUMNS
        )

    inputs = keras.Input(
        shape=(
            sequence_length,
            feature_count,
        ),
        name="trend_sequence_input",
    )

    outputs = DilatedLSTMEncoder(
        units=LSTM_UNITS,
        dilation_rates=DILATION_RATES,
        dropout=LSTM_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        name="dilated_lstm_encoder",
    )(
        inputs
    )

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="dilated_lstm_component",
    )

    return model


# ============================================================
# TEST
# ============================================================

def test_dilated_lstm():

    print()

    print("=" * 75)
    print("AGENT 2 - DILATED LSTM COMPONENT TEST")
    print("=" * 75)

    print()

    feature_count = len(
        TREND_FEATURE_COLUMNS
    )

    if feature_count == 0:

        raise ValueError(
            "TREND_FEATURE_COLUMNS is empty."
        )

    print(
        f"Sequence length : {SEQUENCE_LENGTH}"
    )

    print(
        f"Feature count   : {feature_count}"
    )

    print(
        f"LSTM units      : {LSTM_UNITS}"
    )

    print(
        f"Dilation rates  : {DILATION_RATES}"
    )

    print()

    model = build_dilated_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        feature_count=feature_count,
    )

    model.summary()

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    batch_size = 4

    dummy_input = tf.random.normal(
        (
            batch_size,
            SEQUENCE_LENGTH,
            feature_count,
        )
    )

    output = model(
        dummy_input,
        training=False,
    )

    print()

    print(
        f"Test input shape  : "
        f"{dummy_input.shape}"
    )

    print(
        f"Test output shape : "
        f"{output.shape}"
    )

    expected_shape = (
        batch_size,
        SEQUENCE_LENGTH,
        LSTM_UNITS,
    )

    if tuple(
        output.shape
    ) != expected_shape:

        raise ValueError(
            "Unexpected Dilated LSTM output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Actual  : {tuple(output.shape)}"
        )

    if bool(
        tf.reduce_any(
            tf.math.is_nan(
                output
            )
        )
    ):

        raise ValueError(
            "NaN detected in output."
        )

    print()

    print(
        "Dilated LSTM forward pass PASSED."
    )

    print()

    print("=" * 75)
    print("DILATED LSTM COMPONENT READY")
    print("=" * 75)

    print()

    print(
        "NEXT STAGE: transformer.py"
    )

    print()

    return model


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    test_dilated_lstm()
