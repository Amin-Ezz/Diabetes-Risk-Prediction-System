"""Multi-output deep neural network architecture for simultaneous Diabetes & Hypertension prediction."""

from pathlib import Path
from typing import List, Optional
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)
from tensorflow.keras.regularizers import l2


def build_multioutput_nn(
    n_features: int = 64,
    hidden_units: Optional[List[int]] = None,
    dropout_rates: Optional[List[float]] = None,
    l2_reg: float = 3e-4,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.05,
) -> Model:
    """Build and compile the multi-output neural network architecture.

    Args:
        n_features: Number of input dimensions (default 64 after imputation + scaling).
        hidden_units: Hidden layer unit counts (default [128, 64, 32]).
        dropout_rates: Dropout rates for each layer (default [0.35, 0.25, 0.15]).
        l2_reg: L2 kernel regularization coefficient.
        learning_rate: Adam optimizer learning rate.
        label_smoothing: BinaryCrossentropy label smoothing epsilon.

    Returns:
        Compiled Keras Model with two Sigmoid output heads: 'diabetes' and 'hypertension'.
    """
    if hidden_units is None:
        hidden_units = [128, 64, 32]
    if dropout_rates is None:
        dropout_rates = [0.35, 0.25, 0.15]

    inputs = Input(shape=(n_features,), name="input")
    x = inputs

    for units, drop in zip(hidden_units, dropout_rates):
        x = Dense(units, kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(drop)(x)

    diabetes_out = Dense(1, activation="sigmoid", name="diabetes")(x)
    htn_out = Dense(1, activation="sigmoid", name="hypertension")(x)

    model = Model(inputs=inputs, outputs=[diabetes_out, htn_out], name="multioutput_diabetes_risk_nn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "diabetes": tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
            "hypertension": tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        },
        metrics={
            "diabetes": ["accuracy", tf.keras.metrics.AUC(name="auc")],
            "hypertension": ["accuracy", tf.keras.metrics.AUC(name="auc")],
        },
    )
    return model


def load_trained_model(model_path: str | Path) -> Model:
    """Load trained Keras model from disk.

    Args:
        model_path: Path to the .keras model file.

    Returns:
        Loaded Keras Model instance.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Please train the model first by running: python scripts/train.py"
        )
    return tf.keras.models.load_model(str(path))
