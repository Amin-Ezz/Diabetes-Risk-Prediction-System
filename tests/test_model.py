"""Unit tests for multi-output neural network architecture."""

import sys
from pathlib import Path
import unittest
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diabetes_prediction.models.neural_network import build_multioutput_nn


class TestNeuralNetwork(unittest.TestCase):
    """Test suite for neural network architecture and compilation."""

    def test_model_build_and_forward(self):
        batch_size = 4
        n_features = 64
        model = build_multioutput_nn(n_features=n_features)

        # Check model output heads
        self.assertEqual(len(model.outputs), 2)
        self.assertEqual(model.output_names, ["diabetes", "hypertension"])

        # Forward pass with random tensor
        x_dummy = np.random.randn(batch_size, n_features).astype(np.float32)
        preds = model(x_dummy, training=False)

        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0].shape, (batch_size, 1))
        self.assertEqual(preds[1].shape, (batch_size, 1))

        # Check sigmoid range [0, 1]
        self.assertTrue(np.all(preds[0].numpy() >= 0.0) and np.all(preds[0].numpy() <= 1.0))
        self.assertTrue(np.all(preds[1].numpy() >= 0.0) and np.all(preds[1].numpy() <= 1.0))

    def test_custom_layers_architecture(self):
        model = build_multioutput_nn(
            n_features=32,
            hidden_units=[64, 32],
            dropout_rates=[0.2, 0.1],
        )
        self.assertEqual(model.input_shape, (None, 32))
        self.assertEqual(len(model.outputs), 2)


if __name__ == "__main__":
    unittest.main()
