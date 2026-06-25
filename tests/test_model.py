"""
Unit tests for model architecture.

Verifies that the Baseline CNN produces correct output shapes.
"""

import numpy as np
import pytest
import tensorflow as tf
from src.models.baseline_cnn import build_baseline_cnn


class TestBaselineCNN:
    """Tests for the Baseline CNN architecture."""

    def test_output_shape(self):
        """Model output should be (batch, num_classes)."""
        model = build_baseline_cnn(
            input_shape=(25, 25, 3),
            num_classes=16,
        )
        x = tf.random.normal((2, 25, 25, 3))
        pred = model(x, training=False)
        assert pred.shape == (2, 16)

    def test_softmax_output(self):
        """Output probabilities should sum to ~1."""
        model = build_baseline_cnn(
            input_shape=(25, 25, 3),
            num_classes=10,
        )
        x = tf.random.normal((1, 25, 25, 3))
        pred = model(x, training=False)
        np.testing.assert_allclose(
            tf.reduce_sum(pred, axis=-1).numpy(), [1.0], atol=1e-5
        )
