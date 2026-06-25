"""
Unit tests for preprocessing pipeline.

These tests verify core preprocessing functionality without requiring
the full dataset download. They use small synthetic arrays.
"""

import numpy as np
import pytest
from src.data.preprocessing import (
    normalize_data,
    reduce_dimensions,
    extract_patches,
    generate_rgb_composite,
    split_dataset,
)


class TestNormalization:
    """Tests for spectral normalization."""

    def test_standard_normalization_shape(self):
        """Output shape should match input shape."""
        data = np.random.rand(10, 10, 50).astype(np.float32)
        normalized, scaler = normalize_data(data, method="standard")
        assert normalized.shape == data.shape

    def test_standard_normalization_stats(self):
        """After standard normalization, mean ≈ 0 and std ≈ 1."""
        data = np.random.rand(20, 20, 30).astype(np.float32) * 100
        normalized, _ = normalize_data(data, method="standard")
        pixels = normalized.reshape(-1, 30)
        np.testing.assert_allclose(pixels.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(pixels.std(axis=0), 1.0, atol=1e-5)

    def test_minmax_normalization_range(self):
        """After minmax normalization, values should be in [0, 1]."""
        data = np.random.rand(10, 10, 20).astype(np.float32) * 500
        normalized, _ = normalize_data(data, method="minmax")
        assert normalized.min() >= -1e-6
        assert normalized.max() <= 1.0 + 1e-6

    def test_invalid_method_raises(self):
        """Unknown normalization method should raise ValueError."""
        data = np.random.rand(5, 5, 10).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown normalization"):
            normalize_data(data, method="invalid")


class TestDimensionalityReduction:
    """Tests for FastICA / PCA dimensionality reduction."""

    def test_pca_output_shape(self):
        """PCA should reduce to n_components."""
        data = np.random.rand(20, 20, 50).astype(np.float32)
        reduced, reducer = reduce_dimensions(data, method="pca", n_components=10)
        assert reduced.shape == (20, 20, 10)
        assert reducer is not None

    def test_fastica_output_shape(self):
        """FastICA should reduce to n_components."""
        # FastICA needs enough samples and variance
        data = np.random.rand(30, 30, 50).astype(np.float32)
        reduced, reducer = reduce_dimensions(
            data, method="fastica", n_components=10
        )
        assert reduced.shape == (30, 30, 10)

    def test_none_method_returns_original(self):
        """Method 'none' should return data unchanged."""
        data = np.random.rand(10, 10, 30).astype(np.float32)
        reduced, reducer = reduce_dimensions(data, method="none")
        np.testing.assert_array_equal(reduced, data)
        assert reducer is None


class TestPatchExtraction:
    """Tests for spatial patch extraction."""

    def test_patch_shape(self):
        """Extracted patches should have correct dimensions."""
        data = np.random.rand(20, 20, 10).astype(np.float32)
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5, 5] = 1  # Single labeled pixel
        labels[10, 10] = 2

        patches, patch_labels = extract_patches(data, labels, patch_size=5)
        assert patches.shape == (2, 5, 5, 10)
        assert patch_labels.shape == (2,)

    def test_labels_are_zero_indexed(self):
        """Patch labels should be converted to 0-indexed."""
        data = np.random.rand(15, 15, 5).astype(np.float32)
        labels = np.zeros((15, 15), dtype=np.int32)
        labels[7, 7] = 3  # Class 3 in original → should become 2

        _, patch_labels = extract_patches(data, labels, patch_size=3)
        assert patch_labels[0] == 2  # 0-indexed

    def test_background_excluded(self):
        """Background pixels (label=0) should not produce patches."""
        data = np.random.rand(10, 10, 5).astype(np.float32)
        labels = np.zeros((10, 10), dtype=np.int32)
        # Only 3 labeled pixels
        labels[2, 3] = 1
        labels[5, 5] = 2
        labels[8, 8] = 1

        patches, _ = extract_patches(data, labels, patch_size=3)
        assert len(patches) == 3


class TestRGBComposite:
    """Tests for false-color RGB composite generation."""

    def test_output_shape_and_range(self):
        """RGB composite should be (H, W, 3) with values in [0, 1]."""
        data = np.random.randn(30, 30, 10).astype(np.float32)
        rgb = generate_rgb_composite(data)
        assert rgb.shape == (30, 30, 3)
        assert rgb.min() >= -1e-6
        assert rgb.max() <= 1.0 + 1e-6


class TestDatasetSplit:
    """Tests for stratified train/val/test splitting."""

    def test_split_sizes(self):
        """Split sizes should approximately match requested ratios."""
        n = 1000
        patches = np.random.rand(n, 5, 5, 3).astype(np.float32)
        labels = np.random.randint(0, 5, size=n)

        splits = split_dataset(patches, labels, test_ratio=0.2, val_ratio=0.1)

        total = sum(len(s[1]) for s in splits.values())
        assert total == n

        # Approximate size checks (±5% tolerance)
        assert abs(len(splits["test"][1]) / n - 0.2) < 0.05
        assert abs(len(splits["val"][1]) / n - 0.1) < 0.05

    def test_no_data_leakage(self):
        """Train, val, and test sets should not overlap."""
        n = 500
        patches = np.arange(n).reshape(n, 1, 1, 1).astype(np.float32)
        labels = np.random.randint(0, 3, size=n)

        splits = split_dataset(patches, labels)

        train_ids = set(splits["train"][0].flatten().tolist())
        val_ids = set(splits["val"][0].flatten().tolist())
        test_ids = set(splits["test"][0].flatten().tolist())

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0
