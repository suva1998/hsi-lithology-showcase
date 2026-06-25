"""
HSI Preprocessing Pipeline.

Implements the preprocessing methodology:
    1. Normalization (standard scaling or min-max)
    2. FastICA dimensionality reduction 
    3. Spatial patch extraction for CNN input
    4. Stratified train/val/test splitting

Key design choice  FastICA over PCA:
    ICA separates statistically independent source signals, which maps
    naturally to independent mineral absorption signatures. PCA maximizes
    variance, which often captures illumination changes rather than
    mineralogical differences. Du et al. (2003) showed 15-20% accuracy
    improvements with ICA for mineral classification.
"""

import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any


def normalize_data(
    data: np.ndarray,
    method: str = "standard",
) -> Tuple[np.ndarray, Any]:
    """Normalize hyperspectral data along the spectral axis.

    Args:
        data: HSI cube of shape (H, W, B).
        method: Normalization method  "standard" (zero mean, unit variance)
                or "minmax" (scale to [0, 1]).

    Returns:
        Tuple of (normalized_data, fitted_scaler).

    Raises:
        ValueError: If method is not "standard" or "minmax".
    """
    h, w, b = data.shape
    # Reshape to (pixels, bands) for sklearn
    pixels = data.reshape(-1, b)

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    pixels_norm = scaler.fit_transform(pixels)
    return pixels_norm.reshape(h, w, b), scaler


def reduce_dimensions(
    data: np.ndarray,
    method: str = "fastica",
    n_components: int = 30,
    fastica_params: Optional[Dict] = None,
) -> Tuple[np.ndarray, Any]:
    """Apply dimensionality reduction to HSI data.

    FastICA finds independent components by
    optimizing W = argmax_W E[G(W^T x)] - 0.5 * E[(W^T x)^2]
    where G is a non-quadratic contrast function.

    Args:
        data: HSI cube of shape (H, W, B)  B spectral bands.
        method: "fastica" (recommended), "pca", or "none".
        n_components: Number of output components.
        fastica_params: Optional dict of FastICA parameters.
            Defaults: algorithm="parallel", fun="logcosh",
                      whiten="unit-variance", max_iter=500.

    Returns:
        Tuple of (reduced_data, fitted_reducer).
            reduced_data: shape (H, W, n_components)
    """
    h, w, b = data.shape
    pixels = data.reshape(-1, b)

    if method == "none":
        return data, None

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == "fastica":
        # Default parameters for robust performance
        params = {
            "n_components": n_components,
            "algorithm": "parallel",
            "whiten": "unit-variance",
            "max_iter": 500,
            "tol": 1e-4,
            "fun": "logcosh",      # G(u) = log(cosh(u))
            "random_state": 42,
        }
        if fastica_params:
            params.update(fastica_params)
        reducer = FastICA(**params)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    # Prevent OOM and massively speed up fitting by subsampling
    max_fit_samples = 100000
    if pixels.shape[0] > max_fit_samples:
        idx = np.random.choice(pixels.shape[0], max_fit_samples, replace=False)
        reducer.fit(pixels[idx])
        pixels_reduced = reducer.transform(pixels)
    else:
        pixels_reduced = reducer.fit_transform(pixels)
        
    return pixels_reduced.reshape(h, w, n_components), reducer


def generate_rgb_composite(
    reduced_data: np.ndarray,
) -> np.ndarray:
    """Create false-color RGB composite from first 3 ICA components.

    Maps IC1R, IC2G, IC3B with
    histogram equalization for visual validation.

    Args:
        reduced_data: shape (H, W, C) where C >= 3.

    Returns:
        RGB image of shape (H, W, 3) with values in [0, 1].
    """
    rgb = reduced_data[:, :, :3].copy()

    # Per-channel histogram equalization to [0, 1]
    for c in range(3):
        channel = rgb[:, :, c]
        p_low, p_high = np.percentile(channel, [2, 98])
        channel = np.clip(channel, p_low, p_high)
        channel = (channel - p_low) / (p_high - p_low + 1e-8)
        rgb[:, :, c] = channel

    return rgb


def extract_patches(
    data: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 25,
    max_samples_per_class: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract spatial patches centered on labeled pixels.

    For each labeled pixel (label > 0), extracts a (patch_size  patch_size)
    neighborhood from the HSI cube. Pixels near the border are zero-padded.

    Args:
        data: HSI cube of shape (H, W, C)  C channels (bands or components).
        labels: Label map of shape (H, W)  0 = background (skipped).
        patch_size: Side length of square patches (must be odd).

    Returns:
        Tuple of (patches, patch_labels).
            patches:      shape (N, patch_size, patch_size, C)
            patch_labels: shape (N,)  0-indexed class labels
    """
    h, w, c = data.shape
    margin = patch_size // 2

    # Pad the data cube with zeros
    padded = np.pad(
        data,
        ((margin, margin), (margin, margin), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    # Collect labeled pixel positions
    labeled_rows, labeled_cols = np.where(labels > 0)
    
    if max_samples_per_class is not None:
        selected_rows, selected_cols = [], []
        # Group by label to enforce class balancing/limits
        for lbl in np.unique(labels[labels > 0]):
            mask = labels[labeled_rows, labeled_cols] == lbl
            class_rows = labeled_rows[mask]
            class_cols = labeled_cols[mask]
            
            if len(class_rows) > max_samples_per_class:
                idx = np.random.choice(len(class_rows), max_samples_per_class, replace=False)
                selected_rows.extend(class_rows[idx])
                selected_cols.extend(class_cols[idx])
            else:
                selected_rows.extend(class_rows)
                selected_cols.extend(class_cols)
        labeled_rows = np.array(selected_rows)
        labeled_cols = np.array(selected_cols)

    n_samples = len(labeled_rows)

    patches = np.zeros((n_samples, patch_size, patch_size, c), dtype=np.float32)
    patch_labels = np.zeros(n_samples, dtype=np.int32)

    for i, (row, col) in enumerate(zip(labeled_rows, labeled_cols)):
        # Extract patch from padded data (offset by margin)
        r, c_ = row + margin, col + margin
        patches[i] = padded[
            r - margin : r + margin + 1,
            c_ - margin : c_ + margin + 1,
            :,
        ]
        # Convert labels to 0-indexed (original: 1-based)
        patch_labels[i] = labels[row, col] - 1

    return patches, patch_labels


def prepare_rgb_patches(
    patches: np.ndarray,
) -> np.ndarray:
    """Convert multi-channel patches to 3-channel RGB for standard CNNs.

    Takes the first 3 components (ICA or PCA) and scales to [0, 255]
    for compatibility with ImageNet-pretrained backbones.
    Uses fully vectorized NumPy operations for maximum speed.

    Args:
        patches: shape (N, H, W, C) where C >= 3.

    Returns:
        RGB patches of shape (N, H, W, 3) scaled to [0, 255].
    """
    rgb = patches[:, :, :, :3].copy()

    # Vectorized per-patch normalization to [0, 255]
    # Calculate min and max along the spatial dimensions (H, W), keeping dims for broadcasting
    c_min = rgb.min(axis=(1, 2), keepdims=True)
    c_max = rgb.max(axis=(1, 2), keepdims=True)
    
    range_mask = (c_max - c_min) > 1e-8
    
    # Apply normalization where range is valid, otherwise set to 0.0
    rgb = np.where(
        range_mask,
        (rgb - c_min) / (c_max - c_min + 1e-8) * 255.0,
        0.0
    )

    return rgb.astype(np.float32)


def split_dataset(
    patches: np.ndarray,
    labels: np.ndarray,
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Stratified train/val/test split preserving class proportions.

    Args:
        patches: shape (N, H, W, C).
        labels: shape (N,)  integer class labels.
        test_ratio: Fraction for test set.
        val_ratio: Fraction for validation set (from remaining after test).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys "train", "val", "test", each containing
        (patches, labels) tuple.
    """
    # First split: train+val vs. test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        patches, labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )

    # Second split: train vs. val
    val_fraction = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_trainval,
    )

    print(f" Dataset split (seed={seed}):")
    print(f"   Train: {len(y_train):,} samples")
    print(f"   Val:   {len(y_val):,} samples")
    print(f"   Test:  {len(y_test):,} samples")

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def preprocess_hsi(
    data: np.ndarray,
    labels: np.ndarray,
    config: Dict,
) -> Dict[str, Any]:
    """Full preprocessing pipeline: normalize  reduce  patch  split.

    This is the main entry point for the preprocessing pipeline.
    Orchestrates all steps from raw HSI cube to train-ready patches.

    Args:
        data: Raw HSI cube of shape (H, W, B).
        labels: Label map of shape (H, W).
        config: Configuration dict (loaded from YAML).

    Returns:
        Dict containing:
            - "splits": train/val/test splits
            - "scaler": fitted normalizer
            - "reducer": fitted ICA/PCA transformer
            - "rgb_composite": false-color visualization
            - "class_counts": per-class sample counts
    """
    prep_cfg = config["preprocessing"]
    ds_cfg = config["dataset"]

    # Step 1: Normalize
    print(" Step 1/4: Normalizing spectral data...")
    data_norm, scaler = normalize_data(
        data, method=prep_cfg.get("normalize_method", "standard")
    )

    # Step 2: Dimensionality reduction (FastICA by default)
    print(" Step 2/4: Applying dimensionality reduction...")
    data_reduced, reducer = reduce_dimensions(
        data_norm,
        method=prep_cfg.get("reduction_method", "fastica"),
        n_components=prep_cfg.get("n_components", 30),
        fastica_params=prep_cfg.get("fastica", None),
    )

    # Generate RGB composite for visualization
    rgb_composite = generate_rgb_composite(data_reduced)

    # Step 3: Extract spatial patches
    print(" Step 3/4: Extracting spatial patches...")
    patches, patch_labels = extract_patches(
        data_reduced,
        labels,
        patch_size=ds_cfg.get("patch_size", 25),
        max_samples_per_class=ds_cfg.get("max_samples_per_class", 1500),
    )

    # Convert to 3-channel RGB for CNN inputs
    patches_rgb = prepare_rgb_patches(patches)

    # Step 4: Stratified split
    print(" Step 4/4: Splitting dataset...")
    splits = split_dataset(
        patches_rgb,
        patch_labels,
        test_ratio=ds_cfg.get("test_ratio", 0.20),
        val_ratio=ds_cfg.get("val_ratio", 0.10),
        seed=config["training"].get("seed", 42),
    )

    # Compute class distribution
    unique, counts = np.unique(patch_labels, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))

    return {
        "splits": splits,
        "scaler": scaler,
        "reducer": reducer,
        "rgb_composite": rgb_composite,
        "class_counts": class_counts,
    }
