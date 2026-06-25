"""
Robust loss functions for highly imbalanced datasets.

Changes vs original losses.py:
  - Added effective_number_class_balanced_focal_loss() (beta=0.999)
  - Kept original focal_loss() for backward compatibility
  - Use default gamma 1.5

Reference: Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples"
         Effective number formula (Cui et al. 2019)
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Union


def focal_loss(
    gamma: float = 1.5,  # optimal for this dataset
    alpha: Optional[Union[float, np.ndarray]] = None,
    label_smoothing: float = 0.0,
) -> tf.keras.losses.Loss:
    """Create a focal loss function.

    Args:
        gamma: Focusing parameter. Default 1.5.
               Higher values down-weight easy examples more aggressively.
        alpha: Class balancing weights. Can be:
               - None: no class weighting
               - float: uniform weight
               - np.ndarray: per-class weights of shape (num_classes,)
        label_smoothing: Label smoothing factor (0 = no smoothing).

    Returns:
        Keras-compatible loss function.
    """
    def _focal_loss(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Apply label smoothing if specified
        if label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = (
                y_true * (1.0 - label_smoothing)
                + label_smoothing / num_classes
            )

        # Cross-entropy component
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - y_pred, gamma)

        # Apply class balancing weights alpha_t
        if alpha is not None:
            if isinstance(alpha, np.ndarray):
                alpha_tensor = tf.constant(alpha, dtype=tf.float32)
                # Broadcast alpha across batch: shape (1, num_classes)
                alpha_factor = tf.expand_dims(alpha_tensor, axis=0)
            else:
                alpha_factor = alpha
            focal_weight = alpha_factor * focal_weight

        # Final focal loss
        loss = focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    return _focal_loss


def class_balanced_focal_loss(
    class_counts: dict,
    gamma: float = 1.5,
    label_smoothing: float = 0.0,
    num_classes: int = None,
) -> tf.keras.losses.Loss:
    """Create focal loss with automatic class-balanced alpha_t weights.

    Computes alpha using INVERSE FREQUENCY (original GitHub formula).
    For effective number weighting (Cui et al. 2019), use
    effective_number_class_balanced_focal_loss() instead.
    """
    if num_classes is None:
        num_classes = max(class_counts.keys()) + 1 if class_counts else 1

    present_classes = len(class_counts)
    total_samples = sum(class_counts.values())

    # Compute inverse frequency weights
    alpha = np.zeros(num_classes, dtype=np.float32)
    for cls_idx, count in class_counts.items():
        if cls_idx < num_classes:
            alpha[cls_idx] = total_samples / (present_classes * count)

    # Normalize so weights sum to present_classes
    alpha_sum = alpha.sum()
    if alpha_sum > 0:
        alpha = alpha / alpha_sum * present_classes

    print(f"  Class-balanced focal loss (inverse freq, gamma={gamma}):")
    for i in sorted(class_counts.keys()):
        if i < num_classes:
            print(f"   Class {i:2d}: weight={alpha[i]:.3f}, samples={class_counts[i]:,}")

    return focal_loss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)


def effective_number_class_balanced_focal_loss(
    class_counts: dict,
    gamma: float = 1.5,
    beta: float = 0.999,
    label_smoothing: float = 0.0,
    num_classes: int = None,
) -> tf.keras.losses.Loss:
    """Create focal loss with EFFECTIVE NUMBER class-balanced weights.

    Uses the effective number of samples formula:
        alpha_t ∝ (1 - beta) / (1 - beta^(n_t))

    where n_t is the number of samples for class t.
    beta=0.999 ensures minority classes get proper attention without
    dominating the loss.

    Reference: Cui et al. (2019)

    Args:
        class_counts: Dict mapping class_index -> sample_count.
        gamma: Focal loss focusing parameter.
        beta: Effective number hyperparameter.
        label_smoothing: Label smoothing factor.
        num_classes: Total number of classes.

    Returns:
        Keras-compatible loss function with effective number class balancing.
    """
    if num_classes is None:
        num_classes = max(class_counts.keys()) + 1 if class_counts else 1

    # Compute effective number weights
    alpha = np.zeros(num_classes, dtype=np.float32)
    for cls_idx, count in class_counts.items():
        if cls_idx < num_classes and count > 0:
            effective_num = (1.0 - beta ** count) / (1.0 - beta)
            alpha[cls_idx] = 1.0 / effective_num

    # Normalize
    alpha_sum = alpha.sum()
    if alpha_sum > 0:
        alpha = alpha / alpha_sum * len(class_counts)

    print(f"  Effective number focal loss (beta={beta}, gamma={gamma}):")
    for i in sorted(class_counts.keys()):
        if i < num_classes:
            en = (1.0 - beta ** class_counts[i]) / (1.0 - beta)
            print(f"   Class {i:2d}: alpha={alpha[i]:.4f}, eff_num={en:.2f}, n={class_counts[i]}")

    return focal_loss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)
