"""
MixUp and geometric augmentation for HSI patches.

Implements MixUp regularization creates virtual training examples by linearly interpolating both inputs and labels. 
This produces smoother decision boundaries and reduces overconfident predictions —critical for imbalanced geological
datasets.

Reference:
    Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization"
"""

import numpy as np
import tensorflow as tf
from typing import Tuple


def mixup_batch(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply MixUp augmentation to a batch.

    For each pair (x_i, y_i) and (x_j, y_j), produces:
        x_mix = λ * x_i + (1 - λ) * x_j
        y_mix = λ * y_i + (1 - λ) * y_j
    where λ ~ Beta(α, α).

    Args:
        x: Input patches of shape (batch, H, W, C).
        y: One-hot labels of shape (batch, num_classes).
        alpha: MixUp interpolation strength. Higher = more mixing.
               0.2 is generally a good default 

    Returns:
        Tuple of (mixed_x, mixed_y).
    """
    batch_size = len(x)

    # Sample mixing coefficient from Beta distribution
    lam = np.random.beta(alpha, alpha, size=batch_size)
    lam = np.maximum(lam, 1 - lam)  # Ensure dominant sample is x_i

    # Reshape for broadcasting: (batch, 1, 1, 1) for images
    lam_x = lam.reshape(-1, 1, 1, 1)
    lam_y = lam.reshape(-1, 1)

    # Random permutation for pairing
    indices = np.random.permutation(batch_size)

    mixed_x = lam_x * x + (1 - lam_x) * x[indices]
    mixed_y = lam_y * y + (1 - lam_y) * y[indices]

    return mixed_x.astype(np.float32), mixed_y.astype(np.float32)


def create_augmented_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    batch_size: int = 64,
    mixup_alpha: float = 0.2,
    use_mixup: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset with optional MixUp augmentation.

    Applies geometric augmentations (rotation, flip) that preserve
    geological validity, followed by optional MixUp.

    Args:
        x_train: Training patches, shape (N, H, W, C).
        y_train: Integer labels, shape (N,).
        num_classes: Total number of classes.
        batch_size: Batch size for training.
        mixup_alpha: MixUp interpolation strength.
        use_mixup: Whether to apply MixUp augmentation.
        seed: Random seed.

    Returns:
        tf.data.Dataset yielding (x_batch, y_batch) tuples.
    """
    # Convert to one-hot for MixUp compatibility
    y_onehot = tf.keras.utils.to_categorical(y_train, num_classes)

    # Build dataset pipeline
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_onehot))
    dataset = dataset.shuffle(len(x_train), seed=seed)
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # 1. Apply geometric augmentation using pure TF image ops (Keras 3 safe & avoids interpolation artifacts)
    def augment_geometric(x, y):
        x = tf.image.random_flip_left_right(x, seed=seed)
        x = tf.image.random_flip_up_down(x, seed=seed)
        # Discrete 90-degree rotations avoid spectral interpolation artifacts
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32, seed=seed)
        x = tf.image.rot90(x, k=k)
        return x, y

    dataset = dataset.map(augment_geometric, num_parallel_calls=tf.data.AUTOTUNE)

    # 2. Apply mixup using tf.py_function (since it uses numpy)
    if use_mixup:
        def do_mixup(x_batch, y_batch):
            x_mixed, y_mixed = mixup_batch(x_batch.numpy(), y_batch.numpy(), alpha=mixup_alpha)
            return tf.constant(x_mixed), tf.constant(y_mixed)

        dataset = dataset.map(
            lambda x, y: tf.py_function(
                do_mixup, [x, y],
                [tf.float32, tf.float32],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        
        # tf.py_function loses shape information, which crashes Keras 3. Restore it:
        input_shape = [None, x_train.shape[1], x_train.shape[2], x_train.shape[3]]
        def set_shapes(x, y):
            x.set_shape(input_shape)
            y.set_shape([None, num_classes])
            return x, y
            
        dataset = dataset.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
