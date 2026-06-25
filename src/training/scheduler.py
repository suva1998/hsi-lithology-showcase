"""
Learning rate schedulers.

Implements cosine decay with linear warmup.

The warmup phase prevents the model from making large, poorly-directed
weight updates early in training when the loss landscape is not yet
understood. After warmup, cosine decay gradually reduces the learning
rate, allowing the model to converge to a better minimum.
"""

import numpy as np
import tensorflow as tf


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay learning rate schedule with linear warmup.

    Schedule:
        - Steps [0, warmup_steps): linear ramp from 0 → initial_lr
        - Steps [warmup_steps, total_steps]: cosine decay to min_lr

    Args:
        initial_lr: Peak learning rate after warmup.
        total_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps.
        min_lr: Minimum learning rate at end of cosine decay.
    """

    def __init__(
        self,
        initial_lr: float = 1e-3,
        total_steps: int = 10000,
        warmup_steps: int = 1000,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Linear warmup
        warmup_lr = self.initial_lr * (step / max(self.warmup_steps, 1))

        # Cosine decay
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        decay_progress = (step - self.warmup_steps) / decay_steps
        decay_progress = tf.clip_by_value(decay_progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1 + tf.cos(np.pi * decay_progress)
        )

        # Use warmup for early steps, cosine decay afterwards
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }
