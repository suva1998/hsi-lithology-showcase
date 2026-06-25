"""
Training loop for Baseline CNN.

Handles:
    - Model compilation with focal loss and cosine decay LR
    - Training with MixUp augmentation
    - Checkpoint saving to Google Drive
    - TensorBoard logging
    - Early stopping with patience
    - K-fold cross-validation
"""

import os
import time
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.model_selection import StratifiedKFold

from src.models.baseline_cnn import build_baseline_cnn_from_config
from src.training.losses import class_balanced_focal_loss
from src.training.scheduler import CosineDecayWithWarmup
from src.data.augmentation import create_augmented_dataset


class Trainer:
    """Orchestrates model training, validation, and cross-validation.

    Usage:
        >>> config = yaml.safe_load(open("configs/mla_drillcore.yaml"))
        >>> trainer = Trainer(config)
        >>> history = trainer.train(splits, class_counts)
        >>> results = trainer.evaluate(X_test, y_test)
    """

    def __init__(self, config: Dict):
        """Initialize trainer from config.

        Args:
            config: Loaded YAML configuration dict.
        """
        self.config = config
        self.train_cfg = config["training"]
        self.model_cfg = config["model"]
        self.ds_cfg = config["dataset"]
        self.paths = config.get("paths", {})

        # Set random seeds for reproducibility
        seed = self.train_cfg.get("seed", 42)
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def _build_and_compile(
        self,
        class_counts: Dict[int, int],
    ) -> tf.keras.Model:
        """Build model and compile with focal loss + cosine LR.

        Args:
            class_counts: Per-class sample counts for loss weighting.

        Returns:
            Compiled Keras model.
        """
        # Build architecture
        model = build_baseline_cnn_from_config(self.config)

        # Loss function
        loss_name = self.train_cfg.get("loss", "focal")
        if loss_name == "focal":
            loss_fn = class_balanced_focal_loss(
                class_counts=class_counts,
                gamma=self.train_cfg.get("focal_gamma", 2.0),
                num_classes=self.ds_cfg.get("num_classes", 21),
            )
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy()

        # Learning rate schedule
        lr = self.train_cfg.get("learning_rate", 1e-3)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Compile
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy"],
        )

        return model

    def _get_callbacks(
        self,
        fold: Optional[int] = None,
    ) -> list:
        """Create training callbacks (checkpointing, early stopping, etc.).

        Args:
            fold: Cross-validation fold number (for naming).

        Returns:
            List of Keras callbacks.
        """
        callbacks = []
        suffix = f"_fold{fold}" if fold is not None else ""

        # Checkpoint saving (to Google Drive for persistence)
        ckpt_dir = self.paths.get(
            "checkpoint_dir",
            "checkpoints",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"best_model{suffix}.keras")

        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )

        # Early stopping
        if self.train_cfg.get("early_stopping", True):
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=self.train_cfg.get("monitor", "val_loss"),
                    patience=self.train_cfg.get("patience", 20),
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        # TensorBoard logging
        log_dir = self.paths.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(log_dir, f"run{suffix}"),
                histogram_freq=0,
            )
        )

        return callbacks

    def train(
        self,
        splits: Dict[str, tuple],
        class_counts: Dict[int, int],
    ) -> Dict[str, Any]:
        """Train the model on a single train/val/test split.

        Args:
            splits: Dict with "train", "val", "test" keys,
                    each containing (X, y) tuple.
            class_counts: Per-class sample counts.

        Returns:
            Dict with "model", "history", "best_val_loss".
        """
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]

        # Build and compile
        model = self._build_and_compile(class_counts)

        # Create augmented training dataset
        train_dataset = create_augmented_dataset(
            X_train, y_train,
            num_classes=self.ds_cfg["num_classes"],
            batch_size=self.train_cfg.get("batch_size", 64),
            mixup_alpha=self.train_cfg.get("mixup_alpha", 0.2),
            use_mixup=self.train_cfg.get("mixup", True),
            seed=self.train_cfg.get("seed", 42),
        )

        # Validation data (no augmentation)
        y_val_onehot = tf.keras.utils.to_categorical(
            y_val, self.ds_cfg["num_classes"]
        )

        # Callbacks
        callbacks = self._get_callbacks()

        # Train
        print(f"\n🚀 Training CNN...")
        print(f"   Epochs: {self.train_cfg['epochs']}")
        print(f"   Batch size: {self.train_cfg['batch_size']}")
        print(f"   MixUp: {self.train_cfg.get('mixup', True)}")

        start_time = time.time()
        history = model.fit(
            train_dataset,
            validation_data=(X_val, y_val_onehot),
            epochs=self.train_cfg["epochs"],
            callbacks=callbacks,
            verbose=1,
        )
        elapsed = time.time() - start_time

        print(f"\n✅ Training complete in {elapsed:.1f}s")
        print(f"   Best val loss: {min(history.history['val_loss']):.4f}")
        print(f"   Best val acc:  {max(history.history['val_accuracy']):.4f}")

        return {
            "model": model,
            "history": history.history,
            "training_time": elapsed,
            "best_val_loss": min(history.history["val_loss"]),
        }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_counts: Dict[int, int],
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """Stratified K-fold cross-validation.

        Args:
            X: All patches, shape (N, H, W, C).
            y: All labels, shape (N,).
            class_counts: Per-class sample counts.
            n_folds: Number of CV folds.

        Returns:
            Dict with per-fold results and aggregate statistics.
        """
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=self.train_cfg.get("seed", 42),
        )

        fold_results = []
        print(f"\n{'='*60}")
        print(f"  Stratified {n_folds}-Fold Cross-Validation")
        print(f"{'='*60}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'─'*40}")
            print(f"  Fold {fold + 1}/{n_folds}")
            print(f"{'─'*40}")

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Build fresh model for each fold
            model = self._build_and_compile(class_counts)

            # Create training dataset with augmentation
            train_dataset = create_augmented_dataset(
                X_train, y_train,
                num_classes=self.ds_cfg["num_classes"],
                batch_size=self.train_cfg.get("batch_size", 64),
                mixup_alpha=self.train_cfg.get("mixup_alpha", 0.2),
                use_mixup=self.train_cfg.get("mixup", True),
            )

            y_val_onehot = tf.keras.utils.to_categorical(
                y_val, self.ds_cfg["num_classes"]
            )

            callbacks = self._get_callbacks(fold=fold)

            history = model.fit(
                train_dataset,
                validation_data=(X_val, y_val_onehot),
                epochs=self.train_cfg["epochs"],
                callbacks=callbacks,
                verbose=0,
            )

            best_val_loss = min(history.history["val_loss"])
            best_val_acc = max(history.history["val_accuracy"])

            fold_results.append({
                "fold": fold,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "history": history.history,
            })

            print(f"  Fold {fold+1}: val_loss={best_val_loss:.4f}, "
                  f"val_acc={best_val_acc:.4f}")

        # Aggregate statistics
        val_accs = [r["best_val_acc"] for r in fold_results]
        val_losses = [r["best_val_loss"] for r in fold_results]

        print(f"\n{'='*60}")
        print(f"  Cross-Validation Summary")
        print(f"  Val Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
        print(f"  Val Loss:     {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"{'='*60}")

        return {
            "fold_results": fold_results,
            "mean_val_acc": np.mean(val_accs),
            "std_val_acc": np.std(val_accs),
            "mean_val_loss": np.mean(val_losses),
            "std_val_loss": np.std(val_losses),
        }
