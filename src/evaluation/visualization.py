"""
Visualization utilities for HSI classification results.

Generates publication-quality plots:
    - Confusion matrices (normalized, with class names)
    - Classification maps (spatial prediction overlay)
    - Training history curves (loss, accuracy, learning rate)
    - Per-class F1 bar charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Optional, List, Dict


# ── Publication-quality defaults ────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Plot confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: List of class names for axis labels.
        normalize: If True, normalize rows to show percentages.
        title: Plot title.
        save_path: If provided, save figure to this path.
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        vmin, vmax = 0, 1
    else:
        cm_display = cm
        fmt = "d"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names or range(len(cm)),
        yticklabels=class_names or range(len(cm)),
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    return fig


def plot_classification_map(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Classification Map",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot spatial classification map alongside ground truth.

    Creates a side-by-side comparison of ground truth labels and
    model predictions, overlaid on the spatial extent of the image.

    Args:
        predictions: Predicted label map, shape (H, W).
        ground_truth: Ground-truth label map, shape (H, W).
        class_names: Class name list for the legend.
        title: Overall figure title.
        save_path: If provided, save figure.
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure object.
    """
    num_classes = max(ground_truth.max(), predictions.max()) + 1

    # Create a colormap with distinct colors per class
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    colors[0] = [0, 0, 0, 1]  # Background = black
    cmap = mcolors.ListedColormap(colors)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Ground truth
    axes[0].imshow(ground_truth, cmap=cmap, interpolation="nearest")
    axes[0].set_title("Ground Truth", fontweight="bold")
    axes[0].axis("off")

    # Predictions
    axes[1].imshow(predictions, cmap=cmap, interpolation="nearest")
    axes[1].set_title("Predictions", fontweight="bold")
    axes[1].axis("off")

    # Difference map (errors highlighted in red)
    mask = (ground_truth > 0)  # Only labeled pixels
    error_map = np.zeros_like(ground_truth, dtype=np.float32)
    error_map[mask] = (ground_truth[mask] != predictions[mask]).astype(float)

    axes[2].imshow(error_map, cmap="RdYlGn_r", interpolation="nearest",
                   vmin=0, vmax=1)
    axes[2].set_title("Error Map", fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    return fig


def plot_training_history(
    history: Dict[str, list],
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot training and validation loss/accuracy curves.

    Args:
        history: Keras history.history dict with keys like
                 "loss", "val_loss", "accuracy", "val_accuracy".
        title: Figure title.
        save_path: If provided, save figure.
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["loss"], "b-", label="Train", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=1.5)
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["accuracy"], "b-", label="Train", linewidth=1.5)
    axes[1].plot(epochs, history["val_accuracy"], "r-", label="Val", linewidth=1.5)
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    return fig


def plot_per_class_f1(
    f1_scores: np.ndarray,
    class_names: List[str],
    model_name: str = "Baseline CNN",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot horizontal bar chart of per-class F1 scores.

    Args:
        f1_scores: Per-class F1 scores, shape (num_classes,).
        class_names: Class name list.
        model_name: Model name for the title.
        save_path: If provided, save figure.
        figsize: Figure dimensions.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by F1 score for readability
    sorted_idx = np.argsort(f1_scores)
    sorted_f1 = f1_scores[sorted_idx]
    sorted_names = [class_names[i] for i in sorted_idx]

    # Color bars by performance
    colors = plt.cm.RdYlGn(sorted_f1)

    bars = ax.barh(sorted_names, sorted_f1, color=colors, edgecolor="white",
                   linewidth=0.5)

    # Add value labels
    for bar, f1 in zip(bars, sorted_f1):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{f1:.3f}", va="center", fontsize=9)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 Score")
    ax.set_title(f"{model_name} — Per-Class F1 Scores", fontweight="bold")
    ax.axvline(x=np.mean(f1_scores), color="navy", linestyle="--",
               linewidth=1, label=f"Mean F1: {np.mean(f1_scores):.3f}")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"💾 Saved: {save_path}")

    return fig
