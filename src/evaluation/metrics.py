"""
Evaluation metrics for HSI classification.

Implements the full classification metrics suite:
    - Overall Accuracy (OA)
    - Average Accuracy (AA) — mean of per-class accuracies
    - Cohen's Kappa (κ) — agreement corrected for chance
    - F1-score (macro, per-class)
    - Confusion matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from typing import Dict, Any, Optional, List


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute full classification metrics suite.

    Args:
        y_true: Ground-truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        class_names: Optional list of class names for reporting.

    Returns:
        Dict containing:
            - overall_accuracy: float
            - average_accuracy: float (mean of per-class accuracies)
            - kappa: float (Cohen's κ)
            - f1_macro: float
            - f1_per_class: np.ndarray
            - precision_macro: float
            - recall_macro: float
            - confusion_matrix: np.ndarray
            - per_class_accuracy: np.ndarray
    """
    # Overall Accuracy
    oa = accuracy_score(y_true, y_pred)

    # Per-class accuracy (Average Accuracy)
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    aa = np.mean(per_class_acc)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # F1, Precision, Recall
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "overall_accuracy": oa,
        "average_accuracy": aa,
        "kappa": kappa,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "precision_macro": precision,
        "recall_macro": recall,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
    }

    return metrics


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """Generate a formatted classification report.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Optional class name list.

    Returns:
        Formatted string report.
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )


def print_metrics_summary(
    metrics: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    model_name: str = "Model",
) -> None:
    """Print a formatted summary of classification metrics.

    Args:
        metrics: Dict from compute_metrics().
        class_names: Optional class name list.
        model_name: Name of the model for the header.
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Overall Accuracy (OA):  {metrics['overall_accuracy']:.4f}")
    print(f"  Average Accuracy (AA):  {metrics['average_accuracy']:.4f}")
    print(f"  Cohen's Kappa (κ):      {metrics['kappa']:.4f}")
    print(f"  F1-score (macro):       {metrics['f1_macro']:.4f}")
    print(f"  Precision (macro):      {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):         {metrics['recall_macro']:.4f}")

    if class_names:
        print(f"\n  Per-Class F1 Scores:")
        for i, (name, f1) in enumerate(
            zip(class_names, metrics["f1_per_class"])
        ):
            acc = metrics["per_class_accuracy"][i]
            print(f"    {name:30s}  F1={f1:.4f}  Acc={acc:.4f}")

    print(f"{'='*60}")
