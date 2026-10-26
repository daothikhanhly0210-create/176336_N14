"""Evaluation metrics for 5-class diabetic retinopathy grading.

The report needs more than accuracy: ordinal errors matter clinically (grade 2
predicted as 1 is not the same mistake as grade 2 predicted as 4), so we also
report Quadratic Weighted Kappa (QWK), per-class Sensitivity (recall) and
Specificity (one-vs-rest), plus the confusion matrix used in the slides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

__all__ = [
    "compute_metrics",
    "specificity_per_class",
    "plot_confusion_matrix",
    "metrics_to_row",
    "save_metrics_csv",
]

DEFAULT_CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")


def specificity_per_class(confusion: np.ndarray) -> np.ndarray:
    """One-vs-rest specificity for every class: ``TN / (TN + FP)``."""
    matrix = np.asarray(confusion, dtype=np.float64)
    total = matrix.sum()
    specificities = np.zeros(matrix.shape[0], dtype=np.float64)
    for index in range(matrix.shape[0]):
        true_positive = matrix[index, index]
        false_negative = matrix[index, :].sum() - true_positive
        false_positive = matrix[:, index].sum() - true_positive
        true_negative = total - true_positive - false_negative - false_positive
        denominator = true_negative + false_positive
        specificities[index] = true_negative / denominator if denominator > 0 else 0.0
    return specificities


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int | None = None,
    class_names: Sequence[str] | None = None,
    labels: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Compute the full metric dictionary used by the trainer and the report."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.size == 0:
        raise ValueError("compute_metrics received empty arrays.")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}.")

    num_classes = int(num_classes or (max(int(y_true.max()), int(y_pred.max())) + 1))
    names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)
    if len(names) < num_classes:
        names += [f"class_{index}" for index in range(len(names), num_classes)]
    label_ids = list(labels) if labels is not None else list(range(num_classes))

    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    confusion = confusion_matrix(y_true, y_pred, labels=label_ids)
    specificity = specificity_per_class(confusion)

    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, labels=label_ids, weights="quadratic"))
    except ValueError:  # pragma: no cover - degenerate single-class batches
        qwk = float("nan")

    per_class: Dict[str, Dict[str, float]] = {}
    for index, label in enumerate(label_ids):
        per_class[names[label] if label < len(names) else f"class_{label}"] = {
            "precision": float(precision[index]),
            "sensitivity": float(recall[index]),
            "specificity": float(specificity[index]),
            "f1": float(f1_values[index]),
            "support": int(support[index]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)
        ),
        "qwk": qwk,
        "sensitivity_macro": float(np.mean(recall)),
        "specificity_macro": float(np.mean(specificity)),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "support": int(len(y_true)),
    }


def metrics_to_row(metrics: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    """Flatten a metric dict into a single CSV row (drops nested structures)."""
    row: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            row[f"{prefix}{key}"] = float(value)
    return row


def save_metrics_csv(
    rows: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Append/overwrite a metrics CSV (``results/metrics/*.csv``)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, Mapping):
        frame = pd.DataFrame([dict(rows)])
    else:
        frame = pd.DataFrame([dict(row) for row in rows])
    frame.to_csv(target, index=False)
    return target


def plot_confusion_matrix(
    confusion: Sequence[Sequence[int]] | Mapping[str, Any],
    class_names: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    title: str = "Confusion matrix - test set",
    normalize: bool = False,
    dpi: int = 200,
    cmap: str = "Blues",
):
    """Render the confusion matrix (and optionally save it as PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = (
        np.asarray(confusion["confusion_matrix"], dtype=float)
        if isinstance(confusion, Mapping)
        else np.asarray(confusion, dtype=float)
    )
    names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)[: matrix.shape[0]]

    values = matrix
    fmt = "d"
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        values = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        fmt = ".2f"

    figure, axis = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        values,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "Tỉ lệ" if normalize else "Số ảnh"},
        ax=axis,
    )
    axis.set_xlabel("Nhãn dự đoán (Predicted)")
    axis.set_ylabel("Nhãn thực tế (True)")
    axis.set_title(title)
    figure.tight_layout()

    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return out_path
