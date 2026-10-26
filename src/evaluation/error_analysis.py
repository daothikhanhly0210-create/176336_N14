"""Error analysis: False Positives / False Negatives and misgraded images.

Two outputs feed the report:

* a per-class FP/FN table + bar chart (``results/figures/fp_fn_chart.png``);
* a dataframe with every misclassified image, its true/predicted grade and the
  direction of the error (under-graded vs over-graded), which is what clinicians
  care about most.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")

__all__ = [
    "build_error_dataframe",
    "fp_fn_table",
    "plot_fp_fn_chart",
    "save_error_gallery",
    "summarize_confusions",
]


def _class_name(label: int, names: Sequence[str]) -> str:
    return names[int(label)] if 0 <= int(label) < len(names) else f"class_{int(label)}"


def build_error_dataframe(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    image_paths: Sequence[str | Path] | None = None,
    patient_ids: Sequence[Any] | None = None,
    class_names: Sequence[str] | None = None,
    only_errors: bool = True,
) -> pd.DataFrame:
    """Tabulate predictions, keeping ``under-graded`` / ``over-graded`` labels.

    ``under-graded`` (pred < true) is the dangerous direction for screening: the
    model underestimated the severity of the disease.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}.")

    names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)
    n = len(y_true)

    if image_paths is not None and len(image_paths) != n:
        raise ValueError("image_paths length must match the number of predictions.")
    if patient_ids is not None and len(patient_ids) != n:
        raise ValueError("patient_ids length must match the number of predictions.")

    frame = pd.DataFrame(
        {
            "image_path": [str(p) for p in image_paths] if image_paths is not None else [None] * n,
            "patient_id": list(patient_ids) if patient_ids is not None else [None] * n,
            "true_label": y_true,
            "pred_label": y_pred,
        }
    )
    frame["true_name"] = [_class_name(label, names) for label in y_true]
    frame["pred_name"] = [_class_name(label, names) for label in y_pred]
    frame["is_error"] = frame["true_label"] != frame["pred_label"]
    frame["abs_gap"] = (frame["true_label"] - frame["pred_label"]).abs()

    direction = np.where(
        frame["true_label"] == frame["pred_label"],
        "correct",
        np.where(frame["pred_label"] < frame["true_label"], "under-graded", "over-graded"),
    )
    frame["direction"] = direction

    return frame.loc[frame["is_error"]].reset_index(drop=True) if only_errors else frame


def fp_fn_table(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    num_classes: int = 5,
    class_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One-vs-rest False Positive / False Negative counts per class."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)

    rows: Dict[str, Dict[str, float]] = {}
    for label in range(num_classes):
        true_positive = int(np.sum((y_pred == label) & (y_true == label)))
        false_positive = int(np.sum((y_pred == label) & (y_true != label)))
        false_negative = int(np.sum((y_pred != label) & (y_true == label)))
        true_negative = int(np.sum((y_pred != label) & (y_true != label)))
        rows[_class_name(label, names)] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "fp_rate": false_positive / max(1, false_positive + true_negative),
            "fn_rate": false_negative / max(1, false_negative + true_positive),
        }
    return pd.DataFrame(rows).T


def plot_fp_fn_chart(
    table: pd.DataFrame,
    out_path: str | Path | None = "results/figures/fp_fn_chart.png",
    title: str = "False Positive / False Negative theo từng mức độ",
    dpi: int = 200,
):
    """Grouped bar chart of FP/FN counts per DR grade."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(table.index)
    x = np.arange(len(labels))
    width = 0.38

    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.bar(x - width / 2, table["false_positive"], width, label="False Positive", color="#e07a5f")
    axis.bar(x + width / 2, table["false_negative"], width, label="False Negative", color="#3d5a80")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=15, ha="right")
    axis.set_ylabel("Số ảnh")
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return out_path


def summarize_confusions(errors: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Most frequent ``true -> predicted`` pairs, for the discussion section."""
    if errors.empty:
        return pd.DataFrame(columns=["true_name", "pred_name", "count"])
    counts = (
        errors.groupby(["true_name", "pred_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return counts


def save_error_gallery(
    errors: pd.DataFrame,
    out_path: str | Path = "results/figures/error_gallery.png",
    max_items: int = 12,
    columns: int = 4,
    dpi: int = 150,
):
    """Contact sheet of misclassified images (true vs predicted in the title)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    subset = errors.head(max_items)
    if subset.empty:
        raise ValueError("No misclassified image to display.")

    rows = int(np.ceil(len(subset) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.4))
    axes = np.atleast_1d(axes).ravel()

    for axis, (_, row) in zip(axes, subset.iterrows()):
        path = row.get("image_path")
        try:
            with Image.open(path) as handle:
                axis.imshow(np.asarray(handle.convert("RGB")))
        except Exception:
            axis.text(0.5, 0.5, "ảnh không đọc được", ha="center", va="center")
        axis.set_title(
            f"true: {row['true_name']}\npred: {row['pred_name']} ({row['direction']})",
            fontsize=8,
        )
        axis.axis("off")

    for axis in axes[len(subset) :]:
        axis.axis("off")

    figure.tight_layout()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return out_path
