"""Metrics and error analysis."""

from src.evaluation.error_analysis import (
    build_error_dataframe,
    fp_fn_table,
    plot_fp_fn_chart,
    save_error_gallery,
    summarize_confusions,
)
from src.evaluation.metrics import (
    compute_metrics,
    metrics_to_row,
    plot_confusion_matrix,
    save_metrics_csv,
    specificity_per_class,
)

__all__ = [
    "compute_metrics",
    "metrics_to_row",
    "plot_confusion_matrix",
    "save_metrics_csv",
    "specificity_per_class",
    "build_error_dataframe",
    "fp_fn_table",
    "plot_fp_fn_chart",
    "save_error_gallery",
    "summarize_confusions",
]
