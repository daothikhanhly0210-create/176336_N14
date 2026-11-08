"""
Module Đánh giá mô hình (Evaluation).
Bao gồm: Tính metrics, phân tích lỗi, và vẽ biểu đồ.
"""

from .metrics import calculate_metrics
from .error_analysis import plot_confusion_matrix, plot_fp_fn_chart
