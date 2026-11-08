import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Vẽ và lưu ma trận nhầm lẫn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Test Set", fontweight="bold")
    plt.xlabel("Du doan (Predicted)")
    plt.ylabel("Thuc te (True)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close('all') # Giải phóng bộ nhớ

def plot_fp_fn_chart(df_fp_fn, save_path):
    """Vẽ và lưu biểu đồ so sánh FP và FN."""
    df_fp_fn.plot(x="Muc do benh", y=["False Positive (FP)", "False Negative (FN)"], 
                  kind="bar", figsize=(10, 5), color=["#ff9999", "#66b3ff"])
    plt.title("So sanh so ca Bao dong gia (FP) va Bo sot benh (FN)", fontweight="bold")
    plt.ylabel("So luong ca")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close('all') # Giải phóng bộ nhớ
