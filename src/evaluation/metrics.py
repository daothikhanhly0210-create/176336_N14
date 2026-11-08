from sklearn.metrics import classification_report, multilabel_confusion_matrix
import pandas as pd

def calculate_metrics(y_true, y_pred, class_names):
    """
    Tính toán báo cáo phân loại và bảng phân tích lỗi (FP, FN, Specificity).
    """
    # 1. Báo cáo phân loại
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        zero_division=0, 
        digits=4, 
        output_dict=True
    )
    
    # 2. Tính Specificity, FP, FN bằng Multilabel Confusion Matrix
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    data = []
    for i in range(len(class_names)):
        tn, fp, fn, tp = mcm[i].ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        data.append({
            "Muc do benh": class_names[i],
            "True Positive": tp,
            "False Positive (FP)": fp,
            "False Negative (FN)": fn,
            "True Negative": tn,
            "Specificity": round(spec, 4)
        })
        
    df_fp_fn = pd.DataFrame(data)
    return report, df_fp_fn
