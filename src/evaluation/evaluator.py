import torch
import pandas as pd
from tqdm import tqdm
from .metrics import calculate_metrics

def evaluate_and_save(model, test_loader, device, model_name, save_csv_path="results/metrics/model_comparison.csv"):
    """Đánh giá mô hình trên tập Test và lưu kết quả vào CSV."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Đang đánh giá Test"):
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    class_names = ["0 Không có", "1 Nhẹ", "2 Trung bình", "3 Nặng", "4 Tăng sinh"]
    report, df_fp_fn = calculate_metrics(all_labels, all_preds, class_names)
    
    # Tính Accuracy tổng thể
    acc = (pd.Series(all_labels) == pd.Series(all_preds)).mean()
    
    # Tạo dòng kết quả mới
    new_row = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "F1_Macro": round(report["macro avg"]["f1-score"], 4),
        "Sensitivity_Class4": df_fp_fn.iloc[4]["True Positive"] / (df_fp_fn.iloc[4]["True Positive"] + df_fp_fn.iloc[4]["False Negative"]),
        "Specificity_Class4": df_fp_fn.iloc[4]["Specificity"]
    }])
    
    # Ghi nối vào file CSV
    import os
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    if os.path.exists(save_csv_path):
        new_row.to_csv(save_csv_path, mode='a', header=False, index=False)
    else:
        new_row.to_csv(save_csv_path, mode='w', header=True, index=False)

    return report, df_fp_fn
