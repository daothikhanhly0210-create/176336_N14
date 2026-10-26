# Phân loại mức độ bệnh võng mạc đái tháo đường (BRSET)

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Dataset-BRSET-2E8B57" alt="BRSET">
  <img src="https://img.shields.io/badge/Task-5--class%20DR%20grading-blue" alt="Task">
  <img src="https://img.shields.io/badge/XAI-Grad--CAM-orange" alt="XAI">
</p>

Đề tài phân loại **5 mức độ bệnh võng mạc đái tháo đường** (Diabetic Retinopathy) trên bộ dữ liệu **BRSET** (16.266 ảnh fundus, 8.524 bệnh nhân), so sánh ba kiến trúc **ResNet18 – DenseNet121 – EfficientNet-B0**, kèm **Grad-CAM** để giải thích quyết định của mô hình.

---

## 1. Mục tiêu

| # | Mục tiêu | Cách thực hiện |
|---|----------|----------------|
| 1 | Tiền xử lý ảnh y khoa đúng chuẩn | Cắt viền đen (ROI) + CLAHE trên không gian LAB + mặt nạ tròn + Resize 384×384 |
| 2 | Chống rò rỉ dữ liệu | Chia tập **theo bệnh nhân** (patient-level), có test tự động |
| 3 | Xử lý mất cân bằng lớp | Weighted Cross-Entropy (trọng số `sqrt(inverse frequency)`) |
| 4 | So sánh kiến trúc | ResNet18 / DenseNet121 / EfficientNet-B0, cùng recipe huấn luyện |
| 5 | Đánh giá toàn diện | Accuracy, F1-macro, F1-weighted, **QWK**, Sensitivity, Specificity, Confusion Matrix |
| 6 | Giải thích mô hình | Grad-CAM + sanity check, phân tích FP/FN theo từng mức độ |

## 2. Kết quả

> Bảng dưới đây được điền tự động từ `results/metrics/model_comparison.csv` sau khi chạy `notebooks/03_model_training_colab.ipynb`.

| Model | Params | Accuracy | F1-macro | Sensitivity | Specificity |
|-------|-------:|---------:|---------:|------------:|------------:|
| ResNet18 | – | – | – | – | – |
| DenseNet121 | – | – | – | – | – |
| EfficientNet-B0 | – | – | – | – | – |

Hình ảnh minh hoạ nằm trong `results/figures/`:

- `confusion_matrix_test.png` — ma trận nhầm lẫn trên tập test
- `fp_fn_chart.png` — biểu đồ False Positive / False Negative theo mức độ
- `gradcam_sanity_check.png` — ảnh gốc | heatmap | overlay

## 3. Cấu trúc dự án

```text
diabetic_retinopathy_brset/
├── configs/
│   ├── data_config.yaml              # Đường dẫn, 384x384, tham số ROI/CLAHE/mask
│   ├── model_config.yaml             # AdamW, LR=1e-4, Batch=32, Patience=5
│   └── xai_config.yaml               # Target layer Grad-CAM cho từng model
├── data/
│   ├── raw/                          # Dữ liệu BRSET gốc (không commit)
│   ├── processed/                    # Ảnh sau ROI + CLAHE + Resize (không commit)
│   └── splits/                       # train.csv / val.csv / test.csv (không commit)
├── docs/
│   └── BRSET_DATASET_DOCUMENTATION.md
├── notebooks/
│   ├── 01_eda_visualization.ipynb    # Phân bố nhãn, kiểm tra 2 mắt, biểu đồ
│   ├── 02_preprocessing_pipeline.ipynb # Test hàm ROI/CLAHE, xử lý hàng loạt
│   └── 03_model_training_colab.ipynb # Train, Test, Grad-CAM
├── results/
│   ├── figures/                      # Hình cho slide/báo cáo
│   ├── metrics/                      # CSV kết quả F1/Accuracy/QWK
│   └── models/                       # Trọng số .pth (không commit)
├── src/
│   ├── data/                         # DRDataset, patient-level split
│   ├── preprocessing/                # roi_cropper, enhancement, circular_mask
│   ├── models/                       # architectures, losses, trainer
│   ├── evaluation/                   # metrics, error_analysis
│   ├── xai/                          # gradcam
│   └── utils/                        # config, logger
├── tests/                            # pytest: data leakage + preprocessing
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 4. Pipeline

```text
BRSET raw (.jpg, ảnh gốc + CSV nhãn)
        │
        ▼
[1] EDA            → phân bố nhãn, số mắt/bệnh nhân, kích thước ảnh
        │
        ▼
[2] Split          → patient-level 70/15/15  (chống leakage)
        │
        ▼
[3] Preprocess     → crop viền đen → CLAHE (LAB) → mặt nạ tròn → 384×384
        │
        ▼
[4] Train          → AdamW 1e-4, batch 32, weighted CE, AMP, early stopping (patience 5)
        │
        ▼
[5] Evaluate       → F1-macro, QWK, Sensitivity, Specificity, Confusion Matrix, FP/FN
        │
        ▼
[6] Explain        → Grad-CAM + sanity check
```

## 5. Cài đặt

```bash
git clone https://github.com/daothikhanhly0210-create/176336_N14.git
cd 176336_N14

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> **GPU:** cài PyTorch bản CUDA trước theo hướng dẫn tại <https://pytorch.org/get-started/locally/>,
> sau đó mới chạy `pip install -r requirements.txt`.
> **Colab:** chọn Runtime → Change runtime type → T4 GPU, rồi `!git clone ...` và `!pip install -q -r requirements.txt`.

## 6. Dữ liệu

1. Tải BRSET (xem hướng dẫn trong [`docs/BRSET_DATASET_DOCUMENTATION.md`](docs/BRSET_DATASET_DOCUMENTATION.md)).
2. Giải nén vào `data/raw/` — **không commit ảnh** (đã có `.gitignore`).
3. Tạo file CSV split (`data/splits/train.csv`, `val.csv`, `test.csv`) bằng `notebooks/01_eda_visualization.ipynb` hoặc:

```python
import pandas as pd
from src.data import make_stratified_patient_splits

df = pd.read_csv("data/raw/labels.csv")   # cần cột: image_path, patient_id, label
splits = make_stratified_patient_splits(df, out_dir="data/splits", val_size=0.15, test_size=0.15)
```

Mỗi file CSV cần tối thiểu 3 cột:

| Cột | Ý nghĩa |
|-----|---------|
| `image_path` | Đường dẫn ảnh (tuyệt đối hoặc tương đối so với `data/processed`, `data/raw`) |
| `patient_id` | Mã bệnh nhân — dùng để chia tập, chống leakage |
| `label` | Nhãn 0–4 tương ứng No DR → Proliferative |

## 7. Cách chạy

| Notebook | Nội dung |
|----------|----------|
| `01_eda_visualization.ipynb` | Phân bố nhãn, kiểm tra 2 mắt/bệnh nhân, mẫu ảnh, thống kê kích thước, chia tập |
| `02_preprocessing_pipeline.ipynb` | Test trực quan ROI/CLAHE/mask trên 1 ảnh, xử lý hàng loạt ra `data/processed/` |
| `03_model_training_colab.ipynb` | Train 3 model, đánh giá trên test, vẽ confusion matrix, FP/FN, Grad-CAM |

Huấn luyện bằng code (không cần notebook):

```python
from src.utils.config import load_project_configs
from src.data import build_dataloaders
from src.models import build_model, build_criterion, build_optimizer, build_scheduler, Trainer, compute_class_weight

configs = load_project_configs()
configs["model"]["training"].update({"epochs": 30, "batch_size": 32})

loaders = build_dataloaders(
    "data/splits/train.csv", "data/splits/val.csv",
    config={**configs["data"], **configs["model"]},
)

model = build_model("resnet18", num_classes=5, pretrained=True, dropout=0.2)
weights = compute_class_weight(loaders["train"].dataset.targets, num_classes=5, mode="sqrt")
criterion = build_criterion(configs["model"]["training"], class_weights=weights)
optimizer = build_optimizer(model, configs["model"]["training"])
scheduler = build_scheduler(optimizer, configs["model"]["training"], epochs=30)

trainer = Trainer(
    model, loaders["train"], loaders["val"],
    criterion=criterion, optimizer=optimizer, scheduler=scheduler,
    epochs=30, patience=5,
    checkpoint_path="results/models/resnet18_best.pth",
    history_path="results/metrics/train_history_resnet18.csv",
)
history = trainer.fit()
```

## 8. Cấu hình

| File | Nội dung chính |
|------|----------------|
| `configs/data_config.yaml` | `image.size: 384`, đường dẫn, ROI (`threshold: 7`), CLAHE (`clip_limit: 2.0`, `tile: 8×8`), mask (`radius_ratio: 0.5`), tỉ lệ split 15/15 |
| `configs/model_config.yaml` | AdamW, `lr: 1e-4`, `weight_decay: 1e-4`, `batch_size: 32`, `patience: 5`, cosine schedule, AMP, 3 backbone + tên checkpoint |
| `configs/xai_config.yaml` | `gradcam.target_layers`: ResNet18 → `layer4[-1]`, DenseNet121 → `features.denseblock4`, EfficientNet-B0 → `features[-1]` |

Đọc config trong code:

```python
from src.utils.config import load_yaml, load_project_configs

data_cfg = load_yaml("data_config.yaml")          # tự tìm trong configs/
all_cfg  = load_project_configs()                  # {'data': ..., 'model': ..., 'xai': ...}
```

## 9. Đánh giá & XAI

```python
from src.evaluation import compute_metrics, plot_confusion_matrix, fp_fn_table, plot_fp_fn_chart

metrics = compute_metrics(y_true, y_pred, num_classes=5, class_names=configs["data"]["classes"]["names"])
print(metrics["f1_macro"], metrics["qwk"])

plot_confusion_matrix(metrics, class_names=configs["data"]["classes"]["names"],
                      out_path="results/figures/confusion_matrix_test.png")
plot_fp_fn_chart(fp_fn_table(y_true, y_pred, 5), out_path="results/figures/fp_fn_chart.png")
```

```python
from src.xai import build_explainer, overlay_cam, denormalize

explainer = build_explainer(model, model_name="resnet18", xai_config=configs["xai"], device="cuda")
cam = explainer.explain(image_tensor, target_class=pred_class)   # image_tensor: [1, 3, 384, 384]
overlay = overlay_cam(denormalize(image_tensor[0]), cam, alpha=0.4)
```

## 10. Kiểm thử

```bash
pytest -q
```

- `tests/test_data_leakage.py` — đảm bảo không bệnh nhân nào xuất hiện ở 2 tập, split tái lập được, phát hiện leakage.
- `tests/test_preprocessing.py` — đảm bảo ảnh sau xử lý đúng `384×384×3`, mask xoá góc, CLAHE giữ shape/dtype.

## 11. Lưu ý

- **Không** commit ảnh bệnh nhân, file CSV split hay trọng số `.pth` — `.gitignore` đã chặn.
- Kết quả chỉ mang tính nghiên cứu/đồ án, **không dùng để chẩn đoán**.
- Toàn bộ số liệu trong `docs/BRSET_DATASET_DOCUMENTATION.md` cần đối chiếu lại với bản dataset bạn tải; chỗ chưa chắc chắn được đánh dấu `TODO`.

## 12. Trích dẫn

* Dữ liệu được sử dụng trong đồ án: **Nakayama et al. (2024). BRSET: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos.** [Link Kaggle](https://www.kaggle.com/datasets/tanzinabdul/fundus-patientwise-split)

## 13. Liên hệ

- GitHub: [@daothikhanhly0210-create](https://github.com/daothikhanhly0210-create)
- Repository: [176336_N14](https://github.com/daothikhanhly0210-create/176336_N14)
