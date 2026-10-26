# Tài liệu bộ dữ liệu BRSET

> Tài liệu nội bộ của đồ án, tóm tắt những gì cần biết trước khi huấn luyện.
> Các con số có gắn `TODO` cần đối chiếu lại với phiên bản BRSET bạn tải về
> (bộ dữ liệu có nhiều bản phát hành, số ảnh có thể chênh lệch).

---

## 1. Tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Tên đầy đủ | BRSET — Brazilian Multilabel Ophthalmological Dataset |
| Loại ảnh | Ảnh đáy mắt (retinal fundus), chụp màu |
| Số ảnh | **16.266** |
| Số bệnh nhân | **8.524** |
| Số mắt / bệnh nhân | 2 (trái + phải), một số bệnh nhân chụp nhiều lần |
| Số nhãn trong đồ án | **5** (mức độ bệnh võng mạc đái tháo đường) |
| Ngôn ngữ nhãn gốc | Bệnh án tiếng Bồ Đào Nha (đã ánh xạ sang 0–4) |
| Định dạng ảnh | `.jpg` / `.png` (kích thước gốc thay đổi theo máy chụp) |

> `TODO`: xác nhận lại tổng số ảnh/bệnh nhân và danh sách thiết bị chụp
> (Canon / Nikon / Topcon…) từ file README đi kèm bản dataset bạn tải.

## 2. Định nghĩa 5 nhãn (phân độ ICDR)

| Nhãn | Tên | Mô tả lâm sàng | Đặc điểm ảnh cần chú ý |
|:---:|---|---|---|
| 0 | No DR | Không có dấu hiệu bệnh võng mạc đái tháo đường | Đáy mắt bình thường |
| 1 | Mild | Bệnh võng mạc không tăng sinh nhẹ | Vi phình mạch (microaneurysm) |
| 2 | Moderate | Không tăng sinh trung bình | Xuất huyết, bông gòn, bất thường tĩnh mạch |
| 3 | Severe | Không tăng sinh nặng | Nhiều xuất huyết, IRMA, bất thường vi mạch rõ |
| 4 | Proliferative | Tăng sinh | Tân mạch, xuất huyết dịch kính, màng xơ |

Thứ tự nhãn trong code được khai báo duy nhất tại `configs/data_config.yaml → classes.names`;
mọi module (`metrics`, `error_analysis`, `gradcam`) đều đọc từ đó, tránh lệch thứ tự.

> `TODO`: nếu bản BRSET bạn dùng cung cấp **nhãn đa nhãn** (multi-label: DR, macular edema,
> glaucoma suspect…) thì cần chọn cột DR rồi ánh xạ về 5 mức ở trên **trước khi** chia tập.
> Ghi lại quy tắc ánh xạ vào mục 6 của tài liệu này.

## 3. Phân bố lớp

Bảng dưới được điền tự động ở Cell 2 của `notebooks/01_eda_visualization.ipynb`
(`df["label"].value_counts()`), dùng để quyết định `class_weight`.

| Nhãn | Số ảnh | Tỉ lệ | Ghi chú |
|:---:|---:|---:|---|
| 0 — No DR | TODO | TODO | Lớp đa số |
| 1 — Mild | TODO | TODO | |
| 2 — Moderate | TODO | TODO | |
| 3 — Severe | TODO | TODO | Lớp hiếm |
| 4 — Proliferative | TODO | TODO | Lớp hiếm nhất |
| **Tổng** | 16.266 | 100% | |

Hệ quả: dùng `weighted Cross-Entropy` với trọng số `sqrt(N / (K · n_c))`
(xem `src/models/losses.py::compute_class_weight`), và báo cáo **F1-macro**
thay vì chỉ accuracy — accuracy sẽ bị chi phối bởi lớp No DR.

## 4. Cấu trúc dữ liệu đầu vào

```text
data/raw/
├── images/                  # 16.266 ảnh fundus
│   ├── 0001_left.jpg
│   └── ...
└── labels.csv               # metadata gốc (tên cột có thể khác, cần map lại)

data/splits/                 # do notebook 01 sinh ra
├── train.csv                # 70% bệnh nhân
├── val.csv                  # 15% bệnh nhân
└── test.csv                 # 15% bệnh nhân
```

Mỗi CSV split có tối thiểu:

| Cột | Kiểu | Ghi chú |
|---|---|---|
| `image_path` | str | Tuyệt đối, hoặc tương đối so với `data/processed` / `data/raw` |
| `patient_id` | str | **Bắt buộc** — dùng cho patient-level split |
| `label` | int 0–4 | Nhãn phân độ |

## 5. Quy tắc chia tập (chống data leakage)

Vì mỗi bệnh nhân có 2 mắt và có thể chụp nhiều lần, chia ngẫu nhiên theo **ảnh** sẽ khiến
mắt trái nằm ở train còn mắt phải ở test → mô hình "học thuộc" bệnh nhân, điểm test bị thổi phồng.

Quy tắc của dự án:

1. Gom nhóm theo `patient_id`; mỗi bệnh nhân chỉ thuộc **một** tập.
2. Stratify theo nhãn **đa số của bệnh nhân** để giữ tỉ lệ 5 lớp giữa các tập.
3. Tỉ lệ mặc định **70/15/15** (`val_size=0.15`, `test_size=0.15`), seed `42` để tái lập.
4. `assert_no_leakage()` chạy sau khi chia và trong `tests/test_data_leakage.py`.
5. Báo cáo trong luận văn phải nêu rõ số **bệnh nhân** mỗi tập, không chỉ số ảnh.

## 6. Tiền xử lý

| Bước | Module | Lý do |
|---|---|---|
| Cắt viền đen (ROI) | `src/preprocessing/roi_cropper.py` | Ảnh gốc có khung đen lớn; sau resize 384×384 vùng võng mạc chỉ còn ~50% diện tích → mất chi tiết tổn thương nhỏ |
| CLAHE trên kênh L (LAB) | `src/preprocessing/enhancement.py` | Làm rõ vi phình mạch/xuất huyết ở vùng tối, giữ màu tự nhiên của đáy mắt |
| Mặt nạ tròn | `src/preprocessing/circular_mask.py` | Xoá 4 góc đen — CNN có thể học artefact góc ảnh thay vì tổn thương |
| Resize 384×384 | `src/preprocessing/__init__.py` | Kích thước cố định cho batch; 384 giữ được chi tiết nhỏ tốt hơn 224 |
| Normalize ImageNet | `src/data/loader.py` | Tương thích pretrained weights |

Tham số nằm trong `configs/data_config.yaml → preprocessing`.

## 7. Huấn luyện

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Loss | Weighted Cross-Entropy (`sqrt` inverse frequency) |
| AMP | Bật khi có CUDA |
| Early stopping | Patience 5, monitor `val_f1_macro` |
| Checkpoint | `results/models/{resnet18,densenet121,efficientnet_b0}_best.pth` |

## 8. Đánh giá

- **Accuracy**, **F1-macro**, **F1-weighted**
- **QWK** (Quadratic Weighted Kappa) — phù hợp bài toán phân độ có thứ tự
- **Sensitivity / Specificity** từng lớp (one-vs-rest)
- **Confusion matrix** + phân tích **FP/FN** và hướng sai (under-graded / over-graded)
- **Grad-CAM** cho các ca sai để kiểm tra mô hình có nhìn đúng tổn thương

## 9. Tải dữ liệu

`TODO`: điền link chính thức bạn dùng (Kaggle / Zenodo / PhysioNet / trang nhóm nghiên cứu).
Gợi ý quy trình:

```bash
# 1) tải file nén (ví dụ từ Kaggle CLI)
kaggle datasets download -d <owner>/<brset-dataset> -p data/raw

# 2) giải nén
unzip data/raw/<brset>.zip -d data/raw

# 3) kiểm tra số ảnh
python -c "from pathlib import Path; print(len(list(Path('data/raw').rglob('*.jpg'))))"
```

Sau khi tải, cập nhật lại tên cột trong `labels.csv` cho khớp 3 cột chuẩn ở mục 4.

## 10. Trích dẫn

```bibtex
@article{brset2023,
  title   = {BRSET: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos},
  author  = {Nakayama, Luis Filipe and others},
  year    = {2023},
  note    = {TODO: bổ sung venue/DOI chính xác}
}
```

## 11. Lưu ý pháp lý & đạo đức

- Ảnh fundus là **dữ liệu y tế**: không commit lên GitHub, không chia sẻ công khai ngoài phạm vi dataset cho phép.
- Kết quả của mô hình **không thay thế** chẩn đoán của bác sĩ.
- Khi báo cáo, nêu rõ giới hạn: dữ liệu một quốc gia, một số loại máy chụp, nhãn lấy từ bệnh án.
