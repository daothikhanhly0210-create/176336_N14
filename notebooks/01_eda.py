# ==================== THIẾT LẬP TỰ ĐỘNG & EDA PHÂN BỐ NHÃN ====================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

# 1. TỰ ĐỘNG TÌM ĐƯỜNG DẪN CHỨA FILE TRAIN.CSV
print("Đang tự động tìm kiếm dữ liệu...")
train_csv_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
valid_paths = [p for p in train_csv_paths if "dataset_split" in p]

if not valid_paths:
    raise FileNotFoundError("Không tìm thấy dữ liệu! Hãy chắc chắn bạn đã bấm 'Add Input' và thêm dataset.")

BASE = os.path.dirname(os.path.dirname(valid_paths[0]))
print(f"-> Đã tìm thấy đường dẫn dữ liệu: {BASE}\n")

# 2. ĐỌC DỮ LIỆU
df_train = pd.read_csv(f"{BASE}/train/train.csv")
df_val   = pd.read_csv(f"{BASE}/val/val.csv")
df_test  = pd.read_csv(f"{BASE}/test/test.csv")
df = pd.concat([df_train, df_val, df_test], ignore_index=True)

print("=" * 60)
print("KÍCH THƯỚC TẬP DỮ LIỆU - BRSET")
print("=" * 60)
print(f"Train : {len(df_train):>6,} ảnh  |  {df_train['patient_id'].nunique():>5,} bệnh nhân")
print(f"Val   : {len(df_val):>6,} ảnh  |  {df_val['patient_id'].nunique():>5,} bệnh nhân")
print(f"Test  : {len(df_test):>6,} ảnh  |  {df_test['patient_id'].nunique():>5,} bệnh nhân")

p_tr = set(df_train.patient_id.astype(str))
p_va = set(df_val.patient_id.astype(str))
p_te = set(df_test.patient_id.astype(str))
print(f"\nTrùng Train ↔ Val  : {len(p_tr & p_va)}  {'OK' if len(p_tr & p_va) == 0 else 'LEAK!'}")
print(f"Trùng Train ↔ Test : {len(p_tr & p_te)}  {'OK' if len(p_tr & p_te) == 0 else 'LEAK!'}")
print(f"Trùng Val   ↔ Test : {len(p_va & p_te)}  {'OK' if len(p_va & p_te) == 0 else 'LEAK!'}")

count_all = df["DR_ICDR"].value_counts().sort_index()
ten_nhan = {0: "Không có", 1: "Nhẹ", 2: "Trung bình", 3: "Nặng", 4: "Tăng sinh"}
print("\n" + "=" * 60)
print("PHÂN BỐ MỨC ĐỘ DR_ICDR")
print("=" * 60)
for k in count_all.index:
    print(f"  Nhãn {k} ({ten_nhan.get(k, '?'):<10}): {count_all[k]:>6,} ảnh")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# SỬA LỖI: ax=axes[0] thay vì ax=axes
sns.countplot(data=df, x="DR_ICDR", hue="DR_ICDR", palette="viridis", legend=False, ax=axes[0])
axes[0].set_title("Phân bố tổng mức độ DR_ICDR", fontweight="bold")

dist_table = pd.DataFrame({
    "Train": df_train["DR_ICDR"].value_counts(normalize=True).sort_index() * 100,
    "Val":   df_val["DR_ICDR"].value_counts(normalize=True).sort_index() * 100,
    "Test":  df_test["DR_ICDR"].value_counts(normalize=True).sort_index() * 100,
}).round(2)

# SỬA LỖI: ax=axes[1]
dist_table.T.plot(kind="bar", ax=axes[1], colormap="viridis", width=0.75)
axes[1].set_title("Tỷ lệ % từng nhãn theo Tập", fontweight="bold")

plt.tight_layout()
plt.savefig("/kaggle/working/phan_bo_nhan.png", dpi=150, bbox_inches="tight")
plt.show()
# ==================== KIỂM TRA SỐ MẮT VÀ ẢNH TRÙNG LẶP ====================
print("\n" + "=" * 60)
print("KIỂM TRA SỐ MẮT VÀ ẢNH TRÙNG LẶP")
print("=" * 60)

so_anh_moi_bn = df.groupby("patient_id").size()
so_mat_moi_bn = df.groupby("patient_id")["exam_eye"].nunique(dropna=True)

print(f"Bệnh nhân chỉ có 1 mắt : {(so_mat_moi_bn == 1).sum():,}")
print(f"Bệnh nhân có đủ 2 mắt  : {(so_mat_moi_bn == 2).sum():,}")

anh_cua_tung_mat = df.groupby(["patient_id", "exam_eye"]).size()
trung_lap = anh_cua_tung_mat[anh_cua_tung_mat > 1]
if len(trung_lap) == 0:
    print("-> KHÔNG CÓ hiện tượng trùng lặp ảnh trên cùng 1 mắt.")
else:
    print(f"-> CẢNH BÁO: Có {len(trung_lap)} trường hợp một mắt bị chụp nhiều hơn 1 ảnh!")

pivot_pct = pd.crosstab(df["exam_eye"], df["DR_ICDR"], normalize="index") * 100
print("\nBảng tỷ lệ % nhãn DR theo Mắt (1=Trái, 2=Phải):")
print(pivot_pct.round(2))
# ==================== HIỂN THỊ ẢNH MẪU 5 MỨC ĐỘ ====================
import cv2
import os
import matplotlib.pyplot as plt

ten_muc_do = {0: "Nhãn 0: Không có", 1: "Nhãn 1: Nhẹ", 2: "Nhãn 2: Trung bình", 
              3: "Nhãn 3: Nặng", 4: "Nhãn 4: Tăng sinh"}

anh_mau = {}
for muc in range(5):
    df_muc = df[df["DR_ICDR"] == muc]
    if len(df_muc) > 0:
        row = df_muc.sample(1, random_state=42).iloc[0]
        # Kiểm tra cả 2 cấu trúc thư mục phổ biến để chắc chắn tìm thấy ảnh
        for split in ["train", "val", "test"]:
            path_1 = f"{BASE}/{split}/images/{row['image_id']}.jpg"
            path_2 = f"{BASE}/{split}/{row['image_id']}.jpg"
            
            if os.path.exists(path_1):
                anh_mau[muc] = path_1
                break
            elif os.path.exists(path_2):
                anh_mau[muc] = path_2
                break

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, muc in enumerate(range(5)):
    ax = axes[i]
    # KIỂM TRA AN TOÀN: Đảm bảo khóa 'muc' tồn tại và file thực sự có trên đĩa
    if muc in anh_mau and os.path.exists(anh_mau[muc]):
        img = cv2.imread(anh_mau[muc])
        if img is not None: # Tránh lỗi nếu file hỏng
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(ten_muc_do[muc], fontsize=13, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "Lỗi đọc file ảnh", ha='center', va='center', color='red')
    else:
        ax.text(0.5, 0.5, "Không tìm thấy ảnh", ha='center', va='center', color='red')
    ax.axis("off")

axes[5].axis("off")
plt.suptitle("Ví dụ ảnh đáy mắt theo từng mức độ (BRSET)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("/kaggle/working/anh_mau_theo_muc_do.png", dpi=150, bbox_inches="tight")
plt.show()
# ==================== BIỂU ĐỒ CƯỜNG ĐỘ ĐIỂM ẢNH (TRƯỚC/SAU CLAHE) ====================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

dong = df.sample(1, random_state=42).iloc[0]
ten_anh = dong["image_id"] + ".jpg"
duong_dan = None

# SỬA LỖI 1: Tìm kiếm với cấu trúc thư mục dự phòng
for tap in ["train", "val", "test"]:
    path_1 = f"{BASE}/{tap}/images/{ten_anh}"
    path_2 = f"{BASE}/{tap}/{ten_anh}"
    
    if os.path.exists(path_1):
        duong_dan = path_1
        break
    elif os.path.exists(path_2):
        duong_dan = path_2
        break

# Chặn lỗi crash nếu không tìm thấy ảnh
if duong_dan is None:
    raise FileNotFoundError(f"Không tìm thấy ảnh {ten_anh} trong bất kỳ tập nào!")

anh_goc = cv2.imread(duong_dan)
if anh_goc is None:
    raise ValueError(f"File ảnh tồn tại nhưng không thể đọc được: {duong_dan}")

anh_xam = cv2.cvtColor(anh_goc, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
anh_clahe = clahe.apply(anh_xam)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# SỬA LỖI 2: Mở rộng dải histogram ra toàn bộ 0-255
axes[0].hist(anh_xam.ravel(), bins=256, range=[0, 255], color="#4a4a4a", alpha=0.9)
axes[0].set_title("TRƯỚC KHI ÁP DỤNG CLAHE\nPhân bố lệch về tối", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Cường độ điểm ảnh (0 - 255)")
axes[0].set_ylabel("Số lượng điểm ảnh")

axes[1].hist(anh_clahe.ravel(), bins=256, range=[0, 255], color="#1f77b4", alpha=0.9)
axes[1].set_title("SAU KHI ÁP DỤNG CLAHE\nPhân bố đều hơn", fontsize=14, fontweight="bold", color="#1f77b4")
axes[1].set_xlabel("Cường độ điểm ảnh (0 - 255)")
axes[1].set_ylabel("Số lượng điểm ảnh")

plt.suptitle(f"So sánh phân bố cường độ điểm ảnh: {ten_anh}", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("/kaggle/working/bieu_do_clahe.png", dpi=200, bbox_inches="tight")
plt.show()
print(f"Đã lưu biểu đồ và xử lý thành công ảnh: {ten_anh}")
