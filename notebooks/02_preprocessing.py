# ==================== MINH HỌA QUÁ TRÌNH TIỀN XỬ LÝ ====================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

ten_anh_minh_hoa = df.sample(1, random_state=1).iloc[0]["image_id"] + ".jpg"
duong_dan_goc = None

# SỬA LỖI: Quét cả 3 tập (train, val, test) và 2 cấu trúc thư mục
for split in ["train", "val", "test"]:
    path_1 = f"{BASE}/{split}/images/{ten_anh_minh_hoa}"
    path_2 = f"{BASE}/{split}/{ten_anh_minh_hoa}"
    
    if os.path.exists(path_1):
        duong_dan_goc = path_1
        break
    elif os.path.exists(path_2):
        duong_dan_goc = path_2
        break

if duong_dan_goc is None:
    raise FileNotFoundError(f"Không tìm thấy ảnh mẫu {ten_anh_minh_hoa} trong bất kỳ tập nào!")

anh_goc = cv2.imread(duong_dan_goc)
if anh_goc is None:
    raise ValueError(f"File ảnh tồn tại nhưng không thể đọc được: {duong_dan_goc}")

anh_goc_rgb = cv2.cvtColor(anh_goc, cv2.COLOR_BGR2RGB)

# SỬA LỖI: Thêm bọc lỗi (if img is None) ngay trong hàm
def crop_black_border(img, tol=10):
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img[y0:y1, x0:x1]
    return img

def apply_clahe(img):
    if img is None: return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# Thực hiện pipeline
anh_roi = crop_black_border(anh_goc)
anh_roi_rgb = cv2.cvtColor(anh_roi, cv2.COLOR_BGR2RGB)

anh_clahe = apply_clahe(anh_roi)
anh_clahe_rgb = cv2.cvtColor(anh_clahe, cv2.COLOR_BGR2RGB)

anh_resize = cv2.resize(anh_clahe, (384, 384), interpolation=cv2.INTER_AREA)
anh_resize_rgb = cv2.cvtColor(anh_resize, cv2.COLOR_BGR2RGB)

# Vẽ biểu đồ 4 bước
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes[0].imshow(anh_goc_rgb); axes[0].set_title("1. Ảnh gốc", fontsize=13, fontweight="bold"); axes[0].axis("off")
axes[1].imshow(anh_roi_rgb); axes[1].set_title("2. Cắt viền đen (ROI)", fontsize=13, fontweight="bold"); axes[1].axis("off")
axes[2].imshow(anh_clahe_rgb); axes[2].set_title("3. Sau CLAHE", fontsize=13, fontweight="bold"); axes[2].axis("off")
axes[3].imshow(anh_resize_rgb); axes[3].set_title("4. Resize 384x384", fontsize=13, fontweight="bold"); axes[3].axis("off")

plt.suptitle(f"Quá trình tiền xử lý ảnh minh họa: {ten_anh_minh_hoa}", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("/kaggle/working/truoc_sau_tien_xu_ly.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Đã lưu ảnh minh họa quá trình xử lý thành công.")
# ==================== TIỀN XỬ LÝ HÀNG LOẠT ====================
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading

# Tự động tìm BASE (phòng hờ chạy độc lập)
import glob
train_csv_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
valid_paths = [p for p in train_csv_paths if "dataset_split" in p]
BASE = os.path.dirname(os.path.dirname(valid_paths[0]))

OUT = "/kaggle/working/processed"
SIZE = 384
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUT}/{split}", exist_ok=True)

failed_images = []
# SỬA LỖI 1: Tạo ổ khóa (Lock) để chống Race Condition khi đa luồng ghi dữ liệu
lock = threading.Lock()

def crop_black_border(img, tol=10):
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img[y0:y1, x0:x1]
    return img

def apply_clahe(img):
    if img is None: return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def xu_ly_mot_anh(args):
    img_path, save_path = args
    try:
        img = cv2.imread(img_path)
        if img is None:
            with lock: # Khóa an toàn
                failed_images.append((img_path, "imread=None"))
            return False
        img = crop_black_border(img)
        if img is None or img.size == 0:
            with lock:
                failed_images.append((img_path, "crop_failed"))
            return False
        img = apply_clahe(img)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(save_path, img):
            with lock:
                failed_images.append((img_path, "imwrite_failed"))
            return False
        return True
    except Exception as e:
        with lock:
            failed_images.append((img_path, f"{type(e).__name__}: {e}"))
        return False

def process_split(split_name):
    csv_path = f"{BASE}/{split_name}/{split_name}.csv"
    
    # SỬA LỖI 2: Tự động dò thư mục ảnh (có hoặc không có thư mục con /images/)
    img_dir_1 = f"{BASE}/{split_name}/images"
    img_dir_2 = f"{BASE}/{split_name}"
    img_dir = img_dir_1 if os.path.exists(img_dir_1) else img_dir_2
    
    df_split = pd.read_csv(csv_path)
    
    tasks = []
    for _, row in df_split.iterrows():
        img_name = f"{row['image_id']}.jpg"
        src = os.path.join(img_dir, img_name)
        dst = os.path.join(OUT, split_name, img_name)
        if not os.path.exists(dst):
            tasks.append((src, dst))
    
    print(f"Đang xử lý {split_name}: {len(tasks)} ảnh...")
    if len(tasks) > 0:
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(tqdm(executor.map(xu_ly_mot_anh, tasks), total=len(tasks)))
        print(f"  -> Thành công: {sum(results)}/{len(tasks)}")
    else:
        print("  -> Tất cả ảnh đã được xử lý từ trước.")

print("Bắt đầu tiền xử lý ảnh...")
process_split("train")
process_split("val")
process_split("test")

print("\nSố ảnh đã xử lý:")
for split in ["train", "val", "test"]:
    print(f"  {split}: {len(os.listdir(f'{OUT}/{split}'))} ảnh")

if failed_images:
    with open("/kaggle/working/failed_images.txt", "w", encoding="utf-8") as f:
        for p, reason in failed_images:
            f.write(f"{p}\t{reason}\n")
    print(f"\nCó {len(failed_images)} ảnh lỗi -> xem failed_images.txt")
else:
    print("\nKhông có ảnh lỗi nào!")
# ==================== KIỂM TRA ẢNH SAU TIỀN XỬ LÝ ====================
from PIL import Image
print("=" * 60)
print("KIỂM TRA ẢNH SAU TIỀN XỬ LÝ")
print("=" * 60)
for split in ["train", "val", "test"]:
    folder = f"{OUT}/{split}"
    if not os.path.exists(folder): continue
    files = os.listdir(folder)
    print(f"\n=== {split.upper()} ({len(files)} ảnh) ===")
    for f in files[:3]:
        path = os.path.join(folder, f)
        img = Image.open(path)
        size_kb = os.path.getsize(path) / 1024
        print(f"  {f:<25} | Kích thước: {img.size} | Dung lượng: {size_kb:.1f} KB | Mode: {img.mode}")
