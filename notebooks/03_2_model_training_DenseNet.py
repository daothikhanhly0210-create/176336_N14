# ==================== HUAN LUYEN DENSENET-121 ====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import cv2
import numpy as np
import random
import os
import glob
from tqdm import tqdm

cv2.setNumThreads(0)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiet bi dang dung:", device)

train_csv_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
valid_paths = [p for p in train_csv_paths if "dataset_split" in p]
BASE = os.path.dirname(os.path.dirname(valid_paths[0]))

PROC = "/kaggle/working/processed"
CHECKPOINT_PATH = "/kaggle/working/checkpoint_densenet.pth"
BEST_MODEL_PATH = "/kaggle/working/mo_hinh_densenet_tot_nhat.pth"

# LƯU Ý: 
# - Nếu đây là lần chạy ĐẦU TIÊN (cần xóa checkpoint cũ 12 epochs), hãy XÓA dấu # ở 3 dòng dưới.
# - Nếu đang chạy tiếp (resume), hãy ĐỂ NGUYÊN dấu # để không bị xóa mất checkpoint.
# if os.path.exists(CHECKPOINT_PATH):
#     os.remove(CHECKPOINT_PATH)
#     print("Da xoa checkpoint cu de bat dau train lai tu dau voi T_max=30.")

df_train = pd.read_csv(f"{BASE}/train/train.csv")
df_val   = pd.read_csv(f"{BASE}/val/val.csv")

class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df; self.img_dir = img_dir; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['image_id']}.jpg"
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"Khong doc duoc anh: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(row["DR_ICDR"])
        if self.transform: img = self.transform(image=img)["image"]
        return img, label

train_transform = A.Compose([
    A.Resize(384, 384), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(384, 384), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
])

train_dataset = DRDataset(df_train, f"{PROC}/train", train_transform)
val_dataset   = DRDataset(df_val,   f"{PROC}/val",   val_transform)

# GIẢM BATCH SIZE XUỐNG 16 CHO DENSENET ĐỂ TRÁNH TRÀN RAM (OOM)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

y_train = df_train["DR_ICDR"].values
weights = compute_class_weight(class_weight="balanced", classes=np.array([0,1,2,3,4]), y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
model.classifier = nn.Linear(model.classifier.in_features, 5)
model = model.to(device)

EPOCHS = 30
# SỬA LỖI: Hạ Learning Rate xuống 5e-5 để thích ứng với Batch Size = 16
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler('cuda')

start_epoch, best_acc, patience, patience_counter = 1, 0.0, 5, 0

if os.path.exists(CHECKPOINT_PATH):
    print(f"\nTim thay checkpoint tai {CHECKPOINT_PATH}. Dang khoi phuc...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scheduler.T_max = EPOCHS 
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Dùng .get() để phòng hờ lỗi KeyError
    start_epoch = checkpoint.get('epoch', 1) + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    patience_counter = checkpoint.get('patience_counter', 0)
    print(f"Khoi phuc thanh cong! Tiep tuc train tu Epoch {start_epoch}.\n")
else:
    print("\nKhong tim thay checkpoint. Bat dau train tu dau.\n")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Eval"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

for epoch in range(start_epoch, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
    vl, va = evaluate(model, val_loader, criterion)
    
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    
    print(f"LR: {current_lr:.6f} | Train Loss: {tl:.4f} | Train Acc: {ta:.4f} | Val Loss: {vl:.4f} | Val Acc: {va:.4f}")

    if va > best_acc:
        best_acc = va; patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f" -> Da luu mo hinh tot nhat (Val Acc: {best_acc:.4f})")
    else:
        patience_counter += 1
        print(f" -> Khong cai thien (Patience: {patience_counter}/{patience})")

    checkpoint = {
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(), 'best_acc': best_acc, 'patience_counter': patience_counter
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    if patience_counter >= patience:
        print(f"\nEarly stopping tai epoch {epoch}")
        break

print(f"\nVal Accuracy tot nhat: {best_acc:.4f}")
# ==================== DANH GIA DENSENET-121 ====================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121
import pandas as pd
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

cv2.setNumThreads(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_csv_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
valid_paths = [p for p in train_csv_paths if "dataset_split" in p]
BASE = os.path.dirname(os.path.dirname(valid_paths[0]))

PROC = "/kaggle/working/processed"
MODEL_PATH = "/kaggle/working/mo_hinh_densenet_tot_nhat.pth" 

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Khong tim thay file mo hinh DenseNet! Hay chac chan Cell 8 da chay xong.")

class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): 
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['image_id']}.jpg"
        img = cv2.imread(img_path)
        if img is None: 
            raise FileNotFoundError(f"Khong doc duoc anh: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(row["DR_ICDR"])
        if self.transform: 
            img = self.transform(image=img)["image"]
        return img, label

val_transform = A.Compose([
    A.Resize(384, 384), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2()
])

df_test = pd.read_csv(f"{BASE}/test/test.csv")
test_dataset = DRDataset(df_test, f"{PROC}/test", val_transform)
# Giữ batch_size=16 cho DenseNet khi test
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

model = densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Dang du doan Test"):
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n" + "=" * 60 + "\nBAO CAO PHAN LOAI DENSENET-121 (TEST SET)\n" + "=" * 60)
print(classification_report(all_labels, all_preds, target_names=["0 Khong co", "1 Nhe", "2 Trung binh", "3 Nang", "4 Tang sinh"], zero_division=0, digits=4))

mcm = multilabel_confusion_matrix(all_labels, all_preds)
class_names = ["0 Khong co", "1 Nhe", "2 Trung binh", "3 Nang", "4 Tang sinh"]

data_fp_fn = []
print("Specificity tung lop:")
for i in range(5):
    tn, fp, fn, tp = mcm[i].ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  Lop {i}: {spec:.4f}")
    data_fp_fn.append({"Muc do benh": class_names[i], "True Positive": tp, "False Positive (FP)": fp, "False Negative (FN)": fn, "True Negative": tn, "Specificity": f"{spec:.4f}"})

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1", "2", "3", "4"], yticklabels=["0", "1", "2", "3", "4"])
plt.title("Confusion Matrix DenseNet-121 - Test Set", fontweight="bold")
plt.xlabel("Du doan (Predicted)")
plt.ylabel("Thuc te (True)")
plt.tight_layout()
plt.savefig("/kaggle/working/confusion_matrix_densenet.png", dpi=150)
plt.show()
plt.close('all')

df_fp_fn = pd.DataFrame(data_fp_fn)
print("\n" + "=" * 90 + "\nBANG PHAN TICH TONG HOP: SPECIFICITY, FALSE POSITIVE (FP) VA FALSE NEGATIVE (FN)\n" + "=" * 90)
print(df_fp_fn.to_string(index=False))

df_fp_fn.plot(x="Muc do benh", y=["False Positive (FP)", "False Negative (FN)"], 
              kind="bar", figsize=(10, 5), color=["#ff9999", "#66b3ff"])
plt.title("So sanh so ca Bao dong gia (FP) va Bo sot benh (FN) - DenseNet-121", fontweight="bold")
plt.ylabel("So luong ca")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("/kaggle/working/fp_fn_chart_densenet.png", dpi=150)
plt.show()
plt.close('all')
# ==================== GRAD-CAM DENSENET-121 ====================
import os
try:
    import pytorch_grad_cam
except ImportError:
    os.system('pip install grad-cam -q')

import torch
import torch.nn as nn
from torchvision.models import densenet121
import pandas as pd
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget 

cv2.setNumThreads(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_csv_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
valid_paths = [p for p in train_csv_paths if "dataset_split" in p]
BASE = os.path.dirname(os.path.dirname(valid_paths[0]))

PROC = "/kaggle/working/processed"
MODEL_PATH = "/kaggle/working/mo_hinh_densenet_tot_nhat.pth"

model = densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Target layer cua DenseNet-121 la features[-1]
target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# SỬA LỖI LỆCH TỌA ĐỘ: Chỉ Normalize vì ảnh đã được resize bằng OpenCV
val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2()
])

df_test = pd.read_csv(f"{BASE}/test/test.csv")
fig, axes = plt.subplots(5, 2, figsize=(10, 20))

for ax in axes.flatten(): ax.axis("off")

for i in range(5):
    df_muc = df_test[df_test["DR_ICDR"] == i]
    if len(df_muc) == 0: continue
    row = df_muc.sample(1, random_state=42).iloc[0]
    
    img_path = f"{PROC}/test/{row['image_id']}.jpg"
    img_bgr = cv2.imread(img_path)
    
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_resized = cv2.resize(img_rgb, (384, 384))
        img_float = img_rgb_resized.astype(np.float32) / 255.0
        
        # Truyền img_rgb_resized vào transform
        input_tensor = val_transform(image=img_rgb_resized)["image"].unsqueeze(0).to(device)
        
        targets = [ClassifierOutputTarget(i)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        
        visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        
        axes[i, 0].imshow(img_rgb_resized)
        axes[i, 0].set_title(f"Anh goc - Lop {i}", fontweight="bold")
        
        axes[i, 1].imshow(visualization)
        axes[i, 1].set_title(f"Grad-CAM - Lop {i}", fontweight="bold")

plt.suptitle("Sanity Check DenseNet-121: Grad-CAM", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("/kaggle/working/gradcam_densenet.png", dpi=150)
plt.show()
plt.close('all')
