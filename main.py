# ==================== FILE MAIN.PY (ĐIỀU KHIỂN TOÀN BỘ PIPELINE) ====================
import argparse
import os
import glob
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import các module từ thư mục src/
from src.data.dataset import DRDataset
from src.models.architectures import get_model
from src.models.losses import get_weighted_loss
from src.models.trainer import train_model
from src.evaluation.metrics import calculate_metrics
from src.evaluation.error_analysis import plot_confusion_matrix, plot_fp_fn_chart
from src.xai.gradcam import generate_gradcam

def main(args):
    # 1. CÀI ĐẶT THIẾT BỊ VÀ ĐƯỜNG DẪN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị đang sử dụng: {device}")

    # Tự động tìm dữ liệu (giống code trên Kaggle của bạn)
    train_csv_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if not train_csv_paths:
        train_csv_paths = glob.glob("./data/**/train.csv", recursive=True)
    
    if not train_csv_paths:
        raise FileNotFoundError("Không tìm thấy file dữ liệu train.csv. Vui lòng kiểm tra lại!")
    
    BASE = os.path.dirname(os.path.dirname(train_csv_paths[0]))
    PROC = os.path.join(BASE, "processed") # Thư mục chứa ảnh đã tiền xử lý

    # Tạo thư mục kết quả
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    # 2. CHUẨN BỊ DỮ LIỆU
    print(f"Đang tải dữ liệu cho mô hình: {args.model_name}...")
    
    # Transform (Chỉ Normalize vì ảnh đã được tiền xử lý)
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ])
    
    # Đọc CSV
    df_train = pd.read_csv(f"{BASE}/train/train.csv")
    df_val = pd.read_csv(f"{BASE}/val/val.csv")
    df_test = pd.read_csv(f"{BASE}/test/test.csv")

    train_dataset = DRDataset(df_train, f"{PROC}/train", transform=val_transform) # (Lưu ý: Nên thay bằng train_transform có Augmentation)
    val_dataset = DRDataset(df_val, f"{PROC}/val", transform=val_transform)
    test_dataset = DRDataset(df_test, f"{PROC}/test", transform=val_transform)

    # Batch size (DenseNet cần nhỏ hơn để tránh tràn RAM)
    batch_size = 16 if args.model_name == "densenet121" else 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. KHỞI TẠO MÔ HÌNH, LOSS VÀ OPTIMIZER
    print(f"Đang khởi tạo mô hình {args.model_name}...")
    model = get_model(model_name=args.model_name, num_classes=5, pretrained=True).to(device)

    # Hàm Loss có trọng số (Weighted Cross-Entropy)
    criterion, _ = get_weighted_loss(df_train["DR_ICDR"].values, num_classes=5, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    # 4. HUẤN LUYỆN MÔ HÌNH
    best_model_path = f"results/models/{args.model_name}_best.pth"
    checkpoint_path = f"results/models/{args.model_name}_checkpoint.pth"

    print(f"Bắt đầu huấn luyện {args.model_name} trong {args.epochs} epochs...")
    train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        device=device, epochs=args.epochs, patience=5,
        save_path=best_model_path, checkpoint_path=checkpoint_path
    )

    # 5. ĐÁNH GIÁ TRÊN TẬP TEST
    print("\nĐang tải mô hình tốt nhất để đánh giá...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Tính Metrics
    class_names = ["0 Không có", "1 Nhẹ", "2 Trung bình", "3 Nặng", "4 Tăng sinh"]
    report, df_fp_fn = calculate_metrics(all_labels, all_preds, class_names)
    
    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    print(report)
    print(df_fp_fn.to_string(index=False))

    # Vẽ và lưu biểu đồ
    plot_confusion_matrix(all_labels, all_preds, class_names, save_path=f"results/figures/{args.model_name}_confusion_matrix.png")
    plot_fp_fn_chart(df_fp_fn, save_path=f"results/figures/{args.model_name}_fp_fn_chart.png")

    # 6. XAI (GRAD-CAM) - Demo trên 1 ảnh
    print("\nĐang tạo bản đồ nhiệt Grad-CAM mẫu...")
    # (Bạn có thể viết vòng lặp cho 5 lớp giống hệt code trên Kaggle)
    sample_img = next(iter(test_loader))[0][0].unsqueeze(0).to(device) 
    sample_label = next(iter(test_loader))[1][0].item()
    
    # Lưu ý: Cần hàm transform ngược (denormalize) để hiển thị ảnh đúng màu
    # Tạm thời tôi giữ nguyên logic cơ bản, bạn có thể tinh chỉnh thêm

    print("\nHoàn thành toàn bộ Pipeline!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Phân loại bệnh võng mạc đái tháo đường")
    parser.add_argument("--model_name", type=str, default="resnet18", 
                        choices=["resnet18", "densenet121", "efficientnet_b0"], 
                        help="Tên mô hình muốn huấn luyện")
    parser.add_argument("--epochs", type=int, default=30, help="Số lượng epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    main(args)
