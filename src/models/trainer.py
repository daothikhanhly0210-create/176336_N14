import torch
import os
from tqdm import tqdm

def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, scaler,
    device, epochs, patience, save_path, checkpoint_path
):
    """
    Vòng lặp huấn luyện chính với Early Stopping và lưu Checkpoint.
    """
    start_epoch, best_acc, patience_counter = 1, 0.0, 0

    # Kiểm tra xem có checkpoint cũ không để chạy tiếp (Resume)
    if os.path.exists(checkpoint_path):
        print(f"Tìm thấy checkpoint tại {checkpoint_path}. Đang khôi phục...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Phòng hờ lỗi KeyError
        start_epoch = checkpoint.get('epoch', 1) + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Khôi phục thành công! Tiếp tục từ Epoch {start_epoch}.\n")
    else:
        print("Không tìm thấy checkpoint. Bắt đầu train từ đầu.\n")

    for epoch in range(start_epoch, epochs + 1):
        # --- 1. TRAIN ---
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
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
            
        train_loss, train_acc = total_loss / total, correct / total

        # --- 2. EVALUATE ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Eval]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss, val_acc = val_loss / val_total, val_correct / val_total
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # --- 3. LƯU TRỌNG SỐ TỐT NHẤT ---
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f" -> Đã lưu mô hình tốt nhất (Val Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            print(f" -> Không cải thiện (Patience: {patience_counter}/{patience})")

        # --- 4. LƯU CHECKPOINT ---
        checkpoint = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(), 'best_acc': best_acc, 'patience_counter': patience_counter
        }
        torch.save(checkpoint, checkpoint_path)

        # --- 5. EARLY STOPPING ---
        if patience_counter >= patience:
            print(f"\nEarly Stopping tại Epoch {epoch}. Dừng huấn luyện.")
            break

    print(f"\nHoàn tất! Val Accuracy tốt nhất: {best_acc:.4f}")
    return best_acc
