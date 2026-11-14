import torch
from tqdm import tqdm
from ..utils.early_stopping import EarlyStopping
from ..utils.checkpoint import save_checkpoint, load_checkpoint

def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, 
    device, epochs, patience, save_path, checkpoint_path
):
    # Khởi tạo module Early Stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0)
    
    start_epoch, best_acc = 1, 0.0

    # Thử load checkpoint (nếu có)
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device)
    if checkpoint:
        start_epoch = checkpoint.get('epoch', 1) + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Khôi phục thành công! Tiếp tục từ Epoch {start_epoch}.\n")

    for epoch in range(start_epoch, epochs + 1):
        # --- TRAIN ---
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

        # --- EVALUATE ---
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

        # --- LƯU MÔ HÌNH TỐT NHẤT ---
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f" -> Đã lưu mô hình tốt nhất (Val Acc: {best_acc:.4f})")

        # --- LƯU CHECKPOINT ---
        checkpoint_data = {
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(), 
            'best_acc': best_acc, 
            't_max': epochs
        }
        save_checkpoint(checkpoint_data, checkpoint_path)

        # --- KIỂM TRA EARLY STOPPING ---
        # Sử dụng val_loss để kiểm tra (chuẩn hơn dùng acc trong y tế)
        early_stopping(val_loss) 
        if early_stopping.early_stop:
            print(f"\nEarly stopping tại epoch {epoch} do Val Loss không cải thiện.")
            break

    print(f"\nHoàn tất! Val Accuracy tốt nhất: {best_acc:.4f}")
    return best_acc
