import torch
import os

def save_checkpoint(state, filepath):
    """Lưu trạng thái mô hình (epoch, weights, optimizer, ...)"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f" -> Đã lưu checkpoint tại: {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    """Tải trạng thái mô hình từ file"""
    if not os.path.exists(filepath):
        return None
        
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Xử lý lỗi T_max
        if hasattr(scheduler, 'T_max') and 'epoch' in checkpoint:
             scheduler.T_max = checkpoint.get('t_max', scheduler.T_max)
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    return checkpoint
