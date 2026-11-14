import torch.optim as optim

def build_scheduler(optimizer, epochs=30):
    """Khởi tạo Scheduler (mặc định: CosineAnnealingLR)"""
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
