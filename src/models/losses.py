import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_weighted_loss(labels_array, num_classes=5, device="cuda"):
    """
    Tính toán class weights từ mảng nhãn và trả về hàm CrossEntropyLoss có trọng số.
    """
    # Tính trọng số nghịch đảo tần suất
    weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.arange(num_classes), 
        y=labels_array
    )
    
    # Chuyển thành tensor
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Khởi tạo hàm Loss với trọng số
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    return criterion, class_weights
