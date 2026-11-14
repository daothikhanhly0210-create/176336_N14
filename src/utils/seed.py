import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    """Cố định seed cho toàn bộ quá trình để kết quả có thể tái lập (TRIPOD+AI)."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Đã cố định Seed: {seed}")
