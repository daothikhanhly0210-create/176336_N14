import pandas as pd
from torch.utils.data import DataLoader
from .dataset import DRDataset

def get_dataloaders(train_csv, val_csv, test_csv, img_dir, train_tf, val_tf, batch_size=32):
    """Hàm tạo DataLoader cho cả 3 tập."""
    train_ds = DRDataset(pd.read_csv(train_csv), f"{img_dir}/train", train_tf)
    val_ds = DRDataset(pd.read_csv(val_csv), f"{img_dir}/val", val_tf)
    test_ds = DRDataset(pd.read_csv(test_csv), f"{img_dir}/test", val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader
