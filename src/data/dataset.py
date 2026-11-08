import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df; self.img_dir = img_dir; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(f"{self.img_dir}/{row['image_id']}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(row["DR_ICDR"])
        if self.transform: img = self.transform(image=img)["image"]
        return img, label

def get_transforms():
    train_tf = A.Compose([
        A.Resize(384, 384), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3), A.ColorJitter(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Resize(384, 384), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()
    ])
    return train_tf, val_tf
