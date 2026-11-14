import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=384):
    """
    Trả về pipeline biến đổi ảnh cho tập Train (Bao gồm Data Augmentation).
    Giúp chống Overfitting, đặc biệt quan trọng với các lớp bệnh hiếm (như Nhãn 3).
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(
            scale=(0.9, 1.1), 
            translate_percent=0.05, 
            rotate=15, 
            p=0.5
        ),
        A.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            p=0.3
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

def get_val_test_transforms(image_size=384):
    """
    Trả về pipeline biến đổi ảnh cho tập Validation và Test.
    Tuyệt đối KHÔNG dùng Augmentation ở đây để đảm bảo tính khách quan.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])

def get_inference_transforms():
    """
    Dùng riêng cho Grad-CAM (vì ảnh thường đã được resize sẵn từ trước).
    Chỉ cần Normalize.
    """
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
