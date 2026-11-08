import numpy as np
import cv2
from src.preprocessing.roi_cropper import crop_black_border
from src.preprocessing.enhancement import apply_clahe

def test_crop_black_border():
    """Kiểm tra hàm cắt viền đen có trả về ảnh nhỏ hơn ảnh gốc không."""
    # Tạo ảnh giả (100x100) với viền đen và vùng trắng ở giữa
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img[20:80, 20:80] = 255 # Vùng trắng
    
    cropped_img = crop_black_border(dummy_img)
    
    # Kích thước phải nhỏ hơn 100x100
    assert cropped_img.shape[0] < 100
    assert cropped_img.shape[1] < 100

def test_apply_clahe():
    """Kiểm tra hàm CLAHE giữ nguyên kích thước và số kênh màu."""
    dummy_img = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    
    enhanced_img = apply_clahe(dummy_img)
    
    # Hình dạng (shape) phải giữ nguyên (384, 384, 3)
    assert enhanced_img.shape == (384, 384, 3)
