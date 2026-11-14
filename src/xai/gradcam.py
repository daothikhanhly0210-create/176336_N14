import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def get_target_layers(model, model_name):
    """
    Tự động chọn đúng lớp target layer cuối cùng cho từng kiến trúc mô hình.
    """
    if model_name == "resnet18":
        return [model.layer4[-1]]
    elif model_name == "densenet121":
        return [model.features[-1]]
    elif model_name == "efficientnet_b0":
        return [model.features[-1]]
    else:
        raise ValueError(f"Chưa hỗ trợ Grad-CAM cho mô hình: {model_name}")

def generate_gradcam(model, model_name, input_tensor, rgb_image, true_label, device):
    """
    Sinh bản đồ nhiệt Grad-CAM và chồng lên ảnh gốc.
    """
    # 1. Lấy target layer
    target_layers = get_target_layers(model, model_name)
    
    # 2. Khởi tạo Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 3. Ép mô hình giải thích cho nhãn thực tế (true_label)
    targets = [ClassifierOutputTarget(true_label)]
    
    # 4. Sinh bản đồ nhiệt (grayscale)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    
    # 5. Đè bản đồ nhiệt lên ảnh gốc
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    
        # Đảm bảo ảnh trả về luôn là hệ màu RGB (Bọc thép chống lỗi màu)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB) 
    return visualization
