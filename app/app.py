import streamlit as st
import torch
import cv2
import numpy as np
import sys
import os
import pandas as pd

# Lấy đường dẫn gốc của dự án để tránh lỗi Path khi chạy Streamlit
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.models.architectures import get_model
from src.preprocessing.roi_cropper import crop_black_border
from src.preprocessing.enhancement import apply_clahe
from src.xai.gradcam import generate_gradcam
from src.preprocessing.transforms import get_inference_transforms

st.set_page_config(page_title="Chẩn đoán Bệnh Võng Mạc", layout="wide")
st.title("Hệ thống Phân loại Bệnh Võng Mạc Đái Tháo Đường")
st.write("Tải lên ảnh đáy mắt để nhận kết quả chẩn đoán và bản đồ Grad-CAM giải thích.")

st.sidebar.header("Cấu hình")
model_name = st.sidebar.selectbox("Chọn mô hình:", ["resnet18", "densenet121", "efficientnet_b0"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["0 - Không bệnh", "1 - Nhẹ", "2 - Trung bình", "3 - Nặng", "4 - Tăng sinh"]

@st.cache_resource
def load_model(model_name):
    model = get_model(model_name, num_classes=5, pretrained=False)
    model_path = os.path.join(BASE_DIR, "results", "models", f"{model_name}_best.pth")
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Xử lý tự động nếu file lưu dạng Dict hoặc DataParallel
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            model.load_state_dict(state_dict)
            st.sidebar.success(f"Đã tải trọng số {model_name}")
        except Exception as e:
            st.sidebar.error(f"Lỗi tải trọng số: {e}")
    else:
        st.sidebar.warning(f"Không tìm thấy file {model_path}.")
        
    model.to(device)
    model.eval()
    return model

uploaded_file = st.file_uploader("Chọn ảnh đáy mắt (JPG/PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        # ĐÃ SỬA LỖI DEPRECATION
        st.image(img_rgb_display, caption="Ảnh gốc", use_container_width=True)

    # Tiền xử lý
    img_roi = crop_black_border(img_bgr)
    img_clahe = apply_clahe(img_roi)
    img_resized_bgr = cv2.resize(img_clahe, (384, 384))
    
    # SỬA LỖI KÊNH MÀU: Chuyển sang RGB cho mô hình
    img_resized_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)
    
    # Dự đoán
    model = load_model(model_name)
    transform = get_inference_transforms()
    input_tensor = transform(image=img_resized_rgb)["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    with col2:
        st.subheader("Kết quả chẩn đoán:")
        st.success(f"**{class_names[pred_class]}** (Độ tự tin: {probs[pred_class]*100:.2f}%)")
        
        chart_data = pd.DataFrame({"Xác suất (%)": probs.cpu().numpy() * 100}, index=class_names)
        st.bar_chart(chart_data)

    st.subheader("Giải thích AI (Grad-CAM)")
    
    img_float_rgb = img_resized_rgb.astype(np.float32) / 255.0
    
    # Sinh Grad-CAM
    cam_image = generate_gradcam(
        model=model, 
        model_name=model_name, 
        input_tensor=input_tensor, 
        rgb_image=img_float_rgb, 
        true_label=pred_class, 
        device=device
    )
    # ĐÃ SỬA LỖI DEPRECATION
    st.image(cam_image, caption="Bản đồ nhiệt Grad-CAM", use_container_width=True)
