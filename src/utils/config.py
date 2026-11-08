import os
import yaml

def load_yaml_config(config_path):
    """
    Đọc file cấu hình YAML an toàn.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_default_config():
    """
    Trả về cấu hình mặc định (phòng hờ nếu không dùng YAML).
    """
    return {
        "data": {
            "img_size": 384,
            "num_classes": 5,
            "batch_size": 32
        },
        "training": {
            "epochs": 30,
            "lr": 1e-4,
            "patience": 5
        }
    }
