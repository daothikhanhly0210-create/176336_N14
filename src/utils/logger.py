import logging
import os

def setup_logger(log_file="training.log"):
    """
    Thiết lập logger để ghi log ra màn hình và file.
    """
    logger = logging.getLogger("BRSET_Project")
    logger.setLevel(logging.INFO)
    
    # Nếu logger đã được cấu hình trước đó thì không làm lại
    if logger.handlers:
        return logger

    # Định dạng log: Thời gian - Mức độ - Nội dung
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 1. Ghi ra màn hình (Console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Ghi ra file (File)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
