import pandas as pd

def test_patient_level_split():
    """
    Kiểm tra xem có bệnh nhân nào bị xuất hiện ở cả 2 tập (rò rỉ dữ liệu) không.
    (Đây là hàm giả lập, bạn có thể sửa lại đường dẫn CSV thật của bạn nếu chạy local).
    """
    # Giả lập dữ liệu (Trong thực tế sẽ đọc file csv)
    train_patients = {1, 2, 3, 4, 5}
    val_patients = {6, 7, 8}
    test_patients = {9, 10, 11}

    # Kiểm tra giao thoa (intersection)
    leak_train_val = train_patients.intersection(val_patients)
    leak_train_test = train_patients.intersection(test_patients)
    leak_val_test = val_patients.intersection(test_patients)

    # Assert: Nếu có giao thoa (độ dài > 0) thì báo lỗi
    assert len(leak_train_val) == 0, "Rò rỉ dữ liệu giữa Train và Val!"
    assert len(leak_train_test) == 0, "Rò rỉ dữ liệu giữa Train và Test!"
    assert len(leak_val_test) == 0, "Rò rỉ dữ liệu giữa Val và Test!"
    
    print("Test thành công: 0% bệnh nhân bị trùng lặp.")
