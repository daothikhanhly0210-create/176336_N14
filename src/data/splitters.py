import pandas as pd

def check_patient_leakage(df_train, df_val, df_test):
    p_tr, p_va, p_te = set(df_train.patient_id), set(df_val.patient_id), set(df_test.patient_id)
    assert len(p_tr & p_va) == 0, "Lỗi: Rò rỉ dữ liệu giữa Train và Val!"
    assert len(p_tr & p_te) == 0, "Lỗi: Rò rỉ dữ liệu giữa Train và Test!"
    assert len(p_va & p_te) == 0, "Lỗi: Rò rỉ dữ liệu giữa Val và Test!"
    print("-> Kiểm tra thành công: 0% bệnh nhân trùng lặp giữa 3 tập.")
