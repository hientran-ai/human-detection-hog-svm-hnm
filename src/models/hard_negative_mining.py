# Filepath: src/models/hard_negative_mining.py

import os
import cv2
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import *
from src.utils.windowing import sliding_window
from src.features.hog_extractor import HOGFeatureExtractor

# Import hàm load data ban đầu để dùng lại
from src.models.train_baseline import load_data_and_extract_features

def run_hard_negative_mining():
    print("="*60)
    print("BẮT ĐẦU GIAI ĐOẠN 3: HARD NEGATIVE MINING (INRIA VERSION)")
    print("="*60)
    
    # 1. Load Baseline Model & Scaler
    if not os.path.exists(BASE_MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[!] Lỗi: Chưa có Baseline Model. Hãy chạy train_baseline.py trước!")
        return
        
    base_svm = joblib.load(BASE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    extractor = HOGFeatureExtractor(win_size=WINDOW_SIZE)
    
    # 2. Đọc danh sách file Train Negative (Ảnh thuần nền)
    with open(TRAIN_NEG_LST, 'r') as f:
        neg_files = [os.path.basename(line.strip()) for line in f.readlines() if line.strip()]
        
    hard_negative_features = []
    
    print(f"\n[1] ĐANG QUÉT TÌM HARD NEGATIVES TRÊN {len(neg_files)} ẢNH THUẦN NỀN...")
    
    # Duyệt qua các ảnh Negative
    for filename in tqdm(neg_files, desc="Mining Hard Negatives"):
        # Dừng lại nếu đã thu thập đủ số lượng mục tiêu
        if len(hard_negative_features) >= MAX_HARD_NEGATIVES:
            break
            
        img_path = os.path.join(TRAIN_NEG_DIR, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Quét Sliding Window trên ảnh thuần nền
        # Dùng bước nhảy WINDOW_STEP (32) cấu hình trong config.py
        for (x, y, window) in sliding_window(img, step_size=WINDOW_STEP, window_size=WINDOW_SIZE):
            # Tính HOG cho cửa sổ
            feat = extractor.compute(window)
            
            # Chuẩn hóa (scale)
            feat_scaled = scaler.transform([feat])
            
            # Dự đoán
            pred = base_svm.predict(feat_scaled)[0]
            
            # Nếu mô hình đoán ĐÂY LÀ NGƯỜI (pred == 1) -> CHẮC CHẮN LÀ SAI
            if pred == 1:
                hard_negative_features.append(feat)
                if len(hard_negative_features) >= MAX_HARD_NEGATIVES:
                    break # Dừng lại nếu đã đủ
                        
    print(f"\n[+] Đã tìm được {len(hard_negative_features)} Hard Negatives (Báo động giả)!")
    
    if len(hard_negative_features) == 0:
        print("[!] Không tìm thấy Hard Negative nào. Mô hình baseline đã quá hoàn hảo.")
        return

    # 3. Gộp Hard Negatives vào tập huấn luyện ban đầu
    print("\n[2] ĐANG LOAD LẠI TẬP DỮ LIỆU BAN ĐẦU (EASY POS/NEG)...")
    X_train_base, y_train_base = load_data_and_extract_features()
    
    print("\n[3] GỘP DỮ LIỆU VÀ CHUẨN HÓA LẠI...")
    X_hnm = np.array(hard_negative_features)
    y_hnm = np.zeros(len(X_hnm)) # Nhãn của Hard Negative là 0 (Nền)
    
    # Gộp 2 mảng
    X_train_new = np.vstack((X_train_base, X_hnm))
    y_train_new = np.concatenate((y_train_base, y_hnm))
    
    print(f"[+] Kích thước tập Train MỚI: {X_train_new.shape} (Đã thêm {len(X_hnm)} mẫu khó)")
    
    # Chuẩn hóa lại vì phân phối dữ liệu đã thay đổi
    new_scaler = StandardScaler()
    X_train_new_scaled = new_scaler.fit_transform(X_train_new)
    
    # Cập nhật file Scaler
    joblib.dump(new_scaler, SCALER_PATH)
    
    # 4. Huấn luyện lại mô hình
    print("\n[4] ĐANG HUẤN LUYỆN LẠI LINEAR SVM (Sẽ lâu hơn lần 1)...")
    hnm_svm_model = LinearSVC(C=1.0, max_iter=3000, class_weight='balanced', random_state=42)
    hnm_svm_model.fit(X_train_new_scaled, y_train_new)
    
    joblib.dump(hnm_svm_model, HNM_MODEL_PATH)
    print(f"[+] Đã lưu HNM SVM Model tại: {HNM_MODEL_PATH}")
    
    # Đánh giá lại
    print("\n[5] ĐÁNH GIÁ TRÊN TẬP HUẤN LUYỆN MỚI:")
    y_pred = hnm_svm_model.predict(X_train_new_scaled)
    print(f"Accuracy trên tập Train (mới): {accuracy_score(y_train_new, y_pred):.4f}")
    
    print("\nChi tiết Classification Report:")
    print(classification_report(y_train_new, y_pred, target_names=["Background", "Person"]))
    
    print("\n" + "="*60)
    print("[*] HOÀN THÀNH GIAI ĐOẠN 3!")

if __name__ == "__main__":
    run_hard_negative_mining()
