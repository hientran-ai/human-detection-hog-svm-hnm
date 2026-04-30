# Filepath: src/models/train_baseline.py

import os
import cv2
import glob
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import *
from src.features.hog_extractor import HOGFeatureExtractor

def load_data_and_extract_features():
    """
    Đọc tất cả ảnh trong thư mục processed, trích xuất HOG và gán nhãn.
    Returns:
        X (numpy array): Ma trận đặc trưng HOG.
        y (numpy array): Vector nhãn (1: Pos, 0: Neg).
    """
    extractor = HOGFeatureExtractor(win_size=WINDOW_SIZE)
    features = []
    labels = []
    
    # 1. Đọc ảnh Positive (Nhãn 1)
    pos_files = glob.glob(os.path.join(POS_SAVE_DIR, "*.jpg"))
    print(f"[*] Đang trích xuất HOG từ {len(pos_files)} ảnh Positive...")
    for file in tqdm(pos_files, desc="Positive Images"):
        img = cv2.imread(file)
        if img is not None:
            feat = extractor.compute(img)
            features.append(feat)
            labels.append(1)
            
    # 2. Đọc ảnh Negative (Nhãn 0)
    neg_files = glob.glob(os.path.join(NEG_SAVE_DIR, "*.jpg"))
    print(f"[*] Đang trích xuất HOG từ {len(neg_files)} ảnh Negative...")
    for file in tqdm(neg_files, desc="Negative Images"):
        img = cv2.imread(file)
        if img is not None:
            feat = extractor.compute(img)
            features.append(feat)
            labels.append(0)
            
    # Chuyển đổi list sang ma trận numpy để train SVM
    X = np.array(features)
    y = np.array(labels)
    
    return X, y

def train_baseline_model():
    """Hàm chính thực thi việc train model"""
    # Tạo thư mục lưu model nếu chưa có
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Chuẩn bị dữ liệu
    print("="*50)
    print("[1] BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG HOG")
    X_train, y_train = load_data_and_extract_features()
    print(f"[+] Kích thước ma trận X_train: {X_train.shape}")
    print(f"[+] Kích thước vector y_train: {y_train.shape}")
    
    # 2. Chuẩn hóa dữ liệu (Feature Scaling)
    # SVM rất nhạy cảm với scale của dữ liệu. Cần chuẩn hóa về phân phối chuẩn (mean=0, std=1)
    print("\n[2] ĐANG CHUẨN HÓA DỮ LIỆU (Standard Scaling)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Lưu bộ chuẩn hóa để dùng cho lúc Test (Inference)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[+] Đã lưu Scaler tại: {SCALER_PATH}")
    
    # 3. Huấn luyện Linear SVM
    print("\n[3] ĐANG HUẤN LUYỆN LINEAR SVM (Sẽ mất một chút thời gian)...")
    # LinearSVC ưu việt hơn SVC(kernel='linear') về tốc độ với dữ liệu lớn
    svm_model = LinearSVC(C=1.0, max_iter=2000, class_weight='balanced', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # Lưu mô hình
    joblib.dump(svm_model, BASE_MODEL_PATH)
    print(f"[+] Đã lưu Baseline SVM Model tại: {BASE_MODEL_PATH}")
    
    # 4. Đánh giá sơ bộ trên tập Train
    print("\n[4] ĐÁNH GIÁ SƠ BỘ TRÊN TẬP HUẤN LUYỆN:")
    y_pred = svm_model.predict(X_train_scaled)
    acc = accuracy_score(y_train, y_pred)
    print(f"Accuracy trên tập Train: {acc:.4f}")
    print("\nChi tiết Classification Report:")
    print(classification_report(y_train, y_pred, target_names=["Background", "Person"]))
    
    print("="*50)
    print("[*] HOÀN THÀNH GIAI ĐOẠN 2!")

if __name__ == "__main__":
    train_baseline_model()
