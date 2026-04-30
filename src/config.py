# Filepath: src/config.py

import os

# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU (PATHS)
# Tên thư mục gốc chứa dữ liệu INRIA
DATASET_DIR = "inriadata"

# Các thư mục con của tập Train
TRAIN_DIR = os.path.join(DATASET_DIR, "Train")
TRAIN_POS_DIR = os.path.join(TRAIN_DIR, "pos")       # Chứa ảnh có người
TRAIN_NEG_DIR = os.path.join(TRAIN_DIR, "neg")       # Chứa ảnh thuần nền
TRAIN_ANN_DIR = os.path.join(TRAIN_DIR, "annotations") # Chứa file .txt tọa độ

# Các file danh sách
TRAIN_POS_LST = os.path.join(TRAIN_DIR, "pos.lst")
TRAIN_NEG_LST = os.path.join(TRAIN_DIR, "neg.lst")

# Thư mục lưu trữ dữ liệu sau khi cắt (processed data)
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
POS_SAVE_DIR = os.path.join(PROCESSED_DIR, "positive") # Lưu ảnh người đã crop 64x128
NEG_SAVE_DIR = os.path.join(PROCESSED_DIR, "negative") # Lưu ảnh nền đã crop 64x128

# CẤU HÌNH LƯU MÔ HÌNH (MODEL PATHS)
MODEL_DIR = "models"
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "base_svm_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
HNM_MODEL_PATH = os.path.join(MODEL_DIR, "hnm_svm_model.pkl")

# CẤU HÌNH THUẬT TOÁN (ALGORITHM PARAMS)
# Kích thước cửa sổ chuẩn cho HOG (width, height) - Chuẩn của bài báo gốc INRIA
WINDOW_SIZE = (64, 128)

# Số lượng ảnh nền (Negative) muốn cắt ngẫu nhiên trên MỖI bức ảnh nền gốc
NUM_NEG_PER_IMAGE = 10 

# Các tham số cho Hard Negative Mining (dùng ở GĐ 3)
WINDOW_STEP = 32
MAX_HARD_NEGATIVES = 3000