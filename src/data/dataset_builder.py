# Filepath: src/data/dataset_builder.py

import os
import cv2
import random
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import *
from src.utils.inria_parser import get_bounding_boxes

def create_directories():
    """Tạo các thư mục lưu trữ"""
    os.makedirs(POS_SAVE_DIR, exist_ok=True)
    os.makedirs(NEG_SAVE_DIR, exist_ok=True)
    print(f"[*] Đã chuẩn bị thư mục lưu trữ tại: {PROCESSED_DIR}")

import shutil
def create_directories():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR) # Xóa sạch dữ liệu cũ
    os.makedirs(POS_SAVE_DIR, exist_ok=True)
    os.makedirs(NEG_SAVE_DIR, exist_ok=True)
    
def build_dataset():
    create_directories()
    
    # 1. TRÍCH XUẤT POSITIVE SAMPLES (TỪ THƯ MỤC POS)
    if not os.path.exists(TRAIN_POS_LST):
        print(f"[!] Lỗi: Không tìm thấy file {TRAIN_POS_LST}")
        return

    with open(TRAIN_POS_LST, 'r') as f:
        # Lấy tên file gốc (VD: từ "Train/pos/crop_000001.png" -> "crop_000001.png")
        pos_files = [os.path.basename(line.strip()) for line in f.readlines() if line.strip()]
        
    print(f"\n[*] Đang xử lý {len(pos_files)} ảnh Positive (Có người)...")
    pos_count = 0
    
    for filename in tqdm(pos_files, desc="Processing Positive"):
        img_path = os.path.join(TRAIN_POS_DIR, filename)
        
        # Suy ra tên file annotation text tương ứng
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(TRAIN_ANN_DIR, base_name + ".txt")
        
        img = cv2.imread(img_path)
        if img is None: continue
            
        gt_boxes = get_bounding_boxes(txt_path)
        
        for idx, box in enumerate(gt_boxes):
            xmin, ymin, xmax, ymax = box
            
            # Cắt ảnh
            person_crop = img[ymin:ymax, xmin:xmax]
            if person_crop.size == 0: continue
                
            # Resize về chuẩn 64x128
            person_resized = cv2.resize(person_crop, WINDOW_SIZE)
            
            save_path = os.path.join(POS_SAVE_DIR, f"{base_name}_{idx}.jpg")
            cv2.imwrite(save_path, person_resized)
            pos_count += 1

    # 2. TRÍCH XUẤT NEGATIVE SAMPLES (TỪ THƯ MỤC NEG)
    if not os.path.exists(TRAIN_NEG_LST):
        print(f"[!] Lỗi: Không tìm thấy file {TRAIN_NEG_LST}")
        return

    with open(TRAIN_NEG_LST, 'r') as f:
        neg_files = [os.path.basename(line.strip()) for line in f.readlines() if line.strip()]

    print(f"\n[*] Đang xử lý {len(neg_files)} ảnh Negative (Thuần nền)...")
    neg_count = 0
    
    for filename in tqdm(neg_files, desc="Processing Negative"):
        img_path = os.path.join(TRAIN_NEG_DIR, filename)
        img = cv2.imread(img_path)
        if img is None: continue
            
        img_h, img_w = img.shape[:2]
        
        # Lấy NUM_NEG_PER_IMAGE mẫu từ bức ảnh nền này
        for i in range(NUM_NEG_PER_IMAGE):
            # Tính toán giới hạn chiều rộng để đảm bảo chiều cao = 2 * chiều rộng không vượt viền
            max_w = min(img_w, img_h // 2)
            if max_w <= 64: 
                continue # Ảnh quá nhỏ để lấy tỷ lệ 1:2
                
            # Random kích thước
            rand_w = random.randint(64, max_w)
            rand_h = rand_w * 2
            
            # Random tọa độ
            rand_x = random.randint(0, img_w - rand_w)
            rand_y = random.randint(0, img_h - rand_h)
            
            # Cắt ảnh
            bg_crop = img[rand_y:rand_y+rand_h, rand_x:rand_x+rand_w]
            if bg_crop.size == 0: continue
                
            # Resize
            bg_resized = cv2.resize(bg_crop, WINDOW_SIZE)
            
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(NEG_SAVE_DIR, f"{base_name}_{i}.jpg")
            cv2.imwrite(save_path, bg_resized)
            neg_count += 1

    print("\n" + "="*40)
    print(f"[*] HOÀN THÀNH GIAI ĐOẠN 1 (INRIA DATASET)")
    print(f"[*] Đã trích xuất {pos_count} ảnh Positive (Người).")
    print(f"[*] Đã trích xuất {neg_count} ảnh Negative (Nền).")
    print("="*40)

if __name__ == "__main__":
    build_dataset()
