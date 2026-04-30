# Filepath: src/evaluate.py

import os
import cv2
import time
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import *
from src.utils.inria_parser import get_bounding_boxes
from src.models.detector import PedestrianDetector

def calculate_iou(boxA, boxB):
    """Tính chỉ số Intersection over Union (IoU) giữa 2 bounding box."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    if float(boxAArea + boxBArea - interArea) == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def run_evaluation():
    print("="*60)
    print("BẮT ĐẦU GIAI ĐOẠN 5: ĐÁNH GIÁ HIỆU NĂNG (INRIA DATASET)")
    print("="*60)
    
    detector = PedestrianDetector()

    # Đường dẫn
    TEST_DIR = os.path.join(DATASET_DIR, "Test")
    TEST_POS_DIR = os.path.join(TEST_DIR, "pos")
    TEST_NEG_DIR = os.path.join(TEST_DIR, "neg")
    TEST_ANN_DIR = os.path.join(TEST_DIR, "annotations")
    
    TEST_POS_LST = os.path.join(TEST_DIR, "pos.lst")
    TEST_NEG_LST = os.path.join(TEST_DIR, "neg.lst")

    # Tạo thư mục lưu các ảnh phân tích lỗi để đưa vào Báo cáo
    ERROR_SAVE_DIR = os.path.join(DATASET_DIR, "error_analysis")
    os.makedirs(ERROR_SAVE_DIR, exist_ok=True)
    saved_error_count = 0
    MAX_ERRORS_TO_SAVE = 10 # Chỉ lưu 10 ảnh lỗi làm mẫu để tránh đầy ổ cứng

    # Các biến thống kê
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_GT = 0
    total_time = 0.0
    total_images = 0
    
    IOU_EVAL_THRESH = 0.5 # Ngưỡng để tính là đoán đúng

    # =======================================================
    # 1. ĐÁNH GIÁ TRÊN TẬP POSITIVE (CÓ NGƯỜI)
    # =======================================================
    if os.path.exists(TEST_POS_LST):
        with open(TEST_POS_LST, 'r') as f:
            pos_files = [os.path.basename(line.strip()) for line in f.readlines() if line.strip()]
            
        print(f"\n[*] Đang đánh giá trên {len(pos_files)} ảnh POSITIVE...")
        total_images += len(pos_files)
        
        for filename in tqdm(pos_files, desc="Eval Positive"):
            img_path = os.path.join(TEST_POS_DIR, filename)
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(TEST_ANN_DIR, base_name + ".txt")
            
            img = cv2.imread(img_path)
            if img is None: continue
                
            gt_boxes = get_bounding_boxes(txt_path)
            total_GT += len(gt_boxes)
            gt_matched = [False] * len(gt_boxes)
            
            start_time = time.time()
            # Bước trượt (win_step=16) giúp test nhanh hơn
            pred_boxes, _ = detector.detect(img, win_step=16, scale_factor=1.15, hit_threshold=0.5, nms_overlap=0.3)
            total_time += (time.time() - start_time)
            
            img_has_error = False
            current_TP_boxes = []
            current_FP_boxes = []
            
            # Khớp dự đoán với thực tế
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for idx, gt_box in enumerate(gt_boxes):
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                
                if best_iou >= IOU_EVAL_THRESH and not gt_matched[best_gt_idx]:
                    total_TP += 1
                    gt_matched[best_gt_idx] = True
                    current_TP_boxes.append(pred_box)
                else:
                    total_FP += 1
                    current_FP_boxes.append(pred_box)
                    img_has_error = True # Ghi nhận có báo động giả
                    
            current_FN_boxes = []
            for idx, matched in enumerate(gt_matched):
                if not matched:
                    total_FN += 1
                    current_FN_boxes.append(gt_boxes[idx])
                    img_has_error = True # Ghi nhận có người bị bỏ sót

            # Nếu ảnh có lỗi và chưa lưu đủ quota, vẽ và lưu lại
            if img_has_error and saved_error_count < MAX_ERRORS_TO_SAVE:
                clone = img.copy()
                # Vẽ True Positive (Xanh lá)
                for box in current_TP_boxes:
                    cv2.rectangle(clone, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Vẽ False Positive (Đỏ) - Đoán nhầm
                for box in current_FP_boxes:
                    cv2.rectangle(clone, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(clone, "FP", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Vẽ False Negative (Vàng/Cam) - Bỏ sót
                for box in current_FN_boxes:
                    cv2.rectangle(clone, (box[0], box[1]), (box[2], box[3]), (0, 165, 255), 2)
                    cv2.putText(clone, "FN Missed", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
                cv2.imwrite(os.path.join(ERROR_SAVE_DIR, f"err_{base_name}.jpg"), clone)
                saved_error_count += 1

    # =======================================================
    # 2. ĐÁNH GIÁ TRÊN TẬP NEGATIVE (KHÔNG CÓ NGƯỜI)
    # =======================================================
    if os.path.exists(TEST_NEG_LST):
        with open(TEST_NEG_LST, 'r') as f:
            neg_files = [os.path.basename(line.strip()) for line in f.readlines() if line.strip()]
            
        print(f"\n[*] Đang đánh giá trên {len(neg_files)} ảnh NEGATIVE (Bẫy báo động giả)...")
        total_images += len(neg_files)
        
        for filename in tqdm(neg_files, desc="Eval Negative"):
            img_path = os.path.join(TEST_NEG_DIR, filename)
            img = cv2.imread(img_path)
            if img is None: continue
            
            start_time = time.time()
            pred_boxes, _ = detector.detect(img, win_step=16, scale_factor=1.15, hit_threshold=0.5, nms_overlap=0.3)
            total_time += (time.time() - start_time)
            
            # Mọi dự đoán trên ảnh này đều là báo động giả (False Positive)
            if len(pred_boxes) > 0:
                total_FP += len(pred_boxes)
                
                # Lưu làm mẫu lỗi nếu còn quota
                if saved_error_count < MAX_ERRORS_TO_SAVE:
                    clone = img.copy()
                    for box in pred_boxes:
                        cv2.rectangle(clone, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                        cv2.putText(clone, "FP", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(ERROR_SAVE_DIR, f"err_neg_{os.path.splitext(filename)[0]}.jpg"), clone)
                    saved_error_count += 1

    # =======================================================
    # TÍNH TOÁN METRICS CUỐI CÙNG
    # =======================================================
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_fps = total_images / total_time if total_time > 0 else 0

    print("\n" + "="*60)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (FINAL METRICS) - INRIA DATASET")
    print("="*60)
    print(f"Tổng số đối tượng thực tế (Ground Truths) : {total_GT}")
    print(f"True Positives (Nhận diện đúng)            : {total_TP}")
    print(f"False Positives (Báo động giả)              : {total_FP}")
    print(f"False Negatives (Bỏ sót)                    : {total_FN}")
    print("-" * 60)
    print(f"📌 Precision (Độ chính xác)                 : {precision:.4f} ({(precision*100):.2f}%)")
    print(f"📌 Recall (Độ bao phủ/Độ nhạy)              : {recall:.4f} ({(recall*100):.2f}%)")
    print(f"📌 F1-Score (Trung bình điều hòa)           : {f1_score:.4f} ({(f1_score*100):.2f}%)")
    print(f"⏱️ Tốc độ xử lý (FPS)                        : {avg_fps:.2f} frames/sec")
    print(f"\n[*] Đã lưu các ảnh phân tích lỗi tại: {ERROR_SAVE_DIR}")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()
