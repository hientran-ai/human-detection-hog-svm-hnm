# Filepath: src/models/detector.py

import cv2
import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import *
from src.utils.windowing import sliding_window
from src.utils.nms import non_max_suppression
from src.features.hog_extractor import HOGFeatureExtractor

class PedestrianDetector:
    def __init__(self, model_path=HNM_MODEL_PATH, scaler_path=SCALER_PATH):
        """Khởi tạo bộ nhận diện, nạp mô hình SVM (đã qua HNM), bộ chuẩn hóa và HOG Extractor."""
        print("[*] Đang khởi tạo Bộ nhận diện (Pedestrian Detector)...")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Không tìm thấy Model hoặc Scaler. Hãy chắc chắn GĐ 3 đã chạy thành công.")
            
        self.svm = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.extractor = HOGFeatureExtractor(win_size=WINDOW_SIZE)
        print(f"[+] Đã tải xong Model: {os.path.basename(model_path)}")

    def image_pyramid(self, image, scale_factor=1.15, min_size=(64, 128)):
        """Tạo tháp ảnh bằng cách thu nhỏ ảnh dần dần."""
        yield (image, 1.0)
        
        current_scale = 1.0
        while True:
            w = int(image.shape[1] / scale_factor)
            h = int(image.shape[0] / scale_factor)
            image = cv2.resize(image, (w, h))
            current_scale *= scale_factor
            
            # Dừng lại nếu ảnh nhỏ hơn cửa sổ HOG 64x128
            if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
                break
                
            yield (image, current_scale)

    def detect(self, image, win_step=16, scale_factor=1.15, hit_threshold=0.5, nms_overlap=0.3):
        """Thực thi quét và nhận diện trên ảnh."""
        boxes = []
        confidences = []

        # 1. Quét qua từng tầng Tháp ảnh
        for (resized_img, scale) in self.image_pyramid(image, scale_factor=scale_factor, min_size=WINDOW_SIZE):
            
            # 2. Trượt Cửa sổ trên ảnh hiện tại
            for (x, y, window) in sliding_window(resized_img, step_size=win_step, window_size=WINDOW_SIZE):
                
                # Trích xuất và chuẩn hóa đặc trưng
                feat = self.extractor.compute(window)
                feat_scaled = self.scaler.transform([feat])
                
                # Tính điểm tự tin (khoảng cách tới siêu mặt phẳng của SVM)
                score = self.svm.decision_function(feat_scaled)[0]
                
                # 3. Chỉ lấy box nếu tự tin vượt ngưỡng
                if score > hit_threshold:
                    # Chuyển đổi tọa độ từ ảnh thu nhỏ về ảnh gốc bằng cách NHÂN với scale
                    orig_x = int(x * scale)
                    orig_y = int(y * scale)
                    orig_w = int(WINDOW_SIZE[0] * scale)
                    orig_h = int(WINDOW_SIZE[1] * scale)
                    
                    boxes.append([orig_x, orig_y, orig_x + orig_w, orig_y + orig_h])
                    confidences.append(score)

        # 4. Áp dụng NMS
        if len(boxes) > 0:
            final_boxes = non_max_suppression(boxes, probs=confidences, overlapThresh=nms_overlap)
            return final_boxes, np.array(boxes)
        else:
            return np.array([]), np.array([])
