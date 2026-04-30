# Filepath: src/inference.py

import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.detector import PedestrianDetector
from src.config import DATASET_DIR

def run_inference(image_path):
    if not os.path.exists(image_path):
        print(f"[!] Không tìm thấy ảnh: {image_path}")
        return

    detector = PedestrianDetector()
    
    image = cv2.imread(image_path)
    clone = image.copy()
    
    print(f"\n[*] Đang nhận diện trên ảnh kích thước {image.shape}...")
    print("[*] (Viá quét tháp ảnh bằng CPU nên có thể mất vài giây...)")
    
    # Chạy hàm detect
    final_boxes, raw_boxes = detector.detect(
        image, 
        win_step=16,          # Bước trượt. Để 8 thì cực chuẩn nhưng chạy chậm. Để 16 là cân bằng.
        scale_factor=1.15, 
        hit_threshold=0.5,    # Điểm tự tin > 0.5 mới lấy
        nms_overlap=0.3       # Gộp các khung trùng lặp > 30%
    )
    
    # Vẽ TẤT CẢ các box ban đầu (màu đỏ) trước khi gộp (Để thấy sức mạnh của NMS)
    for (x1, y1, x2, y2) in raw_boxes:
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
    # Vẽ các box CUỐI CÙNG (màu xanh lá) sau khi gộp
    for (x1, y1, x2, y2) in final_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    print(f"[+] Đã tìm thấy {len(final_boxes)} người!")

    # Lưu kết quả
    cv2.imwrite("result_raw_boxes.jpg", clone)
    cv2.imwrite("result_final_nms.jpg", image)
    print("[*] Đã lưu 2 ảnh kết quả (result_raw_boxes.jpg và result_final_nms.jpg) tại thư mục gốc của project.")

if __name__ == "__main__":
    # Tìm ngẫu nhiên một ảnh trong tập Test/pos/ của INRIA
    test_image_dir = os.path.join(DATASET_DIR, "Test", "pos")
    
    test_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.png', '.jpg'))]
    if test_files:
        # Chọn ảnh đầu tiên trong danh sách để test
        sample_image_path = os.path.join(test_image_dir, test_files[0])
        print(f"[*] Ảnh đang test: {sample_image_path}")
        run_inference(sample_image_path)
    else:
        print("[!] Không tìm thấy ảnh nào trong thư mục Test/pos/")
