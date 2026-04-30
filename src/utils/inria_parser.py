# Filepath: src/utils/inria_parser.py

import os
import re

def get_bounding_boxes(txt_path):
    """
    Đọc file .txt của INRIA dataset và trích xuất bounding box.
    Định dạng cần tìm: 
    Bounding box for object 1 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (142, 66) - (340, 646)
    
    Args:
        txt_path (str): Đường dẫn tới file annotation .txt
        
    Returns:
        list: Danh sách các bounding box [xmin, ymin, xmax, ymax].
    """
    boxes = []
    
    if not os.path.exists(txt_path):
        return boxes

    # Mẫu Regex để bắt 4 con số tọa độ
    # \d+ đại diện cho các chữ số. Ký tự \s* để bỏ qua khoảng trắng thừa.
    pattern = r"Bounding box for object \d+ .*? : \((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)"
    
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
        # Tìm tất cả các đoạn khớp với regex trong file
        matches = re.findall(pattern, content)
        
        for match in matches:
            # Chuyển đổi chuỗi sang số nguyên
            xmin = int(match[0])
            ymin = int(match[1])
            xmax = int(match[2])
            ymax = int(match[3])
            
            boxes.append([xmin, ymin, xmax, ymax])
            
    return boxes

# Có thể chạy thử file này bằng cách uncomment đoạn dưới
# if __name__ == "__main__":
#     boxes = get_bounding_boxes("../../inriadata/Train/annotations/crop_000001.txt")
#     print(boxes)
