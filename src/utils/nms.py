# Filepath: src/utils/nms.py

import numpy as np

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    """
    Thuật toán Non-Maximum Suppression (NMS) để loại bỏ các bounding box chồng chéo.
    
    Args:
        boxes (list or np.ndarray): Danh sách các box [xmin, ymin, xmax, ymax].
        probs (list or np.ndarray): Điểm tự tin (confidence score) của từng box.
        overlapThresh (float): Ngưỡng IoU để quyết định xóa box (thường là 0.3 - 0.5).
        
    Returns:
        np.ndarray: Danh sách các box được giữ lại.
    """
    # Nếu không có box nào, trả về mảng rỗng
    if len(boxes) == 0:
        return []

    # Chuyển đổi sang numpy array và ép kiểu float
    if type(boxes).__module__ != np.__name__:
        boxes = np.array(boxes)
    boxes = boxes.astype("float")

    # Lấy tọa độ
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Tính diện tích của các boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sắp xếp các box theo điểm tự tin (probs) từ thấp đến cao
    if probs is not None:
        if type(probs).__module__ != np.__name__:
            probs = np.array(probs)
        idxs = np.argsort(probs)
    else:
        idxs = np.argsort(y2)

    pick = []

    # Vòng lặp xóa các box chồng chéo
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i) # Giữ lại box có điểm cao nhất

        # Tính toán tọa độ vùng giao nhau
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Tính chiều rộng, chiều cao vùng giao nhau
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Tính tỷ lệ chồng chéo (IoU overlap)
        overlap = (w * h) / area[idxs[:last]]

        # Xóa các box có độ chồng chéo lớn hơn ngưỡng overlapThresh
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")
