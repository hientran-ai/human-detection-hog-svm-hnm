# Filepath: src/features/hog_extractor.py

import cv2
import numpy as np

class HOGFeatureExtractor:
    """
    Class chuyên đóng gói bộ trích xuất đặc trưng HOG của OpenCV.
    Sử dụng các tham số chuẩn theo Paper của Dalal & Triggs (2005) cho bài toán Pedestrian.
    """
    def __init__(self, win_size=(64, 128)):
        self.win_size = win_size
        
        # Các tham số chuẩn mực của HOG
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        # Khởi tạo đối tượng HOGDescriptor của OpenCV
        self.hog = cv2.HOGDescriptor(
            self.win_size, 
            block_size, 
            block_stride, 
            cell_size, 
            nbins
        )

    def compute(self, image):
        """
        Tính toán vector đặc trưng HOG cho một bức ảnh.
        
        Args:
            image: Ảnh đầu vào (numpy array), nên có kích thước bằng win_size (64x128).
            
        Returns:
            1D numpy array: Vector đặc trưng HOG (3780 chiều).
        """
        # Đảm bảo ảnh đúng kích thước trước khi trích xuất
        if (image.shape[1], image.shape[0]) != self.win_size:
            image = cv2.resize(image, self.win_size)
            
        # HOG hoạt động tốt nhất trên ảnh xám (Grayscale)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Tính toán HOG. OpenCV trả về mảng 2D (N, 1), ta cần flatten() về mảng 1D (N,)
        features = self.hog.compute(gray)
        return features.flatten()
