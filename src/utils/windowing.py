# Filepath: src/utils/windowing.py

def sliding_window(image, step_size, window_size):
    """
    Trượt một cửa sổ có kích thước cố định qua bức ảnh.
    
    Args:
        image (numpy array): Bức ảnh cần quét.
        step_size (int): Bước nhảy của cửa sổ trượt (pixels).
        window_size (tuple): Kích thước cửa sổ (width, height).
        
    Yields:
        tuple: (tọa độ x, tọa độ y, vùng ảnh bị cắt).
    """
    # Trượt dọc theo chiều cao (y)
    for y in range(0, image.shape[0] - window_size[1], step_size):
        # Trượt ngang theo chiều rộng (x)
        for x in range(0, image.shape[1] - window_size[0], step_size):
            # Trả về tọa độ (x, y) của góc trên bên trái và vùng ảnh (patch)
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
