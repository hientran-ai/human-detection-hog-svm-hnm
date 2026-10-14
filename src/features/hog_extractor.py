"""OpenCV HOG feature extraction."""

import cv2
import numpy as np


class HOGFeatureExtractor:
    """Dalal-Triggs-style HOG descriptor for 64x128 person windows."""

    def __init__(self, win_size=(64, 128)):
        self.win_size = tuple(win_size)
        self.hog = cv2.HOGDescriptor(
            self.win_size,
            (16, 16),
            (8, 8),
            (8, 8),
            9,
        )

    def compute(self, image):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty NumPy array")
        if image.ndim not in (2, 3):
            raise ValueError("image must be grayscale or BGR")
        if (image.shape[1], image.shape[0]) != self.win_size:
            image = cv2.resize(image, self.win_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return self.hog.compute(gray).ravel()
