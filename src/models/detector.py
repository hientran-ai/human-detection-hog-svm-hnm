"""HOG + linear SVM pedestrian detector."""

from pathlib import Path

import cv2
import joblib
import numpy as np

from src.config import HNM_MODEL_PATH, SCALER_PATH, WINDOW_SIZE
from src.features.hog_extractor import HOGFeatureExtractor
from src.utils.nms import non_max_suppression
from src.utils.windowing import sliding_window


class PedestrianDetector:
    def __init__(self, model_path=HNM_MODEL_PATH, scaler_path=SCALER_PATH):
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)
        missing = [str(path) for path in (model_path, scaler_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing model artifact(s): " + ", ".join(missing) + ". Run training first."
            )

        self.svm = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.extractor = HOGFeatureExtractor(win_size=WINDOW_SIZE)

        expected_features = self.extractor.hog.getDescriptorSize()
        for artifact_name, artifact in (("model", self.svm), ("scaler", self.scaler)):
            feature_count = getattr(artifact, "n_features_in_", expected_features)
            if feature_count != expected_features:
                raise ValueError(
                    f"The {artifact_name} expects {feature_count} features, "
                    f"but HOG produces {expected_features}."
                )

    @staticmethod
    def image_pyramid(image, scale_factor=1.15, min_size=WINDOW_SIZE):
        """Yield progressively downscaled images and their original scale."""
        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty NumPy array")
        if scale_factor <= 1:
            raise ValueError("scale_factor must be greater than 1")

        current = image
        current_scale = 1.0
        yield current, current_scale

        while True:
            width = int(current.shape[1] / scale_factor)
            height = int(current.shape[0] / scale_factor)
            if width < min_size[0] or height < min_size[1]:
                break
            current = cv2.resize(current, (width, height), interpolation=cv2.INTER_AREA)
            current_scale *= scale_factor
            yield current, current_scale

    def detect(self, image, win_step=16, scale_factor=1.15, hit_threshold=0.5, nms_overlap=0.3):
        """Return ``(final_boxes, raw_boxes)`` for one BGR or grayscale image."""
        if not 0 <= nms_overlap <= 1:
            raise ValueError("nms_overlap must be between 0 and 1")

        boxes = []
        confidences = []
        for resized_image, scale in self.image_pyramid(image, scale_factor, WINDOW_SIZE):
            for x, y, window in sliding_window(resized_image, win_step, WINDOW_SIZE):
                feature = self.extractor.compute(window)
                scaled_feature = self.scaler.transform([feature])
                score = float(self.svm.decision_function(scaled_feature)[0])
                if score <= hit_threshold:
                    continue

                orig_x = round(x * scale)
                orig_y = round(y * scale)
                orig_w = round(WINDOW_SIZE[0] * scale)
                orig_h = round(WINDOW_SIZE[1] * scale)
                boxes.append([orig_x, orig_y, orig_x + orig_w, orig_y + orig_h])
                confidences.append(score)

        raw_boxes = np.asarray(boxes, dtype=int).reshape(-1, 4)
        final_boxes = non_max_suppression(raw_boxes, confidences, nms_overlap)
        return final_boxes, raw_boxes
