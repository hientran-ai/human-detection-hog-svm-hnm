"""Train the baseline linear SVM on HOG features."""

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from src.config import BASE_MODEL_PATH, BASE_SCALER_PATH, MODEL_DIR, NEG_SAVE_DIR, POS_SAVE_DIR, WINDOW_SIZE
from src.features.hog_extractor import HOGFeatureExtractor


def load_data_and_extract_features():
    extractor = HOGFeatureExtractor(win_size=WINDOW_SIZE)
    features, labels = [], []
    for directory, label, description in (
        (POS_SAVE_DIR, 1, "Positive samples"),
        (NEG_SAVE_DIR, 0, "Negative samples"),
    ):
        files = sorted(Path(directory).glob("*.jpg"))
        for path in tqdm(files, desc=description):
            image = cv2.imread(str(path))
            if image is not None:
                features.append(extractor.compute(image))
                labels.append(label)

    if not features:
        raise ValueError("No processed training images found. Run the dataset build step first.")
    if len(set(labels)) < 2:
        raise ValueError("Training requires at least one positive and one negative sample.")
    return np.asarray(features), np.asarray(labels)


def train_baseline_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features, labels = load_data_and_extract_features()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=42)
    model.fit(scaled_features, labels)

    joblib.dump(scaler, BASE_SCALER_PATH)
    joblib.dump(model, BASE_MODEL_PATH)
    predictions = model.predict(scaled_features)
    return {
        "samples": len(labels),
        "model_path": BASE_MODEL_PATH,
        "scaler_path": BASE_SCALER_PATH,
        "report": classification_report(labels, predictions, target_names=["Background", "Person"]),
    }


if __name__ == "__main__":
    result = train_baseline_model()
    print(result["report"])
    print(f"Saved model to {result['model_path']} and scaler to {result['scaler_path']}")
