"""Mine false positives and retrain the final linear SVM."""

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from src.config import (
    BASE_MODEL_PATH,
    BASE_SCALER_PATH,
    HNM_MODEL_PATH,
    MAX_HARD_NEGATIVES,
    SCALER_PATH,
    TRAIN_NEG_DIR,
    TRAIN_NEG_LST,
    WINDOW_SIZE,
    WINDOW_STEP,
)
from src.features.hog_extractor import HOGFeatureExtractor
from src.models.train_baseline import load_data_and_extract_features
from src.utils.windowing import sliding_window


def _negative_filenames():
    if not TRAIN_NEG_LST.is_file():
        raise FileNotFoundError(f"Dataset list not found: {TRAIN_NEG_LST}")
    with TRAIN_NEG_LST.open(encoding="utf-8", errors="ignore") as file:
        return [Path(line.strip().replace("\\", "/")).name for line in file if line.strip()]


def run_hard_negative_mining():
    if not BASE_MODEL_PATH.is_file() or not BASE_SCALER_PATH.is_file():
        raise FileNotFoundError("Baseline model/scaler missing. Run the baseline training step first.")

    baseline_model = joblib.load(BASE_MODEL_PATH)
    baseline_scaler = joblib.load(BASE_SCALER_PATH)
    extractor = HOGFeatureExtractor(win_size=WINDOW_SIZE)
    hard_negatives = []

    for filename in tqdm(_negative_filenames(), desc="Mining hard negatives"):
        if len(hard_negatives) >= MAX_HARD_NEGATIVES:
            break
        image = cv2.imread(str(TRAIN_NEG_DIR / filename))
        if image is None:
            continue
        for _, _, window in sliding_window(image, WINDOW_STEP, WINDOW_SIZE):
            feature = extractor.compute(window)
            score = baseline_model.decision_function(baseline_scaler.transform([feature]))[0]
            if score > 0:
                hard_negatives.append(feature)
                if len(hard_negatives) >= MAX_HARD_NEGATIVES:
                    break

    if not hard_negatives:
        raise RuntimeError("No hard negatives found; lower the mining threshold or inspect the dataset.")

    base_features, base_labels = load_data_and_extract_features()
    features = np.vstack((base_features, np.asarray(hard_negatives)))
    labels = np.concatenate((base_labels, np.zeros(len(hard_negatives), dtype=int)))
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=42)
    model.fit(scaled_features, labels)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, HNM_MODEL_PATH)

    predictions = model.predict(scaled_features)
    return {
        "hard_negatives": len(hard_negatives),
        "samples": len(labels),
        "model_path": HNM_MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "report": classification_report(labels, predictions, target_names=["Background", "Person"]),
    }


if __name__ == "__main__":
    result = run_hard_negative_mining()
    print(result["report"])
    print(f"Mined {result['hard_negatives']} hard negatives.")
