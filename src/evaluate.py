"""Evaluate the detector on the INRIA Person test split."""

import time
from pathlib import Path

import cv2
from tqdm import tqdm

from src.config import DATASET_DIR
from src.models.detector import PedestrianDetector
from src.utils.inria_parser import get_bounding_boxes


def calculate_iou(box_a, box_b):
    """Calculate IoU for boxes with exclusive bottom-right coordinates."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _listed_filenames(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset list not found: {path}")
    with path.open(encoding="utf-8", errors="ignore") as file:
        return [Path(line.strip().replace("\\", "/")).name for line in file if line.strip()]


def _draw_error_image(image, true_positives, false_positives, false_negatives, output):
    visualization = image.copy()
    for box in true_positives:
        cv2.rectangle(visualization, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
    for box in false_positives:
        cv2.rectangle(visualization, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 2)
    for box in false_negatives:
        cv2.rectangle(visualization, tuple(box[:2]), tuple(box[2:]), (0, 165, 255), 2)
    cv2.imwrite(str(output), visualization)


def run_evaluation(iou_threshold=0.5, max_error_images=10):
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")

    test_dir = DATASET_DIR / "Test"
    positive_dir = test_dir / "pos"
    negative_dir = test_dir / "neg"
    annotation_dir = test_dir / "annotations"
    positive_files = _listed_filenames(test_dir / "pos.lst")
    negative_files = _listed_filenames(test_dir / "neg.lst")
    error_dir = DATASET_DIR / "error_analysis"
    error_dir.mkdir(parents=True, exist_ok=True)

    detector = PedestrianDetector()
    true_positive_count = false_positive_count = false_negative_count = 0
    ground_truth_count = processed_images = saved_errors = 0
    elapsed = 0.0

    for filename in tqdm(positive_files, desc="Evaluating positives"):
        image = cv2.imread(str(positive_dir / filename))
        if image is None:
            continue
        ground_truths = get_bounding_boxes(annotation_dir / f"{Path(filename).stem}.txt")
        # INRIA annotations use inclusive maximum coordinates; normalize here.
        ground_truths = [[x1, y1, x2 + 1, y2 + 1] for x1, y1, x2, y2 in ground_truths]
        ground_truth_count += len(ground_truths)
        processed_images += 1
        started = time.perf_counter()
        predictions, _ = detector.detect(image)
        elapsed += time.perf_counter() - started

        unmatched = set(range(len(ground_truths)))
        true_boxes, false_boxes = [], []
        for prediction in predictions:
            candidates = [(calculate_iou(prediction, ground_truths[index]), index) for index in unmatched]
            best_iou, best_index = max(candidates, default=(0.0, None))
            if best_index is not None and best_iou >= iou_threshold:
                unmatched.remove(best_index)
                true_positive_count += 1
                true_boxes.append(prediction)
            else:
                false_positive_count += 1
                false_boxes.append(prediction)

        missed_boxes = [ground_truths[index] for index in sorted(unmatched)]
        false_negative_count += len(missed_boxes)
        if (false_boxes or missed_boxes) and saved_errors < max_error_images:
            _draw_error_image(image, true_boxes, false_boxes, missed_boxes, error_dir / f"err_{Path(filename).stem}.jpg")
            saved_errors += 1

    for filename in tqdm(negative_files, desc="Evaluating negatives"):
        image = cv2.imread(str(negative_dir / filename))
        if image is None:
            continue
        processed_images += 1
        started = time.perf_counter()
        predictions, _ = detector.detect(image)
        elapsed += time.perf_counter() - started
        false_positive_count += len(predictions)
        if len(predictions) and saved_errors < max_error_images:
            _draw_error_image(image, [], predictions, [], error_dir / f"err_neg_{Path(filename).stem}.jpg")
            saved_errors += 1

    precision_denominator = true_positive_count + false_positive_count
    recall_denominator = true_positive_count + false_negative_count
    precision = true_positive_count / precision_denominator if precision_denominator else 0.0
    recall = true_positive_count / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "images": processed_images,
        "ground_truths": ground_truth_count,
        "true_positives": true_positive_count,
        "false_positives": false_positive_count,
        "false_negatives": false_negative_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fps": processed_images / elapsed if elapsed else 0.0,
        "error_dir": error_dir,
    }


if __name__ == "__main__":
    metrics = run_evaluation()
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
