"""Run pedestrian detection for a single image."""

from pathlib import Path

import cv2

from src.config import PROJECT_ROOT
from src.models.detector import PedestrianDetector


def run_inference(
    image_path,
    output_dir=PROJECT_ROOT,
    model_path=None,
    scaler_path=None,
    win_step=16,
    scale_factor=1.15,
    hit_threshold=0.5,
    nms_overlap=0.3,
):
    """Detect people, save raw/final visualizations, and return their paths."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    detector_kwargs = {}
    if model_path is not None:
        detector_kwargs["model_path"] = model_path
    if scaler_path is not None:
        detector_kwargs["scaler_path"] = scaler_path
    detector = PedestrianDetector(**detector_kwargs)

    final_boxes, raw_boxes = detector.detect(
        image,
        win_step=win_step,
        scale_factor=scale_factor,
        hit_threshold=hit_threshold,
        nms_overlap=nms_overlap,
    )

    raw_visualization = image.copy()
    final_visualization = image.copy()
    for x1, y1, x2, y2 in raw_boxes:
        cv2.rectangle(raw_visualization, (x1, y1), (x2, y2), (0, 0, 255), 1)
    for x1, y1, x2, y2 in final_boxes:
        cv2.rectangle(final_visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_y = max(15, y1 - 8)
        cv2.putText(final_visualization, "Person", (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "result_raw_boxes.jpg"
    final_path = output_dir / "result_final_nms.jpg"
    if not cv2.imwrite(str(raw_path), raw_visualization) or not cv2.imwrite(str(final_path), final_visualization):
        raise OSError(f"Could not write inference outputs to: {output_dir}")

    return {
        "detections": len(final_boxes),
        "raw_detections": len(raw_boxes),
        "raw_output": raw_path,
        "final_output": final_path,
    }
