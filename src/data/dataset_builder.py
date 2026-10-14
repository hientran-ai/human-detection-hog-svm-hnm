"""Build fixed-size positive and negative samples from INRIA Person."""

import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

from src.config import (
    NEG_SAVE_DIR,
    NUM_NEG_PER_IMAGE,
    POS_SAVE_DIR,
    PROCESSED_DIR,
    TRAIN_ANN_DIR,
    TRAIN_NEG_DIR,
    TRAIN_NEG_LST,
    TRAIN_POS_DIR,
    TRAIN_POS_LST,
    WINDOW_SIZE,
)
from src.utils.inria_parser import get_bounding_boxes


def _listed_filenames(list_path):
    list_path = Path(list_path)
    if not list_path.is_file():
        raise FileNotFoundError(f"Dataset list not found: {list_path}")
    with list_path.open(encoding="utf-8", errors="ignore") as file:
        return [Path(line.strip().replace("\\", "/")).name for line in file if line.strip()]


def create_directories(clean=False):
    """Prepare output folders; only delete existing samples when requested."""
    if clean and PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    POS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    NEG_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset(clean=False, seed=42):
    """Create 64x128 training crops and return sample counts."""
    create_directories(clean=clean)
    rng = random.Random(seed)
    positive_files = _listed_filenames(TRAIN_POS_LST)
    negative_files = _listed_filenames(TRAIN_NEG_LST)

    positive_count = 0
    for filename in tqdm(positive_files, desc="Positive images"):
        image_path = TRAIN_POS_DIR / filename
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        annotation_path = TRAIN_ANN_DIR / f"{Path(filename).stem}.txt"
        for index, (xmin, ymin, xmax, ymax) in enumerate(get_bounding_boxes(annotation_path)):
            height, width = image.shape[:2]
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(width, xmax + 1), min(height, ymax + 1)
            crop = image[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, WINDOW_SIZE, interpolation=cv2.INTER_AREA)
            output = POS_SAVE_DIR / f"{Path(filename).stem}_{index}.jpg"
            if cv2.imwrite(str(output), resized):
                positive_count += 1

    negative_count = 0
    for filename in tqdm(negative_files, desc="Negative images"):
        image = cv2.imread(str(TRAIN_NEG_DIR / filename))
        if image is None:
            continue
        image_height, image_width = image.shape[:2]
        max_width = min(image_width, image_height // 2)
        if max_width < WINDOW_SIZE[0]:
            continue

        for index in range(NUM_NEG_PER_IMAGE):
            crop_width = rng.randint(WINDOW_SIZE[0], max_width)
            crop_height = crop_width * 2
            x = rng.randint(0, image_width - crop_width)
            y = rng.randint(0, image_height - crop_height)
            crop = image[y : y + crop_height, x : x + crop_width]
            resized = cv2.resize(crop, WINDOW_SIZE, interpolation=cv2.INTER_AREA)
            output = NEG_SAVE_DIR / f"{Path(filename).stem}_{index}.jpg"
            if cv2.imwrite(str(output), resized):
                negative_count += 1

    return {"positive": positive_count, "negative": negative_count}


if __name__ == "__main__":
    counts = build_dataset(clean=True)
    print(f"Created {counts['positive']} positive and {counts['negative']} negative samples.")
