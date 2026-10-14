"""Project-wide paths and algorithm defaults."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset layout (INRIA Person dataset).
DATASET_DIR = PROJECT_ROOT / "inriadata"
TRAIN_DIR = DATASET_DIR / "Train"
TRAIN_POS_DIR = TRAIN_DIR / "pos"
TRAIN_NEG_DIR = TRAIN_DIR / "neg"
TRAIN_ANN_DIR = TRAIN_DIR / "annotations"
TRAIN_POS_LST = TRAIN_DIR / "pos.lst"
TRAIN_NEG_LST = TRAIN_DIR / "neg.lst"

PROCESSED_DIR = DATASET_DIR / "processed"
POS_SAVE_DIR = PROCESSED_DIR / "positive"
NEG_SAVE_DIR = PROCESSED_DIR / "negative"

# Baseline and HNM models need separate scalers because HNM changes the data.
MODEL_DIR = PROJECT_ROOT / "models"
BASE_MODEL_PATH = MODEL_DIR / "base_svm_model.pkl"
BASE_SCALER_PATH = MODEL_DIR / "base_scaler.pkl"
HNM_MODEL_PATH = MODEL_DIR / "hnm_svm_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"  # Final scaler; backward-compatible name.

WINDOW_SIZE = (64, 128)  # (width, height)
NUM_NEG_PER_IMAGE = 10
WINDOW_STEP = 32
MAX_HARD_NEGATIVES = 3000
