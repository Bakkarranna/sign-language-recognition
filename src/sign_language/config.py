"""Configuration, constants and hyperparameters for the WLASL100 pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    """Allow the data/results roots to be relocated (tests, CI smoke runs)."""
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


DATA_DIR = _env_path("SIGN_LANGUAGE_DATA_DIR", PROJECT_ROOT / "data")
RESULTS_DIR = _env_path("SIGN_LANGUAGE_RESULTS_DIR", PROJECT_ROOT / "results")
RAW_DIR = DATA_DIR / "raw"
VIDEO_DIR = RAW_DIR / "videos"
LANDMARK_DIR = DATA_DIR / "processed" / "landmarks"
MODEL_PATH = RESULTS_DIR / "sign_gru.pt"

# Official WLASL annotation file (from github.com/dxli94/WLASL, start_kit/).
WLASL_JSON = RAW_DIR / "WLASL_v0.3.json"

# WLASL100: the 100 most frequent glosses of WLASL_v0.3.json (2038 instances).
N_CLASSES = 100

# Per-frame landmark layout: 33 pose + 21 left hand + 21 right hand.
N_POSE = 33
N_HAND = 21
N_POINTS = N_POSE + 2 * N_HAND          # 75
N_COORDS = 3                            # x, y, z
FEAT_DIM = N_POINTS * N_COORDS          # 225
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

# Sequence model hyperparameters (from the build guide).
N_FRAMES = 32
HIDDEN = 256
LAYERS = 2
DROPOUT = 0.3

SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
PATIENCE = 6

# MediaPipe detection/tracking confidence gates.
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Cropping margin applied around the WLASL bounding box (fraction of size).
BBOX_MARGIN = 0.35
MAX_DOWNLOAD_WORKERS = 16
