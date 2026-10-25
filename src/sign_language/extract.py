"""Extract MediaPipe Holistic landmarks from WLASL videos.

Each frame becomes 75 landmarks (33 pose + 21 left hand + 21 right hand), each
(x, y, z). Output is the *normalised, resampled* sequence: centred on the
shoulder midpoint, scaled by shoulder width, then resampled to N_FRAMES so
every sequence is the same length. Missing hands are zeros.

MediaPipe 1.x removed the legacy ``mp.solutions`` API, so this uses the Tasks
API (``mediapipe.tasks.python.vision.HolisticLandmarker``) whose landmark
layout is identical: 33 pose + 21 + 21 points, all normalised to [0, 1] image
coordinates. The model bundle is fetched once from Google's public bucket.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import config, metadata

HOLISTIC_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)


def ensure_model() -> Path:
    """Return the local copy of the HolisticLandmarker .task bundle."""
    dest = config.PROJECT_ROOT / "models" / "holistic_landmarker.task"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading HolisticLandmarker model -> {dest}", flush=True)
    urllib.request.urlretrieve(HOLISTIC_URL, dest)
    return dest


# ---------------------------------------------------------------- extraction


def run_extractor():
    """Build a MediaPipe HolisticLandmarker for the Tasks API (IMAGE mode)."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    # MediaPipe 1.x options: only base_options is required. "num_poses" and
    # "min_pose_presence_confidence" no longer exist, so the detector keeps its
    # defaults (single person, 0.5 presence).
    options = vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_hand_landmarks_confidence=config.MIN_DETECTION_CONFIDENCE,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def _first_landmark_list(lms):
    """MediaPipe Tasks versions differ: pose/hand landmarks are either a flat
    list of NormalizedLandmark or a list containing one such list."""
    if not lms:
        return None
    return lms[0] if isinstance(lms[0], (list, tuple)) else lms


def landmarks_from_result(result) -> tuple[np.ndarray, float]:
    """Flatten a HolisticLandmarker result into (75, 3) landmarks + confidence."""
    frame = np.zeros((config.N_POINTS, config.N_COORDS), dtype=np.float32)
    conf = 0.0
    seen = 0
    pose = _first_landmark_list(result.pose_landmarks)
    if pose:
        for i, lm in enumerate(pose[: config.N_POSE]):
            frame[i] = (lm.x, lm.y, lm.z)
        conf += float(np.mean([getattr(lm, "visibility", 1.0) for lm in pose]))
        seen += 1
    for slot, lms in ((config.N_POSE, result.left_hand_landmarks),
                      (config.N_POSE + config.N_HAND, result.right_hand_landmarks)):
        hand = _first_landmark_list(lms)
        if hand:
            for i, lm in enumerate(hand[: config.N_HAND]):
                frame[slot + i] = (lm.x, lm.y, lm.z)
            seen += 2
    return frame, (conf / seen if seen else 0.0)


def _crop_box(width: int, height: int, bbox) -> tuple[int, int, int, int]:
    """WLASL bbox -> pixel box with margin, clamped to the frame."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return 0, 0, width, height
    mx = config.BBOX_MARGIN * (x2 - x1)
    my = config.BBOX_MARGIN * (y2 - y1)
    left = max(0, int(x1 - mx))
    top = max(0, int(y1 - my))
    right = min(width, int(x2 + mx))
    bottom = min(height, int(y2 + my))
    return left, top, right, bottom


def video_sequence(video_path: Path, frame_start: int, frame_end: int,
                   bbox) -> np.ndarray | None:
    """Return the raw (frames, 75, 3) landmark sequence for one video."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    landmarker = run_extractor()
    # WLASL annotates the signed portion with 1-indexed frame_start/frame_end.
    # When those are missing we keep the whole clip.
    first = frame_start if frame_start and frame_start > 0 else 1
    last = frame_end if frame_end and frame_end > 0 else None
    if last is not None and first > 1 and last > first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)

    frames: list[np.ndarray] = []
    idx = first
    while True:
        if last is not None and idx > last:
            break
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        height, width = frame.shape[:2]
        left, top, right, bottom = _crop_box(width, height, bbox)
        crop = frame[top:bottom, left:right]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(rgb))
        result = landmarker.detect(mp_image)
        lm, conf = landmarks_from_result(result)
        if conf > 0:
            frames.append(lm)
    cap.release()
    landmarker.close()
    if not frames:
        return None
    return np.stack(frames)


def normalize(seq: np.ndarray) -> np.ndarray:
    """Shoulder-centred, shoulder-width-scaled landmarks (see build guide)."""
    seq = np.asarray(seq, dtype=np.float32)
    left = seq[:, config.POSE_LEFT_SHOULDER]
    right = seq[:, config.POSE_RIGHT_SHOULDER]
    centre = (left + right) / 2.0
    width = np.linalg.norm(left - right, axis=-1)
    width = np.where(width > 1e-3, width, np.nan)
    with np.errstate(invalid="ignore"):
        fallback = np.nanmedian(width) if np.isfinite(width).any() else 1.0
        width = np.where(np.isfinite(width), width, fallback)
    out = (seq - centre[:, None]) / (width.mean() + 1e-6)
    # Drop frames where the pose was not detected but hands were (rare): they
    # would poison the shoulder-based centring.
    bad = ~np.isfinite(out).all(axis=(1, 2))
    out[bad] = 0.0
    return out


def resample(seq: np.ndarray, n: int = config.N_FRAMES) -> np.ndarray:
    """Fixed-length subsequence: pick n evenly spaced frames."""
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros((n, config.N_POINTS, config.N_COORDS), dtype=np.float32)
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return seq[idx]


def process_sequence(seq: np.ndarray) -> np.ndarray:
    """normalize + resample -> (N_FRAMES, 75, 3) float32."""
    return resample(normalize(seq), config.N_FRAMES).astype(np.float32)


# ------------------------------------------------------------------- worker


def _job(args) -> dict:
    item, video_dir, out_dir = args
    video = video_dir / f"{item.video_id}.mp4"
    out = out_dir / f"{item.video_id}.npz"
    if out.exists():
        return {"video_id": item.video_id, "status": "cached", "n_frames": 0}
    if not video.exists() or video.stat().st_size < 1000:
        return {"video_id": item.video_id, "status": "no_video", "n_frames": 0}
    try:
        seq = video_sequence(video, item.frame_start, item.frame_end, item.bbox)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        return {"video_id": item.video_id, "status": f"error:{type(exc).__name__}",
                "n_frames": 0}
    if seq is None or len(seq) < 2:
        return {"video_id": item.video_id, "status": "no_landmarks", "n_frames": 0}
    processed = process_sequence(seq)
    np.savez_compressed(out, landmarks=processed)
    return {"video_id": item.video_id, "status": "ok",
            "n_frames": int(len(seq)), "gloss": item.gloss, "split": item.split}


def extract_all(gloss_limit: int = config.N_CLASSES, workers: int = 4,
                limit: int = 0, out_dir: Path | None = None) -> dict:
    """Extract landmarks for every usable downloaded video.

    ``limit`` (when > 0) processes only the first ``limit`` instances, which
    keeps smoke tests fast.
    """
    items = metadata.wlasl100_instances()
    if gloss_limit and gloss_limit < config.N_CLASSES:
        keep = set(metadata.top_glosses(gloss_limit))
        items = [it for it in items if it.gloss in keep]
    if limit > 0:
        items = items[:limit]
    out_dir = out_dir or config.LANDMARK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(it, config.VIDEO_DIR, out_dir) for it in items]
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(_job, j) for j in jobs]):
            results.append(fut.result())
            if len(results) % 50 == 0:
                ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
                print(f"  {len(results)}/{len(jobs)} videos, {ok} usable",
                      flush=True)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = {"total": len(results), "statuses": counts,
               "usable": counts.get("ok", 0) + counts.get("cached", 0),
               "out_dir": str(out_dir)}
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="only process the first N instances")
    args = parser.parse_args()
    extract_all(workers=args.workers, limit=args.limit)
