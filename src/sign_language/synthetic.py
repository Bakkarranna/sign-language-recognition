"""Build a small synthetic WLASL-shaped dataset for tests and CI.

The real pipeline needs the ~12 MB WLASL_v0.3.json annotation file and a
landmark cache extracted from downloaded videos, neither of which is in the
repository. The unit tests monkeypatch the instance index instead, which left
CI with nothing to train on.

This module writes an annotation file in exactly the WLASL shape (one entry
per gloss, each with instances carrying the usual fields) plus random landmark
caches in the same (N_FRAMES, 75, 3) float32 layout the extractor produces, so
``metadata -> dataset -> train -> results_table -> visualize`` runs end to end
on synthetic data. Set SIGN_LANGUAGE_DATA_DIR / SIGN_LANGUAGE_RESULTS_DIR to
redirect the roots, exactly as CI does.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import config

# Two-letter split prefixes: "train" and "test" share the letter 't', so a
# single-character tag would give them the same video id and collide.
_SPLIT_PREFIX = {"train": "tr", "val": "va", "test": "te"}
_SYNTHETIC_VIDEO_ID = re.compile(r"^\d{3}(?:tr|va|te)\d{4}$")


def _is_ours(path: Path) -> bool:
    """True when ``path`` is an annotation file this module wrote."""
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(entries) and all(
        entry.get("gloss", "").startswith("synthetic_") for entry in entries)


def _reject_real_dataset(data_dir: Path, landmark_dir: Path) -> None:
    """Refuse to write over a real WLASL download.

    The default data dir *is* the real dataset's location, so without this a
    stray ``python -m sign_language.synthetic`` would replace the 12 MB
    annotation with a synthetic stub and mix fake landmarks into a genuine
    cache. Point --data-dir (or SIGN_LANGUAGE_DATA_DIR) at a scratch dir.
    """
    annotation = data_dir / "raw" / "WLASL_v0.3.json"
    if annotation.exists() and not _is_ours(annotation):
        raise SystemExit(
            f"{annotation} holds the real WLASL annotation; refusing to "
            "overwrite it. Run this with --data-dir pointing at a scratch "
            "directory.")
    if any(not _SYNTHETIC_VIDEO_ID.match(cache.stem)
           for cache in landmark_dir.glob("*.npz")):
        raise SystemExit(
            f"{landmark_dir} already contains real landmark caches; "
            "refusing to mix synthetic ones in. Run this with --data-dir "
            "pointing at a scratch directory.")


def build(data_dir: Path | None = None, results_dir: Path | None = None,
          seed: int = config.SEED, n_train: int = 64, n_val: int = 32,
          n_test: int = 32) -> dict:
    """Write synthetic annotations + landmark caches; return a summary.

    ``n_train``/``n_val``/``n_test`` are totals for the whole split, spread
    round-robin over the glosses, and the defaults are exactly the sizes
    train.py's ``--smoke`` caps at, so nothing is truncated.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    results_dir = Path(results_dir or config.RESULTS_DIR)
    raw_dir = data_dir / "raw"
    landmark_dir = data_dir / "processed" / "landmarks"
    _reject_real_dataset(data_dir, landmark_dir)
    landmark_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    totals = {"train": n_train, "val": n_val, "test": n_test}
    # gloss -> split -> instances, so one gloss can appear in several splits
    by_gloss: list[dict[str, list[dict]]] = [{} for _ in range(config.N_CLASSES)]
    written = 0
    for split, total in totals.items():
        for i in range(total):
            # Round-robin over glosses keeps every split multi-class even when
            # total < n_classes, as at the smoke sizes.
            gloss_index = i % config.N_CLASSES
            video_id = f"{gloss_index:03d}{_SPLIT_PREFIX[split]}{i:04d}"
            by_gloss[gloss_index].setdefault(split, []).append({
                "video_id": video_id,
                "split": split,
                "url": f"https://example.invalid/{video_id}",
                "source": "synthetic",
                "signer_id": 1 + (i % 8),
                "frame_start": 1,
                "frame_end": -1,
                "bbox": [0, 0, 100, 100],
            })
            seq = rng.standard_normal(
                (config.N_FRAMES, config.N_POINTS, config.N_COORDS)
            ).astype(np.float32)
            np.savez_compressed(landmark_dir / f"{video_id}.npz",
                                landmarks=seq)
            written += 1

    entries = [
        {"gloss": f"synthetic_{gi:03d}",
         "instances": [inst for split in ("train", "val", "test")
                       for inst in by_gloss[gi].get(split, [])],
        }
        for gi in range(config.N_CLASSES)
    ]
    annotation_path = raw_dir / "WLASL_v0.3.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(entries, indent=1), encoding="utf-8")

    summary = {
        "annotation_file": str(annotation_path),
        "landmark_dir": str(landmark_dir),
        "instances": written,
        "splits": totals,
        "n_classes": config.N_CLASSES,
    }
    (results_dir / "synthetic_data.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="root for raw/ and processed/ (default: config)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="where smoke outputs go (default: config)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    print(json.dumps(build(args.data_dir, args.results_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
