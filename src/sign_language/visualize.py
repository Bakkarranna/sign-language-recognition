"""Figures for the sign-language results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402
from . import dataset as ds

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def training_curves(history: list[dict], path: Path, title: str = "") -> Path:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy")
    ax1.legend(frameon=False)
    ax2.plot(epochs, [h["train_acc"] for h in history], "o-", label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], "s-", label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str],
                     path: Path) -> Path:
    n = len(classes)
    cm = np.bincount(y_true * n + y_pred, minlength=n * n).reshape(n, n)
    row = cm.sum(1, keepdims=True)
    cm_norm = cm / np.maximum(row, 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("predicted gloss")
    ax.set_ylabel("true gloss")
    ax.set_title("row-normalised confusion matrix (100 WLASL100 glosses)")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=5)
    ax.set_yticklabels(classes, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.03, label="recall")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                       classes: list[str], path: Path, bottom_k: int = 20) -> Path:
    recall = []
    for i in range(len(classes)):
        mask = y_true == i
        recall.append(float(np.mean(y_pred[mask] == i)) if mask.any() else 0.0)
    order = np.argsort(recall)[:bottom_k]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(order))))
    ax.barh([classes[i] for i in order], [recall[i] for i in order])
    ax.set_xlabel("recall")
    ax.set_xlim(0, 1)
    ax.set_title(f"recall on the {len(order)} hardest glosses")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def sample_strip(video_ids: list[str], glosses: list[str], path: Path,
                 landmark_dir: Path | None = None) -> Path:
    """Hand + pose trajectories for a few instances: does a sign read as a
    continuous motion even in the raw normalised landmarks?"""
    cols = min(8, len(glosses))
    rows = (len(glosses) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.6 * rows),
                             squeeze=False)
    for ax, vid, gloss in zip(axes.ravel(), video_ids, glosses):
        seq = ds.load_landmarks(vid, landmark_dir)
        hand = seq[:, config.N_POSE:]  # left then right hand
        mid = config.N_HAND
        for offset, colour, label in ((0, "tab:blue", "left"), (mid, "tab:red", "right")):
            pts = hand[:, offset:offset + config.N_HAND, :2]
            ax.plot(pts[0, :, 0], pts[0, :, 1], color=colour, lw=1)
            ax.plot(pts[:, 0, 0], pts[:, 0, 1], color=colour, lw=0.4, alpha=0.5)
            for a, b in HAND_EDGES:
                ax.plot(pts[0, a:b + 1:1, 0] if False else [pts[0, a, 0], pts[0, b, 0]],
                        [pts[0, a, 1], pts[0, b, 1]], color=colour, lw=0.8)
        ax.set_title(gloss, fontsize=8)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(glosses):]:
        ax.axis("off")
    fig.suptitle("normalised hand landmarks, first frame + wrist trajectory")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all(results_dir: Path | None = None, fig_dir: Path | None = None,
             landmark_dir: Path | None = None) -> list[Path]:
    """Write the standard figure set from the results JSON files."""
    results_dir = Path(results_dir or config.RESULTS_DIR)
    fig_dir = Path(fig_dir or (results_dir / "figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    history = json.loads((results_dir / "sign_train_history.json").read_text())
    if history:
        p = fig_dir / "training_curves.png"
        training_curves(history, p)
        written.append(p)

    pred = np.load(results_dir / "sign_predictions.npz")
    logits, labels = pred["logits"], pred["labels"]
    y_pred = logits.argmax(1)
    classes = ds.class_list()
    p = fig_dir / "confusion_matrix.png"
    confusion_matrix(labels, y_pred, classes, p)
    written.append(p)

    p = fig_dir / "per_class_recall.png"
    per_class_accuracy(labels, y_pred, classes, p)
    written.append(p)

    items = [it for it in ds.usable_instances(landmark_dir) if it.split == "test"]
    picked = items[:: max(1, len(items) // 8)][:8]
    p = fig_dir / "sample_landmarks.png"
    sample_strip([it.video_id for it in picked], [it.gloss for it in picked], p,
                 landmark_dir)
    written.append(p)
    return written


def main() -> None:
    for p in make_all():
        print(p)


if __name__ == "__main__":
    main()
