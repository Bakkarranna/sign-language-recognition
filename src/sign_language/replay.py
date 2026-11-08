"""Replay the committed SignGRU checkpoint on individual WLASL instances.

``train.py`` writes ``results/sign_gru.pt`` (weights plus the class list that
matches the label indices). This module reloads that checkpoint on demand and
runs it on a single instance, so the replay app never retrains anything and its
predictions come from the same weights that produced ``results/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import config, metadata  # noqa: E402
from . import dataset as ds  # noqa: E402
from . import model as model_mod  # noqa: E402
from .visualize import HAND_EDGES  # noqa: E402

# MediaPipe pose skeleton over the 33-blazepose landmarks of the cache.
POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16
_LEFT_HAND, _RIGHT_HAND = config.N_POSE, config.N_POSE + config.N_HAND


@dataclass(frozen=True)
class Prediction:
    """One instance as the committed model sees it."""

    video_id: str
    gloss: str                 # ground truth
    split: str
    signer_id: int
    predicted: str             # top-1 gloss
    correct: bool
    rank: int                  # 1-based rank of the true gloss
    top5: tuple[tuple[str, float], ...]

    @property
    def top5_text(self) -> str:
        return "  ".join(f"{g} {p:.2f}" for g, p in self.top5)

    def as_row(self) -> dict:
        return {"video_id": self.video_id, "true": self.gloss,
                "predicted": self.predicted, "correct": self.correct,
                "rank": self.rank, "top5": self.top5_text,
                "signer": self.signer_id}


def load_model(path: Path | None = None) -> tuple[model_mod.SignGRU, list[str]]:
    """(eval-mode model, class labels) from the committed checkpoint."""
    path = Path(path or config.MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}; run `python -m sign_language.train` first")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    labels = list(ckpt["labels"])
    saved = ckpt.get("config") or {}
    net = model_mod.SignGRU(n_classes=len(labels),
                            hidden=saved.get("hidden", config.HIDDEN),
                            layers=saved.get("layers", config.LAYERS))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, labels


def predict_instance(model: model_mod.SignGRU, labels: list[str],
                     item: metadata.Instance, landmark_dir: Path | None = None,
                     topk: int = 5) -> Prediction:
    """Run the model on one cached instance. No augmentation, as at eval time."""
    seq = torch.as_tensor(ds.load_landmarks(item.video_id, landmark_dir))[None]
    with torch.no_grad():
        probs = torch.softmax(model(seq)[0], dim=0).numpy()
    true_idx = labels.index(item.gloss)
    order = np.argsort(-probs)
    top = [(labels[i], float(probs[i])) for i in order[:topk]]
    return Prediction(video_id=item.video_id, gloss=item.gloss, split=item.split,
                      signer_id=item.signer_id, predicted=top[0][0],
                      correct=top[0][0] == item.gloss,
                      rank=1 + int((probs > probs[true_idx]).sum()),
                      top5=tuple(top))


def replay_split(split: str = "test", landmark_dir: Path | None = None,
                 model_path: Path | None = None,
                 topk: int = 5) -> list[Prediction]:
    """Every usable instance of one split, replayed through the checkpoint."""
    items = [it for it in ds.usable_instances(landmark_dir) if it.split == split]
    if not items:
        raise ValueError(f"no usable instances for split {split!r}")
    model, labels = load_model(model_path)
    return [predict_instance(model, labels, it, landmark_dir, topk) for it in items]


def summary(predictions: list[Prediction]) -> dict:
    """Top-1/top-5 over a replayed split, with chance level for context."""
    n = len(predictions)
    top1 = sum(p.correct for p in predictions) / max(n, 1)
    top5 = sum(p.rank <= 5 for p in predictions) / max(n, 1)
    return {"n": n, "top1": top1, "top5": top5,
            "chance_top1": 1.0 / config.N_CLASSES,
            "n_correct": sum(p.correct for p in predictions)}


def _valid(points: np.ndarray) -> np.ndarray:
    """xy coordinates of the landmarks MediaPipe actually detected."""
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    return xy[~np.isnan(xy).any(axis=1)]


def landmark_figure(seq: np.ndarray, frame: int = 0, gloss: str = "",
                    predicted: str = "") -> plt.Figure:
    """Pose plus both hands of one frame, with the wrist trails so far."""
    frame = int(np.clip(frame, 0, len(seq) - 1))
    fig, ax = plt.subplots(figsize=(4.2, 4.8))
    all_pts: list[np.ndarray] = []

    def draw(points: np.ndarray, colour: str, lw: float) -> None:
        xy = _valid(points)
        if len(xy):
            ax.scatter(xy[:, 0], xy[:, 1], s=6, color=colour, zorder=3)
            all_pts.append(xy)

    pose = _valid(seq[frame, :config.N_POSE])
    if len(pose):
        all_pts.append(pose)
        for a, b in POSE_EDGES:
            pa, pb = seq[frame, a, :2], seq[frame, b, :2]
            if not (np.isnan(pa).any() or np.isnan(pb).any()):
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="0.35", lw=1.2,
                        zorder=2)

    for start, colour in ((_LEFT_HAND, "tab:blue"), (_RIGHT_HAND, "tab:red")):
        hand = seq[frame, start:start + config.N_HAND]
        draw(hand, colour, 0)
        for a, b in HAND_EDGES:
            pa, pb = hand[a, :2], hand[b, :2]
            if not (np.isnan(pa).any() or np.isnan(pb).any()):
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=colour, lw=0.8,
                        zorder=3)

    for wrist, colour in ((POSE_LEFT_WRIST, "tab:blue"),
                          (POSE_RIGHT_WRIST, "tab:red")):
        trail = _valid(seq[:frame + 1, wrist])
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], color=colour, lw=0.8, alpha=0.55,
                    zorder=1)
            all_pts.append(trail)

    if all_pts:
        stacked = np.vstack(all_pts)
        x0, x1 = stacked[:, 0].min(), stacked[:, 0].max()
        y0, y1 = stacked[:, 1].min(), stacked[:, 1].max()
        span = max(x1 - x0, y1 - y0, 1e-3)
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        pad = 0.12 * span
        ax.set_xlim(mid_x - span / 2 - pad, mid_x + span / 2 + pad)
        ax.set_ylim(mid_y - span / 2 - pad, mid_y + span / 2 + pad)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"frame {frame + 1}/{len(seq)}"
    if gloss:
        title += f"  -  true: {gloss}"
    if predicted:
        title += f"  -  predicted: {predicted}"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def topk_figure(pred: Prediction) -> plt.Figure:
    """The model's top-5 probabilities, with the true gloss always shown."""
    entries = list(pred.top5)
    if pred.gloss not in {g for g, _ in entries}:
        entries.append((pred.gloss, 0.0))
        entries.sort(key=lambda e: -e[1])
    glosses = [g for g, _ in entries]
    probs = [p for _, p in entries]
    colours = ["tab:green" if g == pred.gloss else "tab:blue" for g in glosses]
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.barh(range(len(glosses)), probs, color=colours)
    ax.set_yticks(range(len(glosses)))
    ax.set_yticklabels(glosses, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    ax.set_title("top-5 (green = true gloss)", fontsize=9)
    fig.tight_layout()
    return fig
