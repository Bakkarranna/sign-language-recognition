"""Datasets and data loading over the cached landmark sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import augment, config, metadata


def class_list() -> list[str]:
    """The 100 WLASL100 glosses, in instance-count order (the label index)."""
    return metadata.top_glosses(config.N_CLASSES)


def label_map() -> dict[str, int]:
    """gloss -> integer label, stable across runs."""
    return {g: i for i, g in enumerate(class_list())}


def landmark_path(video_id: str, landmark_dir: Path | None = None) -> Path:
    return Path(landmark_dir or config.LANDMARK_DIR) / f"{video_id}.npz"


def load_landmarks(video_id: str, landmark_dir: Path | None = None) -> np.ndarray:
    """(N_FRAMES, 75, 3) float32 landmarks for one cached instance."""
    with np.load(landmark_path(video_id, landmark_dir)) as data:
        return data["landmarks"].astype(np.float32)


def usable_instances(landmark_dir: Path | None = None) -> list[metadata.Instance]:
    """Every WLASL100 instance that has a landmark cache on disk."""
    return [it for it in metadata.wlasl100_instances()
            if landmark_path(it.video_id, landmark_dir).exists()]


def split_report(landmark_dir: Path | None = None) -> dict:
    """Instance counts per official split, restricted to usable instances."""
    items = usable_instances(landmark_dir)
    counts = metadata.split_counts(items)
    return {
        "total": len(items),
        "train": counts.get("train", 0),
        "val": counts.get("val", 0),
        "test": counts.get("test", 0),
        "n_classes": len({it.gloss for it in items}),
    }


class SignDataset(Dataset):
    """One instance per item: landmarks + integer gloss label."""

    def __init__(self, split: str, landmark_dir: Path | None = None,
                 training: bool = False) -> None:
        self.items = [it for it in usable_instances(landmark_dir)
                      if it.split == split]
        if not self.items:
            raise ValueError(f"no usable instances for split {split!r}")
        self.labels = label_map()
        self.landmark_dir = landmark_dir
        self.training = training

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.items[idx]
        seq = load_landmarks(item.video_id, self.landmark_dir)
        if self.training:
            seq = augment.train_augment(seq)
        label = self.labels[item.gloss]
        return torch.as_tensor(seq), label

    @property
    def glosses(self) -> list[str]:
        return [it.gloss for it in self.items]


def collate(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """List of (seq, label) -> (B, N_FRAMES, 75, 3) and (B,)."""
    seqs, labels = zip(*batch)
    return torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in seqs]), \
        torch.as_tensor(labels, dtype=torch.long)


def class_weights(items: list[metadata.Instance] | None = None) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1."""
    items = items or [it for it in usable_instances() if it.split == "train"]
    counts = np.zeros(len(class_list()), dtype=np.float64)
    labels = label_map()
    for it in items:
        counts[labels[it.gloss]] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return torch.as_tensor(weights / weights.mean(), dtype=torch.float32)


def all_features(split: str, landmark_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(n, FEAT_DIM) temporal-mean features, labels and glosses for a split."""
    items = [it for it in usable_instances(landmark_dir) if it.split == split]
    labels = label_map()
    feats = np.stack([load_landmarks(it.video_id, landmark_dir)
                      .reshape(config.N_FRAMES, -1).mean(axis=0)
                      for it in items]).astype(np.float32)
    y = np.asarray([labels[it.gloss] for it in items], dtype=np.int64)
    glosses = [it.gloss for it in items]
    return feats, y, glosses
