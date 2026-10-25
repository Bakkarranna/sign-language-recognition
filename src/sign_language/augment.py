"""Landmark-sequence augmentations.

All functions take a (frames, 75, 3) array and return the same shape. Rotation
and scaling happen in the xy plane around the shoulder centre; mirroring
negates x and swaps the left/right hand landmark blocks, which is the detail
that stops a mirrored sign from becoming its own wrong label.
"""

from __future__ import annotations

import numpy as np

from . import config


def _as_seq(x):
    return np.asarray(x, dtype=np.float32)


def rotate(seq, angle_deg: float):
    """Rotate the normalized sequence in the xy plane."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    out = _as_seq(seq).copy()
    x, y = out[..., 0], out[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def scale(seq, factor: float):
    out = _as_seq(seq).copy()
    out[..., :2] = out[..., :2] * factor
    out[..., 2] = out[..., 2] * factor
    return out


def translate(seq, dx: float, dy: float):
    out = _as_seq(seq).copy()
    out[..., 0] = out[..., 0] + dx
    out[..., 1] = out[..., 1] + dy
    return out


def mirror(seq):
    """Mirror horizontally and swap the left and right hand landmark blocks."""
    out = _as_seq(seq).copy()
    out[..., 0] = -out[..., 0]
    p, n_hand, r = config.N_POSE, config.N_HAND, 2 * config.N_HAND
    hands = out[:, p:p + r].copy()
    out[:, p:p + n_hand] = hands[:, n_hand:]
    out[:, p + n_hand:p + r] = hands[:, :n_hand]
    return out


def train_augment(seq):
    """The augmentation used during training (mirror + small rotation/scale)."""
    if np.random.rand() < 0.5:
        seq = mirror(seq)
    if np.random.rand() < 0.8:
        seq = rotate(seq, np.random.uniform(-12, 12))
    if np.random.rand() < 0.8:
        seq = scale(seq, np.random.uniform(0.9, 1.1))
    if np.random.rand() < 0.5:
        seq = translate(seq, np.random.uniform(-0.05, 0.05),
                        np.random.uniform(-0.05, 0.05))
    return seq
