"""Tests for the sign-language pipeline (no network, no real data files).

Fake landmark caches are written to a tmp dir and ``metadata.wlasl100_instances``
is monkeypatched, so the suite needs neither the dataset nor MediaPipe.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sign_language import augment, baseline, config, evaluate, visualize
from sign_language import dataset as ds
from sign_language import model as model_mod
from sign_language import train as train_mod
from sign_language.metadata import Instance

FAKE_GLOSSES = [f"gloss_{i:03d}" for i in range(config.N_CLASSES)]


def fake_instances(landmark_dir, per_split: int = 8, seed: int = 0) -> list[Instance]:
    """Write small random landmark caches and return matching instances."""
    rng = np.random.default_rng(seed)
    items = []
    for split, offset in (("train", 0), ("val", 100), ("test", 200)):
        for i in range(per_split):
            gloss = FAKE_GLOSSES[i]
            vid = f"{offset + i:05d}"
            seq = rng.standard_normal((config.N_FRAMES, config.N_POINTS,
                                       config.N_COORDS)).astype(np.float32)
            np.savez_compressed(landmark_dir / f"{vid}.npz", landmarks=seq)
            items.append(Instance(video_id=vid, gloss=gloss, split=split, url="",
                                  source="test", signer_id=1, frame_start=1,
                                  frame_end=-1, bbox=(0, 0, 0, 0)))
    return items


@pytest.fixture()
def fake_data(tmp_path, monkeypatch):
    """A tmp landmark dir with 3 splits of 8 glosses each, patched into config."""
    landmark_dir = tmp_path / "landmarks"
    landmark_dir.mkdir()
    monkeypatch.setattr(config, "LANDMARK_DIR", landmark_dir)
    from sign_language import metadata
    # both the instance index and the class list are synthetic, so the label
    # map is built from the same glosses the fake instances carry
    monkeypatch.setattr(metadata, "top_glosses",
                        lambda n=config.N_CLASSES: list(FAKE_GLOSSES[:n]))
    monkeypatch.setattr(metadata, "wlasl100_instances",
                        lambda: fake_instances(landmark_dir))
    return landmark_dir


def test_label_map_is_stable_and_sorted(fake_data):
    classes = ds.class_list()
    assert len(classes) == config.N_CLASSES == 100
    labels = ds.label_map()
    assert len(labels) == 100
    assert sorted(labels.values()) == list(range(100))


def test_class_weights_normalise_to_mean_one(fake_data):
    weights = ds.class_weights()
    assert weights.shape == (config.N_CLASSES,)
    assert weights.min() > 0
    assert abs(float(weights.mean()) - 1.0) < 1e-5


def test_dataset_splits_are_disjoint(fake_data):
    train = ds.SignDataset("train")
    val = ds.SignDataset("val")
    test = ds.SignDataset("test")
    ids = [it.video_id for it in train.items + val.items + test.items]
    assert len(train) == len(val) == len(test) == 8
    assert len(set(ids)) == len(ids), "splits must not share instances"
    assert all(it.split == "train" for it in train.items)


def test_dataset_getitem_shapes_and_labels(fake_data):
    train = ds.SignDataset("train")
    seq, label = train[0]
    assert seq.shape == (config.N_FRAMES, config.N_POINTS, config.N_COORDS)
    assert 0 <= label < config.N_CLASSES
    assert seq.dtype == torch.float32


def test_collate_stacks_batches(fake_data):
    train = ds.SignDataset("train")
    batch = ds.collate([train[i] for i in range(4)])
    seqs, labels = batch
    assert seqs.shape == (4, config.N_FRAMES, config.N_POINTS, config.N_COORDS)
    assert labels.shape == (4,)
    assert labels.dtype == torch.long


def test_augmentations_preserve_shape_and_swap_hands():
    seq = np.random.randn(config.N_FRAMES, config.N_POINTS, 3).astype(np.float32)
    p, n_hand, h = config.N_POSE, config.N_HAND, 2 * config.N_HAND
    mirrored = augment.mirror(seq)
    assert mirrored.shape == seq.shape
    # mirroring negates x and swaps the left/right hand landmark blocks
    expected_left = seq[:, p + n_hand:p + h].copy()
    expected_left[..., 0] *= -1.0
    np.testing.assert_allclose(mirrored[:, p:p + n_hand], expected_left)
    expected_right = seq[:, p:p + n_hand].copy()
    expected_right[..., 0] *= -1.0
    np.testing.assert_allclose(mirrored[:, p + n_hand:p + h], expected_right)
    for fn, args in ((augment.rotate, (25.0,)), (augment.scale, (1.3,)),
                     (augment.translate, (0.05, 0.02))):
        assert fn(seq, *args).shape == seq.shape


def test_bigru_forward_shapes():
    model = model_mod.SignGRU()
    x = torch.randn(4, config.N_FRAMES, config.N_POINTS, config.N_COORDS)
    out = model(x)
    assert out.shape == (4, config.N_CLASSES)
    assert torch.isfinite(out).all()
    assert model_mod.count_parameters(model) > 100_000


def test_topk_and_full_metrics():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 10, size=40)
    y_pred = y_true.copy()
    y_pred[:5] = (y_pred[:5] + 1) % 10
    m = evaluate.compute_metrics(y_true, y_pred)
    assert m["n_samples"] == 40
    assert 0.0 <= m["accuracy"] <= 1.0
    cm = np.array(m["confusion_matrix"]["matrix"])
    assert cm.sum() == 40
    logits = np.eye(10)[y_true] * 5.0
    topk = evaluate.topk_accuracy(logits, y_true, (1, 5))
    assert topk["top1"] == 1.0
    assert topk["top5"] == 1.0


def test_smoke_training_writes_checkpoint_and_results(fake_data, tmp_path):
    summary = train_mod.run_training(
        smoke=True, batch_size=8,
        out_path=tmp_path / "sign_gru.pt",
        results_dir=tmp_path,
    )
    assert (tmp_path / "sign_gru.pt").exists()
    assert (tmp_path / "sign_metrics.json").exists()
    assert (tmp_path / "sign_train_history.json").exists()
    assert summary["test"]["n_samples"] == 8
    assert 0.0 <= summary["test"]["accuracy"] <= 1.0


def test_nearest_mean_baseline_runs_on_fake_data(fake_data):
    metrics = baseline.nearest_mean("val")
    assert metrics["model"].startswith("nearest-mean")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_samples"] == 8


def test_figures_render_without_display(fake_data, tmp_path):
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 100, size=40)
    y_pred = y_true.copy()
    history = [{"epoch": 1, "train_loss": 2.0, "train_acc": 0.1,
                "val_acc": 0.2, "val_macro_f1": 0.15, "seconds": 1.0}]
    p1 = visualize.training_curves(history, tmp_path / "curves.png")
    p2 = visualize.confusion_matrix(y_true, y_pred, ds.class_list(),
                                    tmp_path / "cm.png")
    p3 = visualize.per_class_accuracy(y_true, y_pred, ds.class_list(),
                                      tmp_path / "per_class.png")
    p4 = visualize.sample_strip([it.video_id for it in ds.SignDataset("test").items][:4],
                                ["a", "b", "c", "d"], tmp_path / "strip.png")
    for p in (p1, p2, p3, p4):
        assert p.exists() and p.stat().st_size > 1000
