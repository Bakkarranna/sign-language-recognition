"""Tests for the checkpoint replay (no network, no real data files).

Fake landmark caches and a real SignGRU checkpoint are written into a tmp dir,
mirroring the convention of ``test_sign_pipeline.py``: ``metadata`` is
monkeypatched so neither the WLASL annotations nor MediaPipe are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sign_language import config, replay
from sign_language import dataset as ds
from sign_language import model as model_mod
from sign_language.metadata import Instance

FAKE_GLOSSES = [f"gloss_{i:03d}" for i in range(config.N_CLASSES)]


def fake_instances(landmark_dir, per_split: int = 6, seed: int = 0) -> list[Instance]:
    """Small random landmark caches plus matching instances per split."""
    rng = np.random.default_rng(seed)
    items = []
    for split, offset in (("train", 0), ("val", 100), ("test", 200)):
        for i in range(per_split):
            vid = f"{offset + i:05d}"
            seq = rng.standard_normal((config.N_FRAMES, config.N_POINTS,
                                       config.N_COORDS)).astype(np.float32)
            np.savez_compressed(landmark_dir / f"{vid}.npz", landmarks=seq)
            items.append(Instance(video_id=vid, gloss=FAKE_GLOSSES[i], split=split,
                                  url="", source="test", signer_id=7, frame_start=1,
                                  frame_end=-1, bbox=(0, 0, 0, 0)))
    return items


def save_checkpoint(path, labels, hidden: int = 16, layers: int = 1) -> None:
    """A real (untrained) SignGRU checkpoint in train.py's format."""
    net = model_mod.SignGRU(n_classes=len(labels), hidden=hidden, layers=layers)
    torch.save({"state_dict": net.state_dict(), "labels": list(labels),
                "config": {"hidden": hidden, "layers": layers,
                           "n_frames": config.N_FRAMES}}, path)


@pytest.fixture()
def replay_data(tmp_path, monkeypatch):
    """Fake data on disk plus the monkeypatches needed to index it."""
    landmark_dir = tmp_path / "landmarks"
    landmark_dir.mkdir()
    monkeypatch.setattr(config, "LANDMARK_DIR", landmark_dir)
    from sign_language import metadata
    monkeypatch.setattr(metadata, "top_glosses",
                        lambda n=config.N_CLASSES: list(FAKE_GLOSSES[:n]))
    monkeypatch.setattr(metadata, "wlasl100_instances",
                        lambda: fake_instances(landmark_dir))
    checkpoint = tmp_path / "sign_gru.pt"
    save_checkpoint(checkpoint, FAKE_GLOSSES)
    monkeypatch.setattr(config, "MODEL_PATH", checkpoint)
    return {"landmark_dir": landmark_dir, "checkpoint": checkpoint}


def test_load_model_reads_checkpoint_labels_and_arch(replay_data):
    model, labels = replay.load_model()
    assert labels == FAKE_GLOSSES
    # the saved config, not the defaults, sizes the network
    assert model.gru.hidden_size == 16
    assert model.gru.num_layers == 1
    assert model.head.out_features == config.N_CLASSES
    assert not model.training


def test_load_model_missing_checkpoint_says_how_to_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError, match="sign_language.train"):
        replay.load_model()


def test_predict_matches_a_direct_forward_pass(replay_data):
    """The replay is just the saved model on one instance: same softmax, same
    ranking, computed independently here."""
    model, labels = replay.load_model()
    item = [it for it in ds.usable_instances() if it.split == "test"][1]
    seq = torch.as_tensor(ds.load_landmarks(item.video_id, config.LANDMARK_DIR))[None]
    with torch.no_grad():
        probs = torch.softmax(model(seq)[0], dim=0).numpy()
    true_idx = labels.index(item.gloss)
    order = np.argsort(-probs)

    pred = replay.predict_instance(model, labels, item)
    assert pred.video_id == item.video_id
    assert pred.gloss == item.gloss
    assert pred.predicted == labels[order[0]]
    assert pred.correct == (order[0] == true_idx)
    assert pred.rank == 1 + int(np.sum(probs > probs[true_idx]))
    assert len(pred.top5) == 5
    for gloss, prob in pred.top5:
        assert gloss in labels
        assert 0.0 <= prob <= 1.0
    probs_top = [p for _, p in pred.top5]
    assert probs_top == sorted(probs_top, reverse=True)
    assert sum(probs_top) <= 1.0 + 1e-6


def test_replay_split_covers_every_test_instance(replay_data):
    preds = replay.replay_split(split="test")
    items = [it for it in ds.usable_instances() if it.split == "test"]
    assert [p.video_id for p in preds] == [it.video_id for it in items]
    assert all(p.split == "test" for p in preds)
    assert all(p.rank >= 1 for p in preds)


def test_replay_split_rejects_an_empty_split(tmp_path, monkeypatch):
    from sign_language import metadata
    monkeypatch.setattr(metadata, "wlasl100_instances", lambda: [])
    monkeypatch.setattr(config, "LANDMARK_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "sign_gru.pt")
    save_checkpoint(config.MODEL_PATH, FAKE_GLOSSES)
    with pytest.raises(ValueError, match="no usable instances"):
        replay.replay_split(split="test")
