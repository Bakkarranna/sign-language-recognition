"""Simple non-deep baselines to benchmark the biGRU against."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import config, evaluate
from . import dataset as ds


def _frame_features(seq: np.ndarray) -> np.ndarray:
    """(frames, 75, 3) -> flat per-frame features (frames, FEAT_DIM)."""
    return seq.reshape(seq.shape[0], -1)


def pooled_mean_features(seq: np.ndarray) -> np.ndarray:
    """Temporal-mean feature vector for one sequence."""
    return _frame_features(seq).mean(axis=0)


def nearest_mean(split: str = "val") -> dict:
    """Nearest class mean on temporal-mean features (cosine).

    The classic weak baseline: no sequence modelling at all, so a sign is
    recognised from the average of its frames.
    """
    x_train, y_train, _ = ds.all_features("train")
    x_eval, y_eval, glosses = ds.all_features(split)
    classes = ds.class_list()
    # a class absent from the train split keeps a zero mean instead of NaN
    means = np.zeros((len(classes), x_train.shape[1]), dtype=np.float32)
    for i in range(len(classes)):
        rows = x_train[y_train == i]
        if len(rows):
            means[i] = rows.mean(axis=0)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    means = means / np.maximum(norms, 1e-8)
    xn = x_eval / np.linalg.norm(x_eval, axis=1, keepdims=True)
    sims = xn @ means.T
    y_pred = sims.argmax(axis=1)
    logits = sims
    metrics = evaluate.compute_metrics(y_eval, y_pred, classes)
    metrics["top5"] = evaluate.topk_accuracy(logits, y_eval, (1, 5))["top5"]
    metrics["model"] = "nearest-mean (cosine, temporal mean)"
    return metrics


def frame_random_forest(split: str = "val") -> dict:
    """Per-frame Random Forest, then per-instance majority vote.

    Smooths frame-level predictions inside each instance: a sign moves, so a
    RF trained frame-by-frame still recognises the sign by averaging its
    frames. Trained on the train split's frames only.
    """
    classes = ds.class_list()
    train_items = [it for it in ds.usable_instances() if it.split == "train"]
    eval_items = [it for it in ds.usable_instances() if it.split == split]
    x_train = np.concatenate([_frame_features(ds.load_landmarks(it.video_id))
                              for it in train_items])
    y_train = np.concatenate([np.full(len(ds.load_landmarks(it.video_id)),
                                      ds.label_map()[it.gloss], dtype=np.int64)
                              for it in train_items])
    clf = RandomForestClassifier(n_estimators=200, random_state=config.SEED,
                                 n_jobs=-1)
    clf.fit(x_train, y_train)

    def predict(items):
        probs = []
        for it in items:
            seq = ds.load_landmarks(it.video_id)
            probs.append(clf.predict_proba(_frame_features(seq)).mean(axis=0))
        return np.asarray(probs)

    p_eval = predict(eval_items)
    y_eval = np.asarray([ds.label_map()[it.gloss] for it in eval_items])
    y_pred = p_eval.argmax(axis=1)
    metrics = evaluate.compute_metrics(y_eval, y_pred, classes)
    metrics["top5"] = evaluate.topk_accuracy(p_eval, y_eval, (1, 5))["top5"]
    metrics["model"] = "per-frame RandomForest (200 trees, mean vote)"
    return metrics
