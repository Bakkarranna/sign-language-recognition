"""Evaluation metrics for the 100-class sign classifier."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def topk_accuracy(logits: np.ndarray, labels: np.ndarray,
                  ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    """Top-k accuracy for k in ks. logits (n, n_classes), labels (n,)."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    order = np.argsort(-logits, axis=1)
    hits = {f"top{k}": 0.0 for k in ks}
    for k in ks:
        hits[f"top{k}"] = float(np.mean([label in order[i, :k] for i, label in enumerate(labels)]))
    return hits


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    classes: list[str] | None = None) -> dict:
    """Accuracy, macro F1, per-class precision/recall/F1 and the confusion
    matrix, in the same shape as the ECG project's metrics for consistency."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    classes = classes or [str(i) for i in range(n_classes)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    counts = np.bincount(y_true, minlength=n_classes)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "per_class": {
            classes[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(counts[i]),
            }
            for i in range(n_classes)
        },
        "confusion_matrix": {
            "labels": classes,
            "matrix": np.bincount(
                y_true * n_classes + y_pred,
                minlength=n_classes * n_classes,
            ).reshape(n_classes, n_classes).tolist(),
        },
    }


def worst_classes(metrics: dict, k: int = 15) -> list[tuple[str, float, int]]:
    """Classes sorted by F1 ascending: where to look when accuracy is low."""
    per = metrics["per_class"]
    rows = [(name, v["f1"], v["support"]) for name, v in per.items()]
    return sorted(rows, key=lambda r: r[1])[:k]
