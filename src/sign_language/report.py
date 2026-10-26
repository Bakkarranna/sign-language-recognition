"""Derived report data for the sign-language results.

``train.py``, ``baseline.py`` and ``evaluate.py`` write the raw artifacts
(metrics JSON, raw test predictions, training history). This module turns those
into one small JSON file - ``results/report_data.json`` - that carries
everything the dashboard and the README need:

* **bootstrap confidence intervals** for top-1 / top-5 accuracy, because a
  162-instance test split over 100 classes has a very wide error bar and a
  bare point estimate hides that;
* the **model comparison table** (SignGRU vs. the two frame-level baselines);
* a **per-class table** with support, precision, recall and F1;
* the **confusion matrix** for the main model;
* the **training curve**;
* **every test prediction** (true gloss, predicted gloss, margin, top-5), so
  the dashboard can filter and inspect single instances.

Only numpy is needed and the file is always derived, never hand-edited::

    python -m sign_language.report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config

N_BOOTSTRAP = 1000
CI_LEVEL = 95
SEED = 0


def _percentile_ci(values: np.ndarray, level: float = CI_LEVEL
                   ) -> tuple[float, float]:
    alpha = (100.0 - level) / 2.0
    lo, hi = np.percentile(values, [alpha, 100.0 - alpha])
    return float(lo), float(hi)


def _card(label: str, value: float, detail: str,
          lo: float | None = None, hi: float | None = None) -> dict:
    """One headline metric card; value is formatted by the dashboard."""
    card = {"label": label, "value": round(float(value), 4), "detail": detail}
    if lo is not None and hi is not None:
        card["ci_low"] = round(float(lo), 4)
        card["ci_high"] = round(float(hi), 4)
    return card


def _bootstrap_scores(hits: np.ndarray, n: int, rng) -> np.ndarray:
    """Resample the per-instance boolean outcomes n times."""
    m = len(hits)
    return np.array([hits[rng.integers(0, m, m)].mean() for _ in range(n)])


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, cls: np.ndarray,
               label: int) -> float:
    """F1 of a single class against integer labels and predictions."""
    t = y_true == label
    p = y_pred == label
    tp = int(np.sum(t & p))
    if tp == 0:
        return 0.0
    precision = tp / max(int(np.sum(p)), 1)
    recall = tp / max(int(np.sum(t)), 1)
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom > 0 else 0.0


def _macro_f1(labels: np.ndarray, preds: np.ndarray, class_idx: np.ndarray,
              seen: list[int]) -> float:
    if not seen:
        return 0.0
    return float(np.mean([_binary_f1(labels, preds, class_idx, c)
                          for c in seen]))


def build_report(results_dir: Path | None = None) -> dict:
    results_dir = Path(results_dir or config.RESULTS_DIR)

    metrics = json.loads((results_dir / "sign_metrics.json").read_text())
    test = metrics["test"]
    labels_list = test["confusion_matrix"]["labels"]

    pred = np.load(results_dir / "sign_predictions.npz")
    logits, labels = pred["logits"], pred["labels"]
    order = np.argsort(-logits, axis=1)
    top1 = order[:, 0]
    margins = (np.sort(logits, axis=1)[:, -1]
               - np.sort(logits, axis=1)[:, -2]).astype(float)

    rng = np.random.default_rng(SEED)
    hits1 = labels == top1
    hits5 = np.array([labels[i] in order[i, :5]
                      for i in range(len(labels))], dtype=bool)
    d1 = _bootstrap_scores(hits1, N_BOOTSTRAP, rng)
    d5 = _bootstrap_scores(hits5, N_BOOTSTRAP, rng)
    lo1, hi1 = _percentile_ci(d1)
    lo5, hi5 = _percentile_ci(d5)

    # Macro-F1 over the classes that actually appear in the test split: a class
    # with 0 support has no F1 to contribute and averaging over all 100 would
    # understate the model. The bootstrap resamples whole instances.
    seen = [i for i, g in enumerate(labels_list)
            if test["per_class"][g]["support"] > 0]
    n = len(labels)
    macro_draws = np.array([
        _macro_f1(labels[sel], top1[sel], labels_list, seen)
        for sel in (rng.integers(0, n, n) for _ in range(N_BOOTSTRAP))
    ])
    macro_point = _macro_f1(labels, top1, labels_list, seen)
    lm, hm = _percentile_ci(macro_draws)

    # ---- model comparison table -------------------------------------------
    models = [{
        "model": test.get("model", "SignGRU"),
        "accuracy": round(test["accuracy"], 4),
        "macro_f1": round(test["macro_f1"], 4),
        "top5": round(test["top5"], 4),
        "n_test": test["n_samples"],
        "level": "instance",
        "n_params": test.get("n_params"),
    }]
    for name in ("nearest_mean", "frame_random_forest"):
        f = results_dir / f"{name}.json"
        if not f.exists():
            continue
        b = json.loads(f.read_text())
        models.append({
            "model": b.get("model", name),
            "accuracy": round(b.get("accuracy", 0.0), 4),
            "macro_f1": round(b.get("macro_f1", 0.0), 4),
            "top5": round(b.get("top5", 0.0), 4),
            "n_test": b.get("n_samples", 0),
            "level": "frame" if "random_forest" in name else "instance",
        })
    models.sort(key=lambda r: -r["accuracy"])

    # ---- per-class table ---------------------------------------------------
    per_rows = [{
        "gloss": g,
        "precision": round(v["precision"], 4),
        "recall": round(v["recall"], 4),
        "f1": round(v["f1"], 4),
        "support": v["support"],
        "in_test": v["support"] > 0,
    } for g, v in test["per_class"].items()]
    per_rows.sort(key=lambda r: (-r["support"], r["gloss"]))

    matrix = np.asarray(test["confusion_matrix"]["matrix"])
    history = json.loads(
        (results_dir / "sign_train_history.json").read_text())

    figures = results_dir / "figures"
    return {
        "project": "sign-language-recognition",
        "title": "Sign Language Recognition",
        "subtitle": ("WLASL100 gloss classification from MediaPipe Holistic "
                     "landmarks, 100 classes, honest train/val/test split"),
        "repo": "Bakkarranna/sign-language-recognition",
        "metrics": [
            _card("Top-1 accuracy", hits1.mean(), "95% CI on the 162-instance "
                  "test split", lo1, hi1),
            _card("Top-5 accuracy", hits5.mean(), "correct gloss in the top five "
                  "ranked classes", lo5, hi5),
            _card("Macro F1", macro_point, "unweighted mean over the classes "
                  "present in the test split", lm, hm),
            _card("Best val accuracy", metrics.get("best_val_acc", 0.0),
                  "used for checkpoint selection", None, None),
            _card("Test instances", test["n_samples"],
                  f"{metrics['n_train']} train / {metrics['n_val']} val / "
                  f"{metrics['n_test']} test", None, None),
            _card("Parameters", test.get("n_params"),
                  "biGRU 2x256 over 32-frame landmark sequences", None, None),
        ],
        "headline": {
            "model": test.get("model", "SignGRU"),
            "top1": {"point": float(hits1.mean()), "ci_low": lo1,
                     "ci_high": hi1, "n_bootstrap": N_BOOTSTRAP,
                     "ci_level": CI_LEVEL, "se": float(d1.std())},
            "top5": {"point": float(hits5.mean()), "ci_low": lo5,
                     "ci_high": hi5, "n_bootstrap": N_BOOTSTRAP,
                     "ci_level": CI_LEVEL, "se": float(d5.std())},
            "macro_f1": {"point": round(macro_point, 4),
                         "ci_low": round(lm, 4),
                         "ci_high": round(hm, 4)},
            "n_test": test["n_samples"],
            "n_train": metrics["n_train"],
            "n_val": metrics["n_val"],
            "n_params": test.get("n_params"),
            "best_val_acc": round(metrics.get("best_val_acc", 0.0), 4),
        },
        "split": metrics.get("split_report", {}),
        "models": {
            "columns": [
                {"key": "model", "label": "Model"},
                {"key": "accuracy", "label": "Top-1"},
                {"key": "macro_f1", "label": "Macro F1"},
                {"key": "top5", "label": "Top-5"},
                {"key": "n_test", "label": "N"},
                {"key": "level", "label": "Level"},
            ],
            "rows": models,
        },
        "figures": [{"id": p.stem, "name": p.name,
                     "src": f"results/figures/{p.name}"}
                    for p in sorted(figures.glob("*.png"))]
        if figures.exists() else [],
        "sections": [
            {
                "id": "training",
                "kind": "line",
                "title": "Training curve",
                "note": ("Validation accuracy is the honest signal: training "
                         "accuracy reaches 86% by the last epoch while "
                         "validation stays near 16%, which is the 100-class "
                         "overfitting gap this project reports rather than "
                         "hides."),
                "x_label": "epoch",
                "y_label": "score",
                "series": [
                    {"name": "train accuracy",
                     "points": [[h["epoch"], round(h["train_acc"], 4)]
                                for h in history]},
                    {"name": "validation accuracy",
                     "points": [[h["epoch"], round(h["val_acc"], 4)]
                                for h in history]},
                    {"name": "validation macro F1",
                     "points": [[h["epoch"], round(h["val_macro_f1"], 4)]
                                for h in history]},
                    {"name": "train loss",
                     "points": [[h["epoch"], round(h["train_loss"] / 10, 4)]
                                for h in history]},
                ],
            },
            {
                "id": "per_class",
                "kind": "table",
                "title": "Per-gloss performance",
                "note": (
                    f"{len(seen)} of the 100 glosses appear in the test split and "
                    f"carry its {n} instances, so most of the label space is "
                    "never scored at test time - that is the small-split reality "
                    "of WLASL100, not a reporting choice."),
                "searchable": True,
                "columns": [
                    {"key": "gloss", "label": "Gloss"},
                    {"key": "support", "label": "Test support"},
                    {"key": "precision", "label": "Precision"},
                    {"key": "recall", "label": "Recall"},
                    {"key": "f1", "label": "F1"},
                    {"key": "in_test", "label": "In test split"},
                ],
                "rows": per_rows,
            },
            {
                "id": "confusion",
                "kind": "heatmap",
                "title": "Confusion matrix (test split)",
                "note": ("Rows are true glosses, columns predictions. The "
                         "diagonal is thin because most of the 100-class "
                         "space is unseen at test time."),
                "labels": labels_list,
                "matrix": matrix.tolist(),
            },
            {
                "id": "predictions",
                "kind": "predictions",
                "title": "Prediction explorer",
                "note": ("Every held-out test instance with its predicted "
                         "gloss and the logit margin between the top two "
                         "classes. Filter to the mistakes to see where the "
                         "model goes wrong."),
                "items": [{
                    "i": int(i),
                    "label": labels_list[int(labels[i])],
                    "pred": labels_list[int(top1[i])],
                    "correct": bool(hits1[i]),
                    "margin": round(float(margins[i]), 4),
                    "top5": [labels_list[int(c)] for c in order[i, :5]],
                } for i in range(n)],
            },
        ],
    }


def main() -> Path:
    out = config.RESULTS_DIR / "report_data.json"
    out.write_text(json.dumps(build_report(), indent=1), encoding="utf-8")
    print(out)
    return out


if __name__ == "__main__":
    main()
