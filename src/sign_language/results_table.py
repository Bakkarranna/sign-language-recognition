"""Markdown + CSV results tables for the sign project."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import config
from . import dataset as ds


def _model_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(Path(results_dir).glob("sign_*.json")):
        data = json.loads(f.read_text())
        if f.name == "sign_train_history.json":
            continue
        test = data.get("test", {})
        rows.append({
            "model": test.get("model", f.stem),
            "accuracy": round(test.get("accuracy", 0.0), 4),
            "macro_f1": round(test.get("macro_f1", 0.0), 4),
            "top5": round(test.get("top5", 0.0), 4),
            "n_test": test.get("n_samples", 0),
        })
    for name in ("nearest_mean.json", "frame_random_forest.json"):
        f = Path(results_dir) / name
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        rows.append({
            "model": data.get("model", name),
            "accuracy": round(data.get("accuracy", 0.0), 4),
            "macro_f1": round(data.get("macro_f1", 0.0), 4),
            "top5": round(data.get("top5", 0.0), 4),
            "n_test": data.get("n_samples", 0),
        })
    return rows


def write_tables(results_dir: Path | None = None) -> tuple[Path, Path]:
    results_dir = Path(results_dir or config.RESULTS_DIR)
    rows = _model_rows(results_dir)
    md_lines = [
        "# Sign language recognition - results",
        "",
        "| model | top-1 accuracy | macro F1 | top-5 | n |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} "
            f"| {r['top5']:.4f} | {r['n_test']} |")
    md = results_dir / "results_table.md"
    md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    csv_path = results_dir / "results_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "accuracy", "macro_f1",
                                                "top5", "n_test"])
        writer.writeheader()
        writer.writerows(rows)
    return md, csv_path


def write_per_class(results_dir: Path | None = None) -> Path | None:
    """Per-gloss recall on the test split, worst first."""
    results_dir = Path(results_dir or config.RESULTS_DIR)
    f = results_dir / "sign_metrics.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    per = data["test"]["per_class"]
    rows = sorted(per.items(), key=lambda kv: kv[1]["f1"])
    out = results_dir / "per_class_recall.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gloss", "precision", "recall", "f1", "support"])
        for name, v in rows:
            writer.writerow([name, round(v["precision"], 4), round(v["recall"], 4),
                             round(v["f1"], 4), v["support"]])
    return out


def dataset_summary() -> dict:
    return ds.split_report()


def main() -> None:
    md, csv_path = write_tables()
    per_class = write_per_class()
    print(md)
    print(csv_path)
    if per_class:
        print(per_class)


if __name__ == "__main__":
    main()
