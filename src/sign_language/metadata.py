"""WLASL_v0.3 annotation reading and instance indexing.

WLASL100 is defined as the 100 most frequent glosses of WLASL_v0.3.json; that
yields exactly 2038 video instances. Instances carry their official
train/val/test split, so we never have to invent one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config


@dataclass(frozen=True)
class Instance:
    """One signed word instance: a video clip plus its label."""

    video_id: str
    gloss: str
    split: str                      # train | val | test
    url: str
    source: str
    signer_id: int
    frame_start: int
    frame_end: int
    bbox: tuple[int, int, int, int]


def load_json(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or config.WLASL_JSON
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_instance(entry: dict[str, Any], gloss: str) -> Instance:
    x1, y1, x2, y2 = (entry.get("bbox") or [0, 0, 0, 0])[:4]
    return Instance(
        video_id=str(entry["video_id"]),
        gloss=gloss,
        split=str(entry.get("split", "train")),
        url=str(entry.get("url", "")),
        source=str(entry.get("source", "")),
        signer_id=int(entry.get("signer_id", -1)),
        frame_start=int(entry.get("frame_start", 1)),
        frame_end=int(entry.get("frame_end", -1)),
        bbox=(int(x1), int(y1), int(x2), int(y2)),
    )


def gloss_counts(entries: list[dict[str, Any]] | None = None) -> list[tuple[str, int]]:
    """Glosses ordered by instance count, most frequent first."""
    entries = entries if entries is not None else load_json()
    return sorted(
        ((e["gloss"], len(e.get("instances", []))) for e in entries),
        key=lambda kv: (-kv[1], kv[0]),
    )


def top_glosses(n: int = config.N_CLASSES) -> list[str]:
    """The n most frequent glosses: the official WLASL100 class list."""
    return [g for g, _ in gloss_counts()[:n]]


def instances(glosses: set[str] | None = None) -> list[Instance]:
    """Flatten WLASL entries into instances, optionally filtered by gloss."""
    out: list[Instance] = []
    for entry in load_json():
        gloss = str(entry["gloss"])
        if glosses is not None and gloss not in glosses:
            continue
        out.extend(_parse_instance(i, gloss) for i in entry.get("instances", []))
    return out


def wlasl100_instances() -> list[Instance]:
    """The 2038 instances belonging to the WLASL100 gloss set."""
    return instances(set(top_glosses()))


def split_counts(items: list[Instance]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for it in items:
        counts[it.split] = counts.get(it.split, 0) + 1
    return counts


def describe() -> dict[str, Any]:
    """A JSON-serialisable summary used by the README and results table."""
    items = wlasl100_instances()
    glosses = top_glosses()
    hosts: dict[str, int] = {}
    for it in items:
        host = it.url.split("/")[2] if "//" in it.url else "local"
        hosts[host] = hosts.get(host, 0) + 1
    return {
        "dataset": "WLASL100 (WLASL_v0.3.json)",
        "n_classes": len(glosses),
        "n_instances": len(items),
        "splits": split_counts(items),
        "n_signers": len({it.signer_id for it in items}),
        "url_hosts": dict(sorted(hosts.items(), key=lambda kv: -kv[1])),
        "classes": glosses,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args()
    summary = describe()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"classes:    {summary['n_classes']}")
        print(f"instances:  {summary['n_instances']}  {summary['splits']}")
        print(f"signers:    {summary['n_signers']}")
        print("url hosts:")
    for host, n in summary["url_hosts"].items():
        print(f"  {n:6d}  {host}")

