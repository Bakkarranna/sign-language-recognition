"""Download WLASL100 videos listed in WLASL_v0.3.json.

Most WLASL videos are served as plain HTTP files; the rest are YouTube links.
We fetch the HTTP ones directly (concurrently, with retries) and try yt-dlp for
YouTube links when it is installed. A JSON manifest records what succeeded so
the extractor can skip missing clips instead of guessing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config, metadata

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

# Hosts that refuse this network outright (bot wall, dead links, HTML-only).
# They are recorded as "host_blocked" instead of being retried, which turns a
# multi-hour wall of timeouts into a few milliseconds per instance.
BLOCKED_HOSTS = frozenset({
    "www.signingsavvy.com",     # 403 bot wall (TLS fingerprint)
    "www.handspeak.com",        # connection failure from this network
    "www.aslpro.com",           # 404 - the SWF files are gone
    "www.aslsearch.com",        # serves HTML instead of video
    "aslsignbank.haskins.yale.edu",  # TLS/connection failure
})


def video_path(video_id: str) -> Path:
    return config.VIDEO_DIR / f"{video_id}.mp4"


def is_youtube(url: str) -> bool:
    return "youtu" in url


MAX_VIDEO_BYTES = 200 * 1024 * 1024      # WLASL clips are a few MB
MIN_VIDEO_BYTES = 5 * 1024                     # smaller than this is not a video


def host_of(url: str) -> str:
    return url.split("/")[2] if "//" in url else ""


def _host_referer(url: str) -> str:
    """Same-host referer: several of these hosts reject cross-site hotlinks."""
    host = host_of(url)
    return f"https://{host}/" if host else "https://www.google.com/"


def _http_fetch(url: str, out: Path, attempts: int = 3) -> int:
    """Stream url to out in chunks; return bytes written, or -1 on failure."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                url, headers={**UA, "Referer": _host_referer(url)})
            with urllib.request.urlopen(req, timeout=60) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if ctype.startswith("text/"):
                    raise urllib.error.URLError(f"got {ctype}, not a video")
                total = 0
                tmp = out.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > MAX_VIDEO_BYTES:
                            raise urllib.error.URLError("video too large")
                        fh.write(chunk)
                if total < MIN_VIDEO_BYTES:
                    tmp.unlink(missing_ok=True)
                    raise urllib.error.URLError(f"only {total} bytes")
                tmp.replace(out)
                return total
        except Exception as exc:  # noqa: BLE001 - recorded in the manifest
            last = exc
            out.with_suffix(".tmp").unlink(missing_ok=True)
            time.sleep(0.5 * (attempt + 1))
    print(f"  FAIL {url[:90]} -> {type(last).__name__}", flush=True)
    return -1


def _ytdlp_fetch(url: str, out: Path) -> int:
    # Always invoke yt-dlp through this interpreter so the venv copy is used
    # even when its Scripts directory is not on PATH.
    # The android player client works from this network where the web/tv/tv
    # clients hit YouTube's "sign in to confirm you're not a bot" wall.
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings",
           "--extractor-args", "youtube:player_client=android",
           "-f", "bv*[height<=360]/b[ext=mp4]/18/best[height<=480]", "--no-part",
           "-N", "4", "--socket-timeout", "30", "--retries", "2",
           "-o", str(out), url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception:  # noqa: BLE001
        return -1
    if proc.returncode != 0 or not out.exists():
        return -1
    return out.stat().st_size


def _fetch_one(item: metadata.Instance) -> dict:
    out = video_path(item.video_id)
    if out.exists() and out.stat().st_size > 1000:
        return {"video_id": item.video_id, "status": "cached",
                "bytes": out.stat().st_size, "url": item.url}
    if not item.url:
        return {"video_id": item.video_id, "status": "no_url", "bytes": 0,
                "url": item.url}
    if not is_youtube(item.url) and host_of(item.url) in BLOCKED_HOSTS:
        return {"video_id": item.video_id, "status": "host_blocked",
                "bytes": 0, "url": item.url}
    size = _ytdlp_fetch(item.url, out) if is_youtube(item.url) \
        else _http_fetch(item.url, out)
    status = "ok" if size > 0 else ("youtube_failed" if is_youtube(item.url) else "failed")
    return {"video_id": item.video_id, "status": status, "bytes": max(size, 0),
            "url": item.url}


def download_all(workers: int = config.MAX_DOWNLOAD_WORKERS) -> dict:
    items = metadata.wlasl100_instances()
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = config.RAW_DIR / "video_manifest.json"
    manifest: list[dict] = []
    done = ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, it): it for it in items}
        for fut in as_completed(futures):
            manifest.append(fut.result())
            done += 1
            if manifest[-1]["status"] in ("ok", "cached"):
                ok += 1
            if done % 100 == 0:
                print(f"  {done}/{len(items)} fetched, {ok} usable", flush=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    counts: dict[str, int] = {}
    for m in manifest:
        counts[m["status"]] = counts.get(m["status"], 0) + 1

    # A short summary of what is still missing helps the README stay honest.
    blocked = sorted({host_of(m["url"]) for m in manifest
                      if m["status"] in ("host_blocked", "failed",
                                         "youtube_failed")})
    print(f"manifest: {manifest_path}  {counts}", flush=True)
    return {
        "total": len(items),
        "statuses": counts,
        "usable": counts.get("ok", 0) + counts.get("cached", 0),
        "unreachable_hosts": blocked,
        "manifest": str(manifest_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int,
                        default=config.MAX_DOWNLOAD_WORKERS)
    parser.add_argument("--limit", type=int, default=0,
                        help="only fetch the first N unused instances")
    args = parser.parse_args()
    result = download_all(workers=args.workers)
    print(json.dumps(result, indent=2))
