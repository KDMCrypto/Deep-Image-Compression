"""Parallel, resumable file downloader for large model weights.

HuggingFace/PyTorch CDN throttle a single connection, so the file is split
into ranges fetched by several workers.  Finished chunks are cached next to
the target file, which makes an interrupted download resumable.

Usage:
    python src/scripts/fetch_weights.py URL OUTPUT [--workers 16] [--chunk-mb 8]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

CHUNK_MB = 8
WORKERS = 16
RETRIES = 6

_print_lock = threading.Lock()


def log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def remote_size(url: str) -> tuple[int, bool]:
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    size = int(response.headers.get("Content-Length", 0))
    accept_ranges = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    return size, accept_ranges


def chunk_path(parts_dir: Path, start: int, end: int) -> Path:
    return parts_dir / f"{start:013d}-{end:013d}.part"


def download_chunk(url: str, start: int, end: int, target: Path, stop: threading.Event) -> int:
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        return expected

    tmp = target.with_suffix(target.suffix + ".tmp")
    for attempt in range(1, RETRIES + 1):
        if stop.is_set():
            raise RuntimeError("cancelled")
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {response.status_code}")
                written = 0
                with tmp.open("wb") as f:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if stop.is_set():
                            raise RuntimeError("cancelled")
                        if block:
                            f.write(block)
                            written += len(block)
            if written != expected:
                raise RuntimeError(f"short read {written}/{expected}")
            tmp.replace(target)
            return written
        except Exception as exc:  # noqa: BLE001
            log(f"  chunk {start}: attempt {attempt}/{RETRIES} failed: {exc}")
            time.sleep(min(2 * attempt, 15))
    raise RuntimeError(f"chunk {start}-{end} failed after {RETRIES} retries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--chunk-mb", type=int, default=CHUNK_MB)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    size, accept_ranges = remote_size(args.url)
    if size <= 0:
        raise SystemExit("Server did not report Content-Length")
    if output.exists() and output.stat().st_size == size:
        log(f"Already complete: {output} ({size / 1e9:.2f} GB)")
        return
    if not accept_ranges:
        raise SystemExit("Server does not support range requests")

    chunk_bytes = args.chunk_mb * 1024 * 1024
    ranges = [
        (start, min(start + chunk_bytes - 1, size - 1))
        for start in range(0, size, chunk_bytes)
    ]
    parts_dir = output.parent / (output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    pending = [r for r in ranges if not (chunk_path(parts_dir, *r).exists() and chunk_path(parts_dir, *r).stat().st_size == r[1] - r[0] + 1)]
    done_bytes = size - sum(r[1] - r[0] + 1 for r in pending)
    log(f"Downloading {output.name}: {size / 1e9:.2f} GB, {len(pending)}/{len(ranges)} chunks pending")

    stop = threading.Event()
    started = time.time()
    completed_bytes = done_bytes
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(download_chunk, args.url, start, end, chunk_path(parts_dir, start, end), stop): (start, end)
                for start, end in pending
            }
            for future in as_completed(futures):
                completed_bytes += future.result()
                elapsed = max(time.time() - started, 1e-6)
                speed = (completed_bytes - done_bytes) / elapsed / 1e6
                log(
                    f"  {completed_bytes / 1e9:.2f}/{size / 1e9:.2f} GB "
                    f"({100 * completed_bytes / size:.1f}%) {speed:.2f} MB/s"
                )
    except KeyboardInterrupt:
        stop.set()
        log("Interrupted; finished chunks are kept for resume")
        raise

    log("Merging chunks ...")
    with output.open("wb") as dst:
        for start, end in ranges:
            with chunk_path(parts_dir, start, end).open("rb") as src:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    if output.stat().st_size != size:
        raise SystemExit(f"Size mismatch: {output.stat().st_size} != {size}")
    shutil.rmtree(parts_dir, ignore_errors=True)
    log(f"Done: {output} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
