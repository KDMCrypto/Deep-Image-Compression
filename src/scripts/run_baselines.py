"""Classic-codec baselines (JPEG / WebP) for the DiT-IC comparison.

At very low bitrates JPEG/WebP cannot reach the DiT-IC operating point without
downscaling, so each codec is evaluated over a grid of (scale, quality):
the image is optionally downscaled before encoding and upscaled back after
decoding.  BPP is always computed per original pixel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
CACHE_ROOT = SRC_ROOT / "models" / "cache"
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

import argparse  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms import ToTensor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import build_metrics, evaluate_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JPEG/WebP baselines")
    parser.add_argument("--images", default=str(SRC_ROOT / "images" / "kodak"))
    parser.add_argument("--out", default=str(SRC_ROOT / "result" / "baselines"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scales", default="0.25,0.35,0.5,0.7,1.0")
    parser.add_argument("--jpeg-qualities", default="1,5,10,20,30,50,70,90")
    parser.add_argument("--webp-qualities", default="1,5,10,20,30,50,70,90")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out)
    metrics_dir = SRC_ROOT / "result" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for ext in ("*.png", "*.jpg", "*.jpeg") for p in Path(args.images).glob(ext)
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    scales = [float(s) for s in args.scales.split(",")]
    metrics = build_metrics(device)
    totensor = ToTensor()

    configs = [
        ("jpeg", "JPEG", [int(q) for q in args.jpeg_qualities.split(",")], "jpg"),
        ("webp", "WEBP", [int(q) for q in args.webp_qualities.split(",")], "webp"),
    ]

    rows = []
    for codec_name, pil_format, qualities, ext in configs:
        for scale in scales:
            for quality in qualities:
                for image_path in image_paths:
                    name = image_path.stem
                    gt_pil = Image.open(image_path).convert("RGB")
                    width, height = gt_pil.size
                    gt = totensor(gt_pil).unsqueeze(0).to(device)

                    encode_pil = gt_pil
                    if scale != 1.0:
                        encode_pil = gt_pil.resize(
                            (max(1, round(width * scale)), max(1, round(height * scale))),
                            Image.Resampling.LANCZOS,
                        )

                    buffer = io.BytesIO()
                    t0 = time.perf_counter()
                    encode_pil.save(buffer, format=pil_format, quality=quality)
                    enc_ms = (time.perf_counter() - t0) * 1000
                    payload = buffer.getvalue()
                    n_bytes = len(payload)

                    encoded_dir = (
                        out_dir / "encoded" / codec_name / f"s{scale:g}" / f"q{quality:02d}"
                    )
                    encoded_dir.mkdir(parents=True, exist_ok=True)
                    (encoded_dir / f"{name}.{ext}").write_bytes(payload)

                    t0 = time.perf_counter()
                    rec_pil = Image.open(io.BytesIO(payload)).convert("RGB")
                    if rec_pil.size != (width, height):
                        rec_pil = rec_pil.resize((width, height), Image.Resampling.LANCZOS)
                    rec = totensor(rec_pil).unsqueeze(0).to(device)
                    dec_ms = (time.perf_counter() - t0) * 1000

                    scores = evaluate_pair(metrics, rec, gt)
                    rows.append(
                        {
                            "codec": codec_name,
                            "image": name,
                            "scale": scale,
                            "quality": quality,
                            "width": width,
                            "height": height,
                            "bpp": n_bytes * 8 / (width * height),
                            "bytes": n_bytes,
                            "encode_ms": enc_ms,
                            "decode_ms": dec_ms,
                            **scores,
                        }
                    )
                subset = [
                    r
                    for r in rows
                    if r["codec"] == codec_name
                    and r["quality"] == quality
                    and r["scale"] == scale
                ]
                print(
                    f"{codec_name:>4} s={scale:.2f} q{quality:02d}: "
                    f"bpp {np.mean([r['bpp'] for r in subset]):.4f} | "
                    f"PSNR {np.mean([r['psnr'] for r in subset]):.3f} | "
                    f"LPIPS {np.mean([r['lpips'] for r in subset]):.4f}"
                )

    csv_path = metrics_dir / "baselines_kodak.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for codec_name, _, qualities, _ in configs:
        for scale in scales:
            for quality in qualities:
                subset = [
                    r
                    for r in rows
                    if r["codec"] == codec_name
                    and r["quality"] == quality
                    and r["scale"] == scale
                ]
                summary.append(
                    {
                        "codec": codec_name,
                        "scale": scale,
                        "quality": quality,
                        "bpp": float(np.mean([r["bpp"] for r in subset])),
                        "psnr": float(np.mean([r["psnr"] for r in subset])),
                        "ms_ssim": float(np.mean([r["ms_ssim"] for r in subset])),
                        "lpips": float(np.mean([r["lpips"] for r in subset])),
                        "dists": float(np.mean([r["dists"] for r in subset])),
                        "clipiqa": float(np.mean([r["clipiqa"] for r in subset])),
                        "musiq": float(np.mean([r["musiq"] for r in subset])),
                        "encode_ms": float(np.mean([r["encode_ms"] for r in subset])),
                        "decode_ms": float(np.mean([r["decode_ms"] for r in subset])),
                    }
                )

    summary_path = metrics_dir / "baselines_kodak_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nSaved: {csv_path}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
