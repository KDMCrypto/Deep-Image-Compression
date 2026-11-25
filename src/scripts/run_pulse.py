"""Add Microsoft GenCodec PULSE to the Kodak comparison of this experiment.

PULSE (https://github.com/microsoft/GenCodec/tree/main/PULSE) lives in the
`pulse` branch of this project, with its own Python environment, pinned
upstream source and the four released model bundles. This script joins the
two worlds in two separate steps, so neither environment has to contain the
other's packages:

1. Coding step (PULSE environment, subprocess): the official PULSE
   `evaluate.py` encodes and independently decodes every Kodak image at every
   requested QP. It reports the real `.pulse` stream size (header included)
   and saves each reconstruction as a uint8 PNG, i.e. exactly what a receiver
   would get.
2. Scoring step (this environment): the saved reconstructions are scored with
   `metrics_utils.build_metrics`, the same pyiqa / torchmetrics code that
   scored DiT-IC, JPEG and WebP, so every number in the plots comes from one
   metric implementation.

Outputs (same columns as the JPEG/WebP baseline files, with `qp` in place of
`scale`/`quality`):
    src/result/metrics/pulse_kodak.csv          one row per image, model and QP
    src/result/metrics/pulse_kodak_summary.csv  one row per model and QP (means)
    src/result/pulse/<model>/evaluate.json      raw output of PULSE evaluate.py
    src/result/pulse/<model>/rec/qp<N>/*.png    reconstructions (git-ignored)

Rate: bpp = 8 * stream_bytes / (width * height) of the original image, like
every other codec here. QP 0 is the lowest rate, QP 7 the highest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
CACHE_ROOT = SRC_ROOT / "models" / "cache"
# Same cache locations as the other scripts, so metric weights are shared.
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms import ToTensor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import build_metrics, evaluate_pair  # noqa: E402

# The four released bundles, from the smallest to the largest decoder.
PULSE_MODELS = ("pulse-s-mse", "pulse-mse", "pulse-perceptual", "pulse-l-perceptual")
METRIC_NAMES = ("psnr", "ms_ssim", "lpips", "dists", "clipiqa", "musiq")
DEFAULT_IMAGES = SRC_ROOT / "images" / "kodak"
# make_plots.py reads <DEFAULT_TAG>_summary.csv and labels it as Kodak.
DEFAULT_TAG = "pulse_kodak"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PULSE on Kodak with the DiT-IC metrics")
    parser.add_argument(
        "--pulse-root",
        required=True,
        help="checkout of the `pulse` branch that holds .venv, checkpoints/ and scripts/pulse.py",
    )
    parser.add_argument("--images", default=str(DEFAULT_IMAGES))
    parser.add_argument("--out", default=str(SRC_ROOT / "result" / "pulse"))
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="name of the CSV files in src/result/metrics; must be changed for a non-Kodak image set",
    )
    parser.add_argument("--models", default=",".join(PULSE_MODELS))
    parser.add_argument("--qps", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--recode",
        action="store_true",
        help="run the PULSE coding step again even if evaluate.json already exists",
    )
    return parser.parse_args()


def coding_settings(args: argparse.Namespace, image_paths: list[Path]) -> dict:
    """Everything that decides the content of evaluate.json, stored next to it."""
    return {
        "images": [str(p) for p in image_paths],
        "qps": [int(q) for q in args.qps.split(",")],
        "device": args.device,
    }


def run_pulse_coding(args: argparse.Namespace, model: str, model_out: Path, image_paths: list[Path]) -> Path:
    """Step 1: encode and decode all images with one PULSE bundle.

    Runs in the PULSE environment as a separate process, so its GPU memory is
    released before the metric networks are loaded. An existing evaluate.json
    is reused only when it was made with the same images, QPs and device
    (run_settings.json), which makes an interrupted run resumable model by
    model without ever mixing results of different settings.
    """
    pulse_root = Path(args.pulse_root).resolve()
    result_json = model_out / "evaluate.json"
    settings_json = model_out / "run_settings.json"
    settings = coding_settings(args, image_paths)
    if result_json.exists() and not args.recode:
        previous = json.loads(settings_json.read_text(encoding="utf-8")) if settings_json.exists() else None
        if previous != settings:
            raise SystemExit(
                f"[{model}] {result_json} was made with other settings (images, QPs or device); "
                "use --recode to replace it, or a new --out directory"
            )
        print(f"[{model}] reusing {result_json}")
        return result_json

    command = [
        str(pulse_root / ".venv" / "bin" / "python"),
        str(pulse_root / "scripts" / "pulse.py"),
        "evaluate",
        "--model", str(pulse_root / "checkpoints" / model),
        "--input", str(Path(args.images).resolve()),
        "--output", str(result_json),
        "--save-reconstructions", str(model_out / "rec"),
        "--device", args.device,
        "--qps", *args.qps.split(","),
    ]
    print(f"[{model}] coding: {' '.join(command)}", flush=True)
    model_out.mkdir(parents=True, exist_ok=True)
    # The log keeps the per-image JSON lines that evaluate.py prints.
    with (model_out / "evaluate.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, cwd=pulse_root, stdout=log, stderr=subprocess.STDOUT)
    settings_json.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return result_json


def score_model(
    model: str, result_json: Path, model_out: Path, images_root: Path, metrics: dict, device
) -> list[dict]:
    """Step 2: score every saved reconstruction of one model against its original."""
    evaluation = json.loads(result_json.read_text(encoding="utf-8"))
    totensor = ToTensor()
    rows = []
    for item in evaluation["images"]:
        original_path = Path(item["image"])
        qp = int(item["qp"])
        # evaluate.py keeps the folder structure below --input, so subfolders
        # of the image set map to subfolders of rec/qp<N>/.
        relative = original_path.relative_to(images_root)
        name = relative.with_suffix("").as_posix()
        reconstruction_path = model_out / "rec" / f"qp{qp}" / relative.with_suffix(".png")

        gt_pil = Image.open(original_path).convert("RGB")
        rec_pil = Image.open(reconstruction_path).convert("RGB")
        if rec_pil.size != gt_pil.size:
            raise RuntimeError(f"{reconstruction_path} has size {rec_pil.size}, expected {gt_pil.size}")
        width, height = gt_pil.size

        # Same tensor layout as run_baselines.py: RGB in [0, 1], shape [1, 3, H, W].
        gt = totensor(gt_pil).unsqueeze(0).to(device)
        rec = totensor(rec_pil).unsqueeze(0).to(device)
        scores = evaluate_pair(metrics, rec, gt)

        rows.append(
            {
                "codec": model,
                "image": name,
                "qp": qp,
                "width": width,
                "height": height,
                # Real stream size from evaluate.py, per original pixel.
                "bpp": item["bytes"] * 8 / (width * height),
                "bytes": item["bytes"],
                "encode_ms": item["encode_ms"],
                "decode_ms": item["decode_ms"],
                **scores,
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Average over the images for every (model, QP) pair, as the baseline summary does."""
    summary = []
    keys = sorted({(r["codec"], r["qp"]) for r in rows}, key=lambda k: (PULSE_MODELS.index(k[0]), k[1]))
    for codec, qp in keys:
        subset = [r for r in rows if r["codec"] == codec and r["qp"] == qp]
        entry = {"codec": codec, "qp": qp, "images": len(subset)}
        for key in ("bpp", *METRIC_NAMES, "encode_ms", "decode_ms"):
            entry[key] = float(np.mean([r[key] for r in subset]))
        summary.append(entry)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out).resolve()
    models = args.models.split(",")
    unknown = [m for m in models if m not in PULSE_MODELS]
    if unknown:
        raise SystemExit(f"unknown PULSE models: {unknown}")

    images_root = Path(args.images).resolve()
    # A test on other images must not overwrite the Kodak files that the plots read.
    if images_root != DEFAULT_IMAGES.resolve() and args.tag == DEFAULT_TAG:
        raise SystemExit(f"--images is not the Kodak set; give a different --tag than {DEFAULT_TAG}")
    # Same file selection as PULSE codec/images.py image_paths (recursive, sorted).
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(p.resolve() for p in images_root.rglob("*") if p.suffix.lower() in image_suffixes)
    if not image_paths:
        raise SystemExit(f"No images found in {images_root}")

    # Step 1 for every model first; no metric network is on the GPU meanwhile.
    result_files = {m: run_pulse_coding(args, m, out_root / m, image_paths) for m in models}

    # Step 2 with one set of metric networks for all models.
    device = torch.device(args.device)
    metrics = build_metrics(device)
    rows = []
    for model in models:
        model_rows = score_model(model, result_files[model], out_root / model, images_root, metrics, device)
        rows.extend(model_rows)
        for entry in summarize(model_rows):
            print(
                f"{model:>18} qp{entry['qp']}: bpp {entry['bpp']:.4f} | PSNR {entry['psnr']:.3f} | "
                f"LPIPS {entry['lpips']:.4f} | DISTS {entry['dists']:.4f}",
                flush=True,
            )

    # Every (model, QP, image) must be present, otherwise the averages would
    # silently cover different image subsets.
    expected = len(models) * len(args.qps.split(",")) * len(image_paths)
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} scored rows, got {len(rows)}; nothing was written")

    metrics_dir = SRC_ROOT / "result" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = metrics_dir / f"{args.tag}.csv"
    summary_csv = metrics_dir / f"{args.tag}_summary.csv"
    write_csv(per_image_csv, rows)
    write_csv(summary_csv, summarize(rows))
    print(f"\nSaved: {per_image_csv}\nSaved: {summary_csv}")


if __name__ == "__main__":
    main()
