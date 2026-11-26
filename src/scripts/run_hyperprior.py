"""Evaluate the scale-hyperprior codec on Kodak: our trained weights and the official ones.

Two codecs with the same architecture (CompressAI bmshj2018-hyperprior, N=128,
M=192) are evaluated:
    hyperprior-ours  weights trained by us on DIV2K (train_hyperprior_kaggle.py)
    hyperprior-zoo   official CompressAI weights, qualities 1-4, which were
                     trained with the same lambdas (0.0018, 0.0035, 0.0067,
                     0.0130) for much longer; the reference for our training

Every image goes through the real entropy coder (model.compress -> bytes ->
model.decompress). bpp = 8 * bytes / (width * height) of the original image;
bytes are the coded y and z strings plus 4 bytes for the latent shape, which a
file would also need. Reconstructions are rounded to uint8 and scored with
metrics_utils.py, exactly like JPEG/WebP and PULSE.

Outputs:
    src/result/metrics/<tag>.csv           one row per image and model
    src/result/metrics/<tag>_summary.csv   one row per model (means)
    src/result/hyperprior/rec/<codec>/<setting>/*.png   reconstructions (git-ignored)
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

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from compressai.models import ScaleHyperprior  # noqa: E402
from compressai.zoo import bmshj2018_hyperprior  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms import ToTensor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import build_metrics, evaluate_pair  # noqa: E402

METRIC_NAMES = ("psnr", "ms_ssim", "lpips", "dists", "clipiqa", "musiq")
# CompressAI zoo: quality -> training lambda of bmshj2018-hyperprior (MSE).
ZOO_LAMBDAS = {1: 0.0018, 2: 0.0035, 3: 0.0067, 4: 0.0130}
SHAPE_HEADER_BYTES = 4  # two uint16 values: latent height and width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale hyperprior on Kodak")
    parser.add_argument("--images", default=str(SRC_ROOT / "images" / "kodak"))
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="our .pth files (default: src/result/hyperprior/kaggle/night-*/hyperprior_lmbda*.pth)",
    )
    parser.add_argument("--zoo-qualities", default="1,2,3,4", help="empty string skips the zoo models")
    parser.add_argument("--tag", default="hyperprior_kodak")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-rec", action="store_true", help="do not save reconstructions")
    return parser.parse_args()


def load_models(args: argparse.Namespace, device: torch.device) -> list[dict]:
    """Return the models to evaluate, each with its codec name, setting and lambda."""
    models = []
    checkpoints = args.checkpoints
    if checkpoints is None:
        checkpoints = sorted(str(p) for p in (SRC_ROOT / "result" / "hyperprior" / "kaggle").glob("night-*/hyperprior_lmbda*.pth"))
    for path in checkpoints:
        path = Path(path)
        lmbda = float(path.stem.replace("hyperprior_lmbda", ""))
        state = torch.load(path, map_location="cpu")
        # from_state_dict reads N and M from the weights and resizes the entropy-coder buffers.
        model = ScaleHyperprior.from_state_dict(state)
        info_path = path.parent / "run_info.json"
        steps = json.loads(info_path.read_text())["steps"] if info_path.exists() else None
        models.append({"codec": "hyperprior-ours", "setting": f"lambda {lmbda}", "lmbda": lmbda,
                       "train_steps": steps, "model": model})
    for quality in [int(q) for q in args.zoo_qualities.split(",") if q]:
        model = bmshj2018_hyperprior(quality=quality, metric="mse", pretrained=True)
        models.append({"codec": "hyperprior-zoo", "setting": f"quality {quality}", "lmbda": ZOO_LAMBDAS[quality],
                       "train_steps": None, "model": model})
    for entry in models:
        entry["model"] = entry["model"].to(device).eval()
        # Builds the integer CDF tables that the real entropy coder uses.
        entry["model"].update(force=True)
    return models


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def code_image(model, x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int, float, float]:
    """Compress and decompress one image; returns the uint8-rounded reconstruction, bytes and times."""
    height, width = x.shape[-2:]
    # The hyperprior downsamples by 64 in total, so pad to a multiple of 64.
    pad_h, pad_w = (-height) % 64, (-width) % 64
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    synchronize(device)
    t0 = time.perf_counter()
    encoded = model.compress(x_padded)
    synchronize(device)
    encode_ms = (time.perf_counter() - t0) * 1000
    n_bytes = sum(len(s[0]) for s in encoded["strings"]) + SHAPE_HEADER_BYTES

    t0 = time.perf_counter()
    x_hat = model.decompress(encoded["strings"], encoded["shape"])["x_hat"]
    synchronize(device)
    decode_ms = (time.perf_counter() - t0) * 1000

    x_hat = x_hat[..., :height, :width].clamp(0, 1)
    # Round to 8-bit pixels: what a receiver would display or save.
    x_hat = (x_hat * 255).round() / 255
    return x_hat, n_bytes, encode_ms, decode_ms


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    image_paths = sorted(Path(args.images).glob("*.png"))
    if not image_paths:
        raise SystemExit(f"No images in {args.images}")

    models = load_models(args, device)
    metrics = build_metrics(device)
    totensor = ToTensor()
    rows = []
    for entry in models:
        rec_dir = SRC_ROOT / "result" / "hyperprior" / "rec" / entry["codec"] / entry["setting"].replace(" ", "_")
        for path in image_paths:
            gt_pil = Image.open(path).convert("RGB")
            width, height = gt_pil.size
            gt = totensor(gt_pil).unsqueeze(0).to(device)
            rec, n_bytes, encode_ms, decode_ms = code_image(entry["model"], gt, device)
            scores = evaluate_pair(metrics, rec, gt)
            if not args.no_rec:
                rec_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray((rec[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(
                    rec_dir / f"{path.stem}.png"
                )
            rows.append({
                "codec": entry["codec"], "setting": entry["setting"], "lmbda": entry["lmbda"],
                "train_steps": entry["train_steps"], "image": path.stem, "width": width, "height": height,
                "bpp": n_bytes * 8 / (width * height), "bytes": n_bytes,
                "encode_ms": encode_ms, "decode_ms": decode_ms, **scores,
            })
        subset = [r for r in rows if r["codec"] == entry["codec"] and r["setting"] == entry["setting"]]
        print(f"{entry['codec']:>16} {entry['setting']:>14}: bpp {np.mean([r['bpp'] for r in subset]):.4f} | "
              f"PSNR {np.mean([r['psnr'] for r in subset]):.3f} | LPIPS {np.mean([r['lpips'] for r in subset]):.4f}",
              flush=True)

    summary = []
    for entry in models:
        subset = [r for r in rows if r["codec"] == entry["codec"] and r["setting"] == entry["setting"]]
        item = {"codec": entry["codec"], "setting": entry["setting"], "lmbda": entry["lmbda"],
                "train_steps": entry["train_steps"], "images": len(subset)}
        for key in ("bpp", *METRIC_NAMES, "encode_ms", "decode_ms"):
            item[key] = float(np.mean([r[key] for r in subset]))
        summary.append(item)

    metrics_dir = SRC_ROOT / "result" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for path, data in ((metrics_dir / f"{args.tag}.csv", rows), (metrics_dir / f"{args.tag}_summary.csv", summary)):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
