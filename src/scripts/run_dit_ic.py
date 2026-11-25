"""Run DiT-IC inference on a folder of images without modifying DiT-IC.

The official repository is imported read-only; every artifact is written under
``src/result``.  Example (from the project root)::

    .venv\\Scripts\\python.exe src/scripts/run_dit_ic.py --limit 1

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Keep DiT-IC clean: no __pycache__ inside the submodule.
sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DIT_IC_ROOT = PROJECT_ROOT / "DiT-IC"
CACHE_ROOT = SRC_ROOT / "models" / "cache"

# Third-party caches must not leak into the repository or the user profile.
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

# Make the official package importable (models.DiT_IC, eval.compress_utils, ELIC).
sys.path.insert(0, str(DIT_IC_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from eval.compress_utils import filesize, read_body, write_body  # noqa: E402
from models.DiT_IC import Codec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import build_metrics, evaluate_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiT-IC Kodak inference runner")
    parser.add_argument(
        "--config",
        default=str(SRC_ROOT / "models" / "configs" / "inference_q3.yaml"),
        help="YAML config with model parameters",
    )
    parser.add_argument(
        "--images",
        default=str(SRC_ROOT / "images" / "kodak"),
        help="Folder with PNG/JPEG test images",
    )
    parser.add_argument(
        "--out",
        default=str(SRC_ROOT / "result" / "kodak" / "q3"),
        help="Output folder (bin/, rec/, metrics)",
    )
    parser.add_argument("--checkpoint", default=None, help="Override codec_path")
    parser.add_argument("--limit", type=int, default=0, help="Only first N images")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=903)
    parser.add_argument("--no-rec", action="store_true", help="Do not save reconstructions")
    parser.add_argument("--tag", default="dit_ic_q3")
    return parser.parse_args()


def load_config(path: Path, checkpoint: str | None) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    params = config["model"]["params"]
    if checkpoint:
        params["codec_path"] = checkpoint
    for key in ("dit_path", "codec_path", "elic_path"):
        value = params.get(key)
        if value and not Path(value).is_absolute():
            params[key] = str(PROJECT_ROOT / value)
    return config


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.out)
    bin_dir = out_dir / "bin"
    rec_dir = out_dir / "rec"
    bin_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_rec:
        rec_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config), args.checkpoint)
    params = config["model"]["params"]

    print(f"[env] project root : {PROJECT_ROOT}")
    print(f"[env] python       : {sys.version.split()[0]}")
    print(f"[env] torch        : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"[env] device       : {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")
    print(f"[model] checkpoint : {params['codec_path']}")
    print(f"[model] sana       : {params['dit_path']}")

    t0 = time.perf_counter()
    net = Codec(device=str(device), model_type=config["model"]["model_type"], **params)
    net.to(device).eval()
    net.codec.update(force=True)
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[model] loaded in {load_s:.1f}s, parameters: {n_params / 1e9:.3f}B")

    metrics = build_metrics(device)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    image_paths = sorted(
        p for ext in ("*.png", "*.jpg", "*.jpeg") for p in Path(args.images).glob(ext)
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")
    print(f"[data] {len(image_paths)} images from {args.images}")

    rows = []
    for idx, image_path in enumerate(image_paths, 1):
        name = image_path.stem
        gt_pil = Image.open(image_path).convert("RGB")
        gt = transform(gt_pil).unsqueeze(0).to(device)
        ori_h, ori_w = gt.shape[-2:]
        pad_h = (-ori_h) % 256
        pad_w = (-ori_w) % 256
        padded = F.pad(gt, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else gt

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_enc = time.perf_counter()
        with torch.no_grad():
            output = net.compress(padded)
        if device.type == "cuda":
            torch.cuda.synchronize()
        enc_ms = (time.perf_counter() - t_enc) * 1000

        bin_path = bin_dir / f"{name}.bin"
        with bin_path.open("wb") as f:
            write_body(f, output["shape"], output["strings"])
        n_bytes = filesize(str(bin_path))
        bpp = n_bytes * 8 / (ori_h * ori_w)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_dec = time.perf_counter()
        with bin_path.open("rb") as f:
            strings, shape = read_body(f)
        with torch.no_grad():
            rec = net.decompress(strings, shape)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dec_ms = (time.perf_counter() - t_dec) * 1000

        rec = rec[:, :, :ori_h, :ori_w]
        rec = (rec * 0.5 + 0.5).clamp(0.0, 1.0).float()
        gt01 = (gt * 0.5 + 0.5).clamp(0.0, 1.0).float()

        scores = evaluate_pair(metrics, rec, gt01)
        row = {
            "image": name,
            "width": ori_w,
            "height": ori_h,
            "bpp": bpp,
            "bytes": n_bytes,
            "encode_ms": enc_ms,
            "decode_ms": dec_ms,
            **scores,
        }
        rows.append(row)

        if not args.no_rec:
            transforms.ToPILImage()(rec[0].cpu()).save(rec_dir / f"{name}.png")

        print(
            f"[{idx:02d}/{len(image_paths):02d}] {name}: "
            f"{bpp:.4f} bpp | PSNR {scores['psnr']:.3f} | LPIPS {scores['lpips']:.4f} | "
            f"DISTS {scores['dists']:.4f} | enc {enc_ms:.0f} ms"
        )

    summary = {
        "tag": args.tag,
        "checkpoint": params["codec_path"],
        "images": len(rows),
        "parameters_b": n_params / 1e9,
        "load_time_s": load_s,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
        ),
    }
    for key in ("bpp", "psnr", "ms_ssim", "lpips", "dists", "clipiqa", "musiq", "encode_ms", "decode_ms", "bytes"):
        summary[f"avg_{key}"] = float(np.mean([r[key] for r in rows]))
    summary["median_encode_ms"] = float(np.median([r["encode_ms"] for r in rows]))
    summary["median_decode_ms"] = float(np.median([r["decode_ms"] for r in rows]))

    metrics_dir = SRC_ROOT / "result" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / f"{args.tag}.csv"
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = metrics_dir / f"{args.tag}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Average ===")
    for key, value in summary.items():
        if key.startswith("avg_"):
            print(f"{key[4:]:>10}: {value:.5f}")
    print(f"median encode: {summary['median_encode_ms']:.0f} ms | median decode: {summary['median_decode_ms']:.0f} ms")
    print(f"\nSaved: {csv_path}\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
