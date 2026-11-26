"""Train the scale-hyperprior image codec (Balle et al. 2018) from scratch on DIV2K.

Runs as one Kaggle job through quota-run (2x T4). Each GPU trains the same
architecture for one rate-distortion trade-off (lambda), so one job produces
two operating points of the rate-distortion curve.

Model: CompressAI `ScaleHyperprior` with N=128, M=192, the configuration of
the CompressAI model zoo qualities 1-5 (`bmshj2018-hyperprior`), so the same
model trained fully by the CompressAI authors serves as a reference.

Loss (standard CompressAI recipe):
    L = lambda * 255^2 * MSE(x, x_hat) + bpp(likelihoods)
The bpp term is the estimated rate of the latents y and hyper-latents z.
A separate "aux" loss fits the quantiles of the entropy bottleneck, which the
real entropy coder needs.

Data: DIV2K train HR (800 images), kept in RAM as uint8 arrays, at full
resolution and at half resolution (half of the crops come from each, so the
model also sees the coarser detail scale of 768x512 test photos). Random
256x256 crops with horizontal flips. Kodak is never used here.

Outputs (QUOTA_RUN_OUTPUT_DIR, downloaded to the PC after the job):
    hyperprior_lmbda<value>.pth   model weights (saved every --save-minutes and at the end)
    train_log.csv                 loss, mse, bpp per lambda over time
    run_info.json                 settings, speed, GPU, versions, round-trip check
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def install_compressai() -> None:
    """Install CompressAI without touching the preinstalled torch/numpy of the Kaggle image."""
    try:
        import compressai  # noqa: F401
    except ImportError:
        pip = [sys.executable, "-m", "pip", "install", "-q"]
        # CompressAI itself without dependencies: its pins (numpy<2, a matching torch)
        # would replace the image's working torch/numpy.
        subprocess.run([*pip, "--no-deps", "compressai==1.2.8"], check=True)
        # Imported by compressai at load time (torch_geometric via its point-cloud
        # registry, which also needs xxhash); these do not pin torch or numpy.
        subprocess.run([*pip, "torch-geometric==2.8.1", "xxhash", "pytorch-msssim", "einops"], check=True)


install_compressai()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from compressai.zoo import bmshj2018_hyperprior  # noqa: E402
from PIL import Image  # noqa: E402

OUTPUT_DIR = Path(os.environ.get("QUOTA_RUN_OUTPUT_DIR", "output"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lambdas", default="0.0018,0.0035", help="one lambda per GPU")
    parser.add_argument("--train-hours", type=float, default=9.0, help="training time budget per run")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-drop-at", type=float, default=0.8, help="fraction of the time budget where lr drops 10x")
    parser.add_argument("--max-images", type=int, default=0, help="limit DIV2K images (probe runs)")
    parser.add_argument("--save-minutes", type=float, default=30.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="/kaggle/input", help="folder searched for DIV2K train_HR PNGs")
    return parser.parse_args()


# ---------------------------------------------------------------- data ----

def find_div2k_train(data_dir: str) -> list[Path]:
    """Locate the DIV2K train HR PNGs of the attached Kaggle dataset."""
    paths = sorted(p for p in Path(data_dir).rglob("*.png") if "train_HR" in str(p))
    if not paths:
        raise SystemExit(f"DIV2K train HR images not found under {data_dir}")
    return paths


def load_image_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the image at full and at half resolution, both uint8 HxWx3."""
    image = Image.open(path).convert("RGB")
    half = image.resize((image.width // 2, image.height // 2), Image.Resampling.LANCZOS)
    return np.asarray(image), np.asarray(half)


class RandomCropSampler:
    """Draws batches of random crops from images held in RAM (no disk IO during training)."""

    def __init__(self, images: list[np.ndarray], crop: int, seed: int):
        self.images = [im for im in images if im.shape[0] >= crop and im.shape[1] >= crop]
        self.crop = crop
        self.rng = np.random.default_rng(seed)

    def batch(self, size: int) -> torch.Tensor:
        crops = np.empty((size, self.crop, self.crop, 3), dtype=np.uint8)
        for i in range(size):
            image = self.images[self.rng.integers(len(self.images))]
            top = self.rng.integers(image.shape[0] - self.crop + 1)
            left = self.rng.integers(image.shape[1] - self.crop + 1)
            patch = image[top : top + self.crop, left : left + self.crop]
            crops[i] = patch[:, ::-1] if self.rng.random() < 0.5 else patch
        # NHWC uint8 -> NCHW float in [0, 1]; pinned memory for a fast copy to the GPUs.
        return torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous().pin_memory()


# --------------------------------------------------------------- model ----

class Run:
    """One lambda: its model, optimizers and running statistics, on its own GPU."""

    def __init__(self, lmbda: float, device: torch.device, lr: float):
        self.lmbda = lmbda
        self.device = device
        # quality=1 only selects N=128, M=192; the weights start random (pretrained=False).
        self.model = bmshj2018_hyperprior(quality=1, metric="mse", pretrained=False).to(device).train()
        main_params = [p for n, p in self.model.named_parameters() if not n.endswith(".quantiles")]
        aux_params = [p for n, p in self.model.named_parameters() if n.endswith(".quantiles")]
        self.optimizer = torch.optim.Adam(main_params, lr=lr)
        self.aux_optimizer = torch.optim.Adam(aux_params, lr=1e-3)
        self.steps = 0
        self.window = {"loss": 0.0, "mse": 0.0, "bpp": 0.0, "n": 0}

    def step(self, batch_uint8: torch.Tensor) -> None:
        x = batch_uint8.to(self.device, non_blocking=True).float().div_(255.0)
        out = self.model(x)
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        bpp = sum(torch.log(l).sum() for l in out["likelihoods"].values()) / (-np.log(2) * num_pixels)
        mse = F.mse_loss(out["x_hat"], x)
        loss = self.lmbda * 255.0**2 * mse + bpp

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.aux_optimizer.zero_grad(set_to_none=True)
        self.model.aux_loss().backward()
        self.aux_optimizer.step()

        self.steps += 1
        # Detached sums; read (and synchronized) only when logging.
        self.window["loss"] += loss.detach()
        self.window["mse"] += mse.detach()
        self.window["bpp"] += bpp.detach()
        self.window["n"] += 1

    def drop_lr(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] *= 0.1

    def read_window(self) -> dict:
        n = max(self.window["n"], 1)
        values = {k: float(self.window[k]) / n for k in ("loss", "mse", "bpp")}
        values["psnr"] = -10 * np.log10(max(values["mse"], 1e-12))
        self.window = {"loss": 0.0, "mse": 0.0, "bpp": 0.0, "n": 0}
        return values

    def checkpoint_path(self) -> Path:
        return OUTPUT_DIR / f"hyperprior_lmbda{self.lmbda}.pth"

    def save(self) -> None:
        torch.save(self.model.state_dict(), self.checkpoint_path())

    @torch.no_grad()
    def round_trip(self, image_uint8: np.ndarray) -> dict:
        """Real entropy coding check: compress to bytes, decompress, compare."""
        self.model.eval()
        self.model.update(force=True)
        x = torch.from_numpy(image_uint8).permute(2, 0, 1)[None].float().div(255).to(self.device)
        encoded = self.model.compress(x)
        decoded = self.model.decompress(encoded["strings"], encoded["shape"])["x_hat"].clamp(0, 1)
        n_bytes = sum(len(s[0]) for s in encoded["strings"])
        mse = float(F.mse_loss(decoded, x))
        self.model.train()
        return {"bytes": n_bytes, "bpp": 8 * n_bytes / (x.shape[2] * x.shape[3]), "psnr": -10 * np.log10(mse)}


# ---------------------------------------------------------------- main ----

def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    lambdas = [float(v) for v in args.lambdas.split(",")]
    gpu_count = torch.cuda.device_count()
    if gpu_count < len(lambdas):
        raise SystemExit(f"{len(lambdas)} lambdas need {len(lambdas)} GPUs, found {gpu_count}")

    # Load DIV2K once into RAM, decoding PNGs in parallel threads (PIL releases the GIL).
    paths = find_div2k_train(args.data_dir)
    if args.max_images:
        paths = paths[: args.max_images]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        pairs = list(pool.map(load_image_pair, paths))
    images = [im for pair in pairs for im in pair]
    load_seconds = time.time() - t0
    print(f"loaded {len(paths)} DIV2K images (+ half-size copies) in {load_seconds:.0f} s", flush=True)
    sampler = RandomCropSampler(images, args.crop, args.seed)

    runs = [Run(l, torch.device(f"cuda:{i}"), args.lr) for i, l in enumerate(lambdas)]
    params = sum(p.numel() for p in runs[0].model.parameters())

    log_path = OUTPUT_DIR / "train_log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["elapsed_min", "step", "lambda", "loss", "mse", "psnr", "bpp", "lr"])

    budget_s = args.train_hours * 3600
    start = last_save = time.time()
    lr_dropped = False
    step = 0
    while time.time() - start < budget_s:
        batch = sampler.batch(args.batch)
        # Both GPUs work at the same time: kernel launches are asynchronous.
        for run in runs:
            run.step(batch)
        step += 1

        elapsed = time.time() - start
        if not lr_dropped and elapsed > args.lr_drop_at * budget_s:
            for run in runs:
                run.drop_lr()
            lr_dropped = True
            print(f"step {step}: learning rate dropped 10x", flush=True)

        if step % args.log_every == 0:
            with log_path.open("a", newline="") as f:
                writer = csv.writer(f)
                for run in runs:
                    w = run.read_window()
                    lr = run.optimizer.param_groups[0]["lr"]
                    writer.writerow([f"{elapsed / 60:.1f}", step, run.lmbda, w["loss"], w["mse"], w["psnr"], w["bpp"], lr])
                    print(
                        f"{elapsed / 60:6.1f} min step {step:7d} lambda {run.lmbda}: "
                        f"loss {w['loss']:.4f} psnr {w['psnr']:.2f} bpp {w['bpp']:.4f} "
                        f"({step / elapsed:.2f} steps/s)",
                        flush=True,
                    )

        if time.time() - last_save > args.save_minutes * 60:
            for run in runs:
                run.save()
            last_save = time.time()

    total_s = time.time() - start
    info = {
        "lambdas": lambdas,
        "steps": step,
        "train_seconds": total_s,
        "steps_per_second": step / total_s,
        "batch": args.batch,
        "crop": args.crop,
        "images": len(paths),
        "image_load_seconds": load_seconds,
        "parameters": params,
        "gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        "peak_memory_gb": [torch.cuda.max_memory_allocated(i) / 1e9 for i in range(gpu_count)],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "args": vars(args),
        # Real bitstream on a 512x768 crop of one training image, as a pipeline check.
        "round_trip": {},
    }
    check_image = images[0][:512, :768]
    for run in runs:
        run.save()
        info["round_trip"][str(run.lmbda)] = run.round_trip(check_image)
    (OUTPUT_DIR / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
