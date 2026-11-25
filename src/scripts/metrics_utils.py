"""Shared metric helpers for DiT-IC experiments."""

from __future__ import annotations

import torch


def build_metrics(device: torch.device):
    import pyiqa
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    metrics = {
        "psnr": pyiqa.create_metric("psnr").to(device).eval(),
        "ms_ssim": pyiqa.create_metric("ms_ssim").to(device).eval(),
        "dists": pyiqa.create_metric("dists").to(device).eval(),
        "clipiqa": pyiqa.create_metric("clipiqa").to(device).eval(),
        "musiq": pyiqa.create_metric("musiq").to(device).eval(),
        "lpips": LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        .to(device)
        .eval(),
    }
    return metrics


@torch.no_grad()
def evaluate_pair(metrics: dict, rec: torch.Tensor, gt: torch.Tensor) -> dict:
    """Compute all metrics; both tensors must be RGB in [0, 1] with shape [1,3,H,W]."""
    return {name: float(metric(rec, gt).item()) for name, metric in metrics.items()}
