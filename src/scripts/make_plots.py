"""Build rate-distortion plots and a matched-BPP table for DiT-IC vs JPEG/WebP.

When src/result/metrics/pulse_kodak_summary.csv exists (src/scripts/run_pulse.py),
the four PULSE models are added to the rate-distortion plots as QP curves.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
CACHE_ROOT = SRC_ROOT / "models" / "cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METRICS_DIR = SRC_ROOT / "result" / "metrics"
FIGURES_DIR = SRC_ROOT / "result" / "figures"

METRIC_LABELS = {
    "psnr": ("PSNR (dB)", True),
    "ms_ssim": ("MS-SSIM", True),
    "lpips": ("LPIPS", False),
    "dists": ("DISTS", False),
    "clipiqa": ("CLIP-IQA", True),
    "musiq": ("MUSIQ", True),
}

STYLES = {
    "jpeg": dict(color="#d95f02", marker="o", label="JPEG"),
    "webp": dict(color="#1b9e77", marker="s", label="WebP"),
}

# PULSE bundles from src/scripts/run_pulse.py; dashed lines, one point per QP 0..7.
PULSE_STYLES = {
    "pulse-s-mse": dict(color="#1f78b4", marker="v", label="PULSE-S MSE"),
    "pulse-mse": dict(color="#33a02c", marker="^", label="PULSE MSE"),
    "pulse-perceptual": dict(color="#e7298a", marker="D", label="PULSE perceptual"),
    "pulse-l-perceptual": dict(color="#6a3d9a", marker="P", label="PULSE-L perceptual"),
}
PULSE_SUMMARY = METRICS_DIR / "pulse_kodak_summary.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pareto_curve(rows: list[dict], metric: str, higher_is_better: bool) -> tuple[np.ndarray, np.ndarray]:
    """Envelope of the best metric value at each bitrate (monotone frontier)."""
    points = sorted(((float(r["bpp"]), float(r[metric])) for r in rows), key=lambda p: p[0])
    best = None
    frontier = []
    for bpp, value in points:
        if best is None or (value > best if higher_is_better else value < best):
            best = value
            frontier.append((bpp, value))
    return (
        np.array([p[0] for p in frontier]),
        np.array([p[1] for p in frontier]),
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with (METRICS_DIR / "dit_ic_q3_summary.json").open("r", encoding="utf-8") as f:
        dit_summary = json.load(f)
    baseline_rows = read_csv(METRICS_DIR / "baselines_kodak_summary.csv")

    dit_bpp = dit_summary["avg_bpp"]
    dit_metrics = {key[4:]: dit_summary[key] for key in dit_summary if key.startswith("avg_")}

    codec_rows = {
        codec: [r for r in baseline_rows if r["codec"] == codec]
        for codec in ("jpeg", "webp")
    }

    # PULSE is optional: the plots still build when run_pulse.py has not run.
    pulse_rows = {}
    if PULSE_SUMMARY.exists():
        all_pulse = read_csv(PULSE_SUMMARY)
        pulse_rows = {
            model: sorted((r for r in all_pulse if r["codec"] == model), key=lambda r: int(r["qp"]))
            for model in PULSE_STYLES
        }
        pulse_rows = {model: rows for model, rows in pulse_rows.items() if rows}
    pulse_min_bpp = min((float(r["bpp"]) for rows in pulse_rows.values() for r in rows), default=0.0)

    # ---- Rate-distortion plots -------------------------------------------
    for metric, (label, higher_is_better) in METRIC_LABELS.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for codec, rows in codec_rows.items():
            bpp, values = pareto_curve(rows, metric, higher_is_better)
            ax.plot(bpp, values, linewidth=1.8, markersize=4, **STYLES[codec])
        # PULSE has one learned operating point per QP, so its real QP curve is
        # drawn, not a Pareto envelope over many settings like JPEG/WebP.
        for model, rows in pulse_rows.items():
            ax.plot(
                [float(r["bpp"]) for r in rows],
                [float(r[metric]) for r in rows],
                linewidth=1.4,
                linestyle="--",
                markersize=4,
                **PULSE_STYLES[model],
            )
        ax.scatter(
            [dit_bpp],
            [dit_metrics[metric]],
            color="#7570b3",
            marker="*",
            s=220,
            zorder=5,
            label="DiT-IC q3",
        )
        ax.set_xlabel("BPP")
        ax.set_ylabel(label)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7 if pulse_rows else None)
        ax.set_title(f"Kodak: {label} vs bitrate")
        if pulse_rows:
            # The two kinds of curves are built differently; say so on the figure.
            fig.text(
                0.01,
                0.01,
                "JPEG/WebP: best tested (scale, quality) per bitrate for this metric. "
                f"PULSE: measured QP points, lowest rate {pulse_min_bpp:.3f} bpp.",
                fontsize=6,
                color="#555555",
            )
        fig.tight_layout(rect=(0, 0.03, 1, 1) if pulse_rows else None)
        fig.savefig(FIGURES_DIR / f"rd_{metric}.png", dpi=160)
        plt.close(fig)

    # ---- Matched-BPP comparison -----------------------------------------
    matched_rows = []
    for codec, rows in codec_rows.items():
        row = {"codec": codec, "bpp": dit_bpp}
        for metric, (_, higher_is_better) in METRIC_LABELS.items():
            bpp, values = pareto_curve(rows, metric, higher_is_better)
            if dit_bpp < bpp[0] or dit_bpp > bpp[-1]:
                print(
                    f"[warn] DiT-IC BPP {dit_bpp:.4f} is outside the {codec} sweep "
                    f"[{bpp[0]:.4f}, {bpp[-1]:.4f}] for {metric}; values are clamped."
                )
            row[metric] = float(np.interp(dit_bpp, bpp, values))
        matched_rows.append(row)

    dit_row = {"codec": "dit_ic_q3", "bpp": dit_bpp, **{m: dit_metrics[m] for m in METRIC_LABELS}}
    matched_rows.append(dit_row)
    matched_rows.sort(key=lambda r: r["codec"])

    matched_path = METRICS_DIR / "matched_bpp.csv"
    with matched_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(matched_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matched_rows)

    # ---- Bar chart at matched BPP ---------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    bar_metrics = ["psnr", "lpips", "dists"]
    order = ["dit_ic_q3", "jpeg", "webp"]
    colors = {"dit_ic_q3": "#7570b3", "jpeg": "#d95f02", "webp": "#1b9e77"}
    labels = {"dit_ic_q3": "DiT-IC q3", "jpeg": "JPEG", "webp": "WebP"}
    rows_by_codec = {r["codec"]: r for r in matched_rows}
    for ax, metric in zip(axes, bar_metrics):
        values = [rows_by_codec[c][metric] for c in order]
        ax.bar([labels[c] for c in order], values, color=[colors[c] for c in order])
        ax.set_title(METRIC_LABELS[metric][0])
        ax.grid(True, axis="y", alpha=0.3)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Kodak at matched bitrate (BPP = {dit_bpp:.4f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matched_bpp_bars.png", dpi=160)
    plt.close(fig)

    print(f"DiT-IC q3: {dit_bpp:.4f} bpp")
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved {matched_path}")


if __name__ == "__main__":
    main()
