"""Generate the experiment notebook (executed separately with nbconvert)."""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "experiments" / "DiT-IC.ipynb"

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(
    """
# DiT-IC: запуск, тестирование и сравнение с JPEG / WebP

Ноутбук демонстрирует полный цикл работы с моделью **DiT-IC** (CVPR 2026):

1. проверка окружения и структуры проекта;
2. загрузка официального чекпоинта `q3_merge_ema.pt`;
3. сжатие и восстановление одного изображения;
4. прогон на датасете **Kodak** (24 изображения);
5. сравнение с классическими кодеками **JPEG** и **WebP** при одинаковом битрейте;
6. rate-distortion кривые и визуальное сравнение.

Исходники `DiT-IC/` не изменяются: все артефакты пишутся в `src/result`.
"""
)

code(
    """
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "DiT-IC").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC = PROJECT_ROOT / "src"
METRICS = SRC / "result" / "metrics"
FIGURES = SRC / "result" / "figures"
PY = sys.executable

print("project:", PROJECT_ROOT)
print("python :", sys.version.split()[0])
print("torch  :", torch.__version__, "| CUDA:", torch.version.cuda)
print("gpu    :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
"""
)

md(
    """
## 1. Структура проекта

Все результаты экспериментов находятся в `src/result`, веса и конфиги — в `src/models`.
"""
)

code(
    """
for folder in ["models/checkpoints", "models/sana", "images/kodak", "result", "experiments", "scripts"]:
    path = SRC / folder
    n_files = sum(1 for p in path.rglob("*") if p.is_file()) if path.exists() else 0
    print(f"{folder:22s} {n_files:4d} files")

ckpt = SRC / "models" / "checkpoints" / "q3_merge_ema.pt"
print(f"\\ncheckpoint: {ckpt.name}, {ckpt.stat().st_size / 1e9:.2f} GB")
"""
)

md(
    """
## 2. Запуск DiT-IC на одном изображении

Используется merged-чекпоинт (`use_merge=True`): LoRA-веса уже влиты в базовую модель,
поэтому загрузка и инференс быстрее. Скрипт вызывает штатные `Codec.compress` / `Codec.decompress`
из `DiT-IC`, сохраняет реальный битстрим и считает BPP по его размеру.
"""
)

code(
    """
cmd = [PY, str(SRC / "scripts" / "run_dit_ic.py"), "--limit", "2", "--no-rec"]
result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
print(result.stdout[-4000:])
if result.returncode != 0:
    print(result.stderr[-4000:])
result.check_returncode()
"""
)

code(
    """
df_one = pd.read_csv(METRICS / "dit_ic_q3.csv")
df_one[["image", "bpp", "psnr", "ms_ssim", "lpips", "dists", "encode_ms", "decode_ms"]].round(4)
"""
)

md(
    """
## 3. Полный прогон на Kodak

Kodak — 24 изображения 768×512, стандартный benchmark для neural compression.
Для каждого изображения сохраняются битстрим и реконструкция; метрики усредняются.
"""
)

code(
    """
cmd = [PY, str(SRC / "scripts" / "run_dit_ic.py")]
result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
print(result.stdout[-5000:])
if result.returncode != 0:
    print(result.stderr[-5000:])
result.check_returncode()
"""
)

code(
    """
df_dit = pd.read_csv(METRICS / "dit_ic_q3.csv")
summary = json.loads((METRICS / "dit_ic_q3_summary.json").read_text())

avg = df_dit[["bpp", "psnr", "ms_ssim", "lpips", "dists", "clipiqa", "musiq", "encode_ms", "decode_ms"]].mean()
print(f"images: {summary['images']} | parameters: {summary['parameters_b']:.3f}B | peak VRAM: {summary['peak_vram_gb']:.2f} GB")
print(f"median encode: {summary['median_encode_ms']:.0f} ms | median decode: {summary['median_decode_ms']:.0f} ms")
display(avg.to_frame("average").round(4))
df_dit[["image", "bpp", "psnr", "lpips", "dists"]].round(4).head(10)
"""
)

md(
    """
## 4. Пример реконструкции

Ниже — оригинал, результат DiT-IC и карта абсолютной разницы для одного изображения.
"""
)

code(
    """
import numpy as np

name = "kodim01"
orig = Image.open(SRC / "images" / "kodak" / f"{name}.png").convert("RGB")
rec = Image.open(SRC / "result" / "kodak" / "q3" / "rec" / f"{name}.png").convert("RGB")
diff = np.abs(np.asarray(orig, dtype=np.int16) - np.asarray(rec, dtype=np.int16)).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, image, title in zip(
    axes,
    [orig, rec, diff],
    ["Оригинал", f"DiT-IC q3 ({summary['avg_bpp']:.4f} bpp)", "|разница|"],
):
    ax.imshow(image, cmap="magma" if title.startswith("|") else None)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
plt.show()
"""
)

md(
    """
## 5. Классические кодеки: JPEG и WebP

JPEG и WebP не умеют работать на ~0.08 BPP без потери разрешения, поэтому для них
используется сетка **(масштаб, качество)**: изображение уменьшается перед кодированием
и возвращается к исходному размеру после декодирования. BPP всегда считается по
исходному числу пикселей. Метрики — те же, что и для DiT-IC.
"""
)

code(
    """
cmd = [PY, str(SRC / "scripts" / "run_baselines.py")]
result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
print(result.stdout[-5000:])
if result.returncode != 0:
    print(result.stderr[-5000:])
result.check_returncode()
"""
)

code(
    """
df_base = pd.read_csv(METRICS / "baselines_kodak_summary.csv")
table = (
    df_base.groupby(["codec", "scale"])
    .agg(bpp_min=("bpp", "min"), bpp_max=("bpp", "max"),
         psnr_min=("psnr", "min"), psnr_max=("psnr", "max"),
         lpips_min=("lpips", "min"), lpips_max=("lpips", "max"))
    .round(4)
)
display(table)
df_base[["codec", "scale", "quality", "bpp", "psnr", "ms_ssim", "lpips", "dists"]].round(4).head(12)
"""
)

md(
    """
## 6. Rate-distortion кривые и сравнение при одинаковом BPP

Точка DiT-IC сравнивается с интерполированными значениями JPEG/WebP при том же BPP.
"""
)

code(
    """
cmd = [PY, str(SRC / "scripts" / "make_plots.py")]
result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr[-3000:])
result.check_returncode()

for name in ["rd_psnr", "rd_lpips", "rd_dists"]:
    display(Image.open(FIGURES / f"{name}.png"))
"""
)

code(
    """
matched = pd.read_csv(METRICS / "matched_bpp.csv")
display(Image.open(FIGURES / "matched_bpp_bars.png"))
matched.round(4)
"""
)

md("## 7. Визуальное сравнение\n\nФрагмент изображения при близком битрейте: DiT-IC, JPEG и WebP.")

code(
    """
name = "kodim01"
orig = Image.open(SRC / "images" / "kodak" / f"{name}.png").convert("RGB")
dit_target = float(df_dit.loc[df_dit.image == name, "bpp"].iloc[0])

def nearest_row(codec):
    sub = df_base[df_base.codec == codec]
    return sub.iloc[(sub.bpp - dit_target).abs().argsort()[:1]].iloc[0]

def load_baseline(codec, row):
    ext = "jpg" if codec == "jpeg" else "webp"
    path = (
        SRC / "result" / "baselines" / "encoded" / codec
        / f"s{float(row.scale):g}" / f"q{int(row.quality):02d}" / f"{name}.{ext}"
    )
    image = Image.open(path).convert("RGB")
    if image.size != orig.size:
        image = image.resize(orig.size, Image.Resampling.LANCZOS)
    return image

row_jpeg = nearest_row("jpeg")
row_webp = nearest_row("webp")
images = {
    "Original": orig,
    f"DiT-IC q3 ({dit_target:.4f} bpp)": Image.open(SRC / "result" / "kodak" / "q3" / "rec" / f"{name}.png").convert("RGB"),
    f"JPEG s={float(row_jpeg.scale):g}, q{int(row_jpeg.quality)} ({float(row_jpeg.bpp):.4f} bpp)": load_baseline("jpeg", row_jpeg),
    f"WebP s={float(row_webp.scale):g}, q{int(row_webp.quality)} ({float(row_webp.bpp):.4f} bpp)": load_baseline("webp", row_webp),
}

w, h = orig.size
crop = (w // 3, h // 4, w // 3 + 320, h // 4 + 240)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
for ax, (title, img) in zip(axes, images.items()):
    ax.imshow(img.crop(crop))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.tight_layout()
plt.show()
"""
)

md("## 8. Выводы")

code(
    """
r = {row["codec"]: row for row in matched.to_dict("records")}
dit = r["dit_ic_q3"]

print(f"DiT-IC q3: {dit['bpp']:.4f} bpp | PSNR {dit['psnr']:.2f} dB | LPIPS {dit['lpips']:.4f} | DISTS {dit['dists']:.4f}")
for codec in ("jpeg", "webp"):
    row = r[codec]
    print(
        f"{codec.upper():4s} @ {row['bpp']:.4f} bpp | "
        f"PSNR {row['psnr']:.2f} dB ({row['psnr'] - dit['psnr']:+.2f}) | "
        f"LPIPS {row['lpips']:.4f} ({row['lpips'] - dit['lpips']:+.4f}) | "
        f"DISTS {row['dists']:.4f} ({row['dists'] - dit['dists']:+.4f})"
    )

print("\\nВремя на изображение (медиана, после прогрева):")
print(f"encode: {summary['median_encode_ms']:.0f} ms | decode: {summary['median_decode_ms']:.0f} ms")
"""
)

md(
    """
**Интерпретация**

- DiT-IC работает в режиме экстремально низкого битрейта (~0.08 BPP), где классические JPEG/WebP
  уже теряют структуру изображения.
- По PSNR классические кодеки при том же BPP могут быть конкурентны, но перцептивные метрики
  (LPIPS, DISTS) и визуальное качество у DiT-IC заметно лучше — это и есть цель diffusion-based codec.
- Основная цена — скорость: один шаг DiT на GPU против миллисекунд у JPEG/WebP.
"""
)

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "DiT-IC (.venv)",
    "language": "python",
    "name": "ditic",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"written {OUT}")
