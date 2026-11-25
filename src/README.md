# src — рабочая директория эксперимента DiT-IC

Оригинальный репозиторий `DiT-IC/` не изменяется; здесь лежат веса, данные,
скрипты и результаты.

## Быстрый старт

```powershell
# 1. окружение (из корня проекта)
powershell -ExecutionPolicy Bypass -File src\scripts\setup_env.ps1

# либо вручную:
# uv venv --python 3.12 .venv
# uv pip install --python .venv\Scripts\python.exe torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# uv pip install --python .venv\Scripts\python.exe --no-deps -r src\requirements-lock.txt

# 2. проверка окружения
.venv\Scripts\python.exe src\scripts\env_check.py

# 3. инференс
.venv\Scripts\python.exe src\scripts\run_dit_ic.py --limit 1
.venv\Scripts\python.exe src\scripts\run_dit_ic.py

# 4. baseline JPEG/WebP и графики
.venv\Scripts\python.exe src\scripts\run_baselines.py
.venv\Scripts\python.exe src\scripts\make_plots.py
```

Ноутбук: `src/experiments/DiT-IC.ipynb`.

## Структура

| Путь | Содержимое |
| --- | --- |
| `models/checkpoints/` | merged-чекпоинт `q3_merge_ema.pt` |
| `models/sana/` | конфиги SANA (scheduler / transformer / vae) |
| `models/configs/` | `inference_q3.yaml` с путями внутри `src` |
| `images/kodak/` | 24 тестовых изображения Kodak |
| `result/kodak/` | битстримы и реконструкции |
| `result/metrics/` | CSV/JSON с метриками |
| `result/figures/` | RD-кривые и сравнительные графики |
| `scripts/` | запуск, baseline, графики, загрузка весов |
| `experiments/` | ноутбук с полным циклом |

## Скрипты

- `run_dit_ic.py` — сжатие/распаковка DiT-IC, BPP по фактическому битстриму,
  PSNR / MS-SSIM / LPIPS / DISTS / CLIP-IQA / MUSIQ, время и VRAM.
- `run_baselines.py` — JPEG и WebP по сетке качеств с теми же метриками.
- `make_plots.py` — RD-кривые и сравнение при одинаковом BPP.
- `fetch_weights.py` — параллельная докачиваемая загрузка больших файлов.
- `env_check.py` — отчёт об окружении.
- `build_notebook.py` — генерация ноутбука.
