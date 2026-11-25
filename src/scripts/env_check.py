"""Write a reproducible environment report to src/result/logs/env_info.txt."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
CACHE_ROOT = SRC_ROOT / "models" / "cache"
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))

import torch  # noqa: E402


def main() -> None:
    info = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cudnn": torch.backends.cudnn.version(),
    }
    logs = SRC_ROOT / "result" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    text = json.dumps(info, indent=2)
    (logs / "env_info.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
