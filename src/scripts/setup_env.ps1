# Create the DiT-IC environment in the project root (.venv).
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File src\scripts\setup_env.ps1
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

function Find-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) { return "uv" }
    $candidate = Join-Path $env:APPDATA "Python\Python311\Scripts\uv.exe"
    if (Test-Path $candidate) { return $candidate }
    throw "uv not found. Install it first: python -m pip install --user uv"
}

$Uv = Find-Uv
$Python = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "[1/4] Python 3.12"
& $Uv python install 3.12
if (-not (Test-Path $Python)) {
    & $Uv venv --python 3.12 .venv
}

Write-Host "[2/4] PyTorch 2.8 + CUDA 12.8"
& $Uv pip install --python $Python torch==2.8.0 torchvision==0.23.0 `
    --index-url https://download.pytorch.org/whl/cu128

Write-Host "[3/4] Exact environment from requirements-lock.txt"
& $Uv pip install --python $Python --no-deps -r (Join-Path $Root "src\requirements-lock.txt")

Write-Host "[4/4] Jupyter kernel 'ditic'"
& $Python -m ipykernel install --prefix $Root\.venv --name ditic --display-name "DiT-IC (.venv)"

Write-Host "Done. Check with: .venv\Scripts\python.exe src\scripts\env_check.py"
