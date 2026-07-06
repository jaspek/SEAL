# commands_prune_adaface.ps1 - AdaFace pruning replication, overnight run.
# 1. one-shot DepGraph sweep (3 criteria x 6 ratios, ~1 h)
# 2. distillation recovery at 10/30 percent (~2 h)
# Total: ~3 h on the RTX 3060. Run with:
#   powershell -ExecutionPolicy Bypass -File .\commands_prune_adaface.ps1
# Results: pruning_sweep_adaface.csv, pruning_finetuned_adaface.csv

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "seminar-face-fusion\.venv\Scripts\python.exe"

Start-Transcript -Path "outputs\results\run_log_prune_adaface.txt" -Append

function Step {
    Write-Host "`n=== python $($args -join ' ') ===" -ForegroundColor Cyan
    & $py @args
    if ($LASTEXITCODE -ne 0) { Stop-Transcript; throw "FAILED: python $($args -join ' ')" }
}

# --- 1. one-shot sweep (collapse pattern on the strongest backbone?) ---------
Step experiments/run_pruning.py --model adaface

# --- 2. distillation recovery (skip 50%: one epoch cannot fix it) ------------
Step experiments/finetune_pruned.py --model adaface --ratios 0.1 0.3

Stop-Transcript
Write-Host "`nDONE" -ForegroundColor Green

