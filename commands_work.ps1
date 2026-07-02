# commands_work.ps1 — WORK machine (i5, 32 GB RAM, no GPU): everything that
# starts from cached emb_*.npz. Pure numpy/sklearn — no torch/insightface needed.
#
# One-time setup on this box:
#   git clone https://github.com/jaspek/SEAL.git ; cd SEAL
#   pip install -e .
#   copy the emb_*.npz files (from the home box) into outputs\embeddings\
#
# Run with:  powershell -ExecutionPolicy Bypass -File .\commands_work.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = "python"   # system python where `pip install -e .` was run

Start-Transcript -Path "outputs\results\run_log_work.txt" -Append

function Step {
    Write-Host "`n=== python $($args -join ' ') ===" -ForegroundColor Cyan
    & $py @args
    if ($LASTEXITCODE -ne 0) { Stop-Transcript; throw "FAILED: python $($args -join ' ')" }
}

# --- 1. LFW gallery sweep incl. the 1M headline (needs the 32 GB) -----------
if (Test-Path "outputs\embeddings\lfw_embeddings.npz") {
    foreach ($N in 0, 10000, 100000, 500000, 1000000) {
        Step experiments/run_rate_distortion.py --emb outputs/embeddings/lfw_embeddings.npz --num-distractors $N --out "rd_lfw_N$N.csv"
    }
} else {
    Write-Host "SKIP sweep: outputs\embeddings\lfw_embeddings.npz not found" -ForegroundColor Yellow
}

# --- 2. TinyFace FULL evaluation (153k real distractors, RAM-heavy) ---------
if (Test-Path "outputs\embeddings\emb_tinyface_full.npz") {
    Step experiments/run_rate_distortion.py --emb outputs/embeddings/emb_tinyface_full.npz --out rd_tinyface_full.csv --plot
} else {
    Write-Host "SKIP TinyFace-full: outputs\embeddings\emb_tinyface_full.npz not found (build it on the home GPU first)" -ForegroundColor Yellow
}

Stop-Transcript
Write-Host "`nALL DONE — tables in outputs\results\, log in outputs\results\run_log_work.txt" -ForegroundColor Green
