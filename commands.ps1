# commands.ps1 — HOME machine (RTX 3060, 16 GB): everything that needs the GPU.
# Sequential, stops on first failure, logs to outputs\results\run_log.txt.
# Run with:  powershell -ExecutionPolicy Bypass -File .\commands.ps1
#
# The gallery sweep and the full-TinyFace EVALUATION live in commands_work.ps1
# (work box, 32 GB, CPU-only). Bridge between machines = outputs\embeddings\*.npz
# (gitignored — copy by USB/cloud, do NOT commit).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# venv python directly — no activation needed, immune to wrong-env mistakes
$py = Join-Path $PSScriptRoot "seminar-face-fusion\.venv\Scripts\python.exe"

Start-Transcript -Path "outputs\results\run_log.txt" -Append

function Step {
    Write-Host "`n=== python $($args -join ' ') ===" -ForegroundColor Cyan
    & $py @args
    if ($LASTEXITCODE -ne 0) { Stop-Transcript; throw "FAILED: python $($args -join ' ')" }
}

# --- 1. four .bin datasets: build + evaluate -------------------------------
foreach ($d in "cplfw","xqlfw","calfw","cfp_fp") {
    Step experiments/build_embeddings.py --dataset $d
    Step experiments/run_rate_distortion.py --emb "outputs/embeddings/emb_$d.npz" --out "rd_$d.csv" --plot
}

# --- 2. same-training-set control (ms1mv2 ArcFace) -------------------------
foreach ($d in "lfw","cplfw","xqlfw","calfw","cfp_fp") {
    Step experiments/build_embeddings.py --dataset "${d}_ctrl" --bin "data/insightface_bins/$d.bin" --arcface ms1mv2
}
Step experiments/build_embeddings.py --dataset sllfw_ctrl --source sllfw --arcface ms1mv2
foreach ($d in "lfw","cplfw","xqlfw","calfw","cfp_fp","sllfw") {
    Step experiments/run_rate_distortion.py --emb "outputs/embeddings/emb_${d}_ctrl.npz" --out "rd_${d}_ctrl.csv" --plot
}

# --- 3. TinyFace 1:N, capped first pass (build + eval here) ----------------
Step experiments/build_embeddings.py --dataset tinyface --source tinyface --max-distractors 20000
Step experiments/run_rate_distortion.py --emb outputs/embeddings/emb_tinyface.npz --out rd_tinyface.csv --plot

# --- 4. TinyFace FULL gallery: BUILD ONLY (needs the GPU; ~6 GB image array,
#        close other apps). Its evaluation runs on the work box.  Comment this
#        block out if tonight's run should stay short. ----------------------
Step experiments/build_embeddings.py --dataset tinyface_full --source tinyface

Stop-Transcript
Write-Host "`nALL DONE." -ForegroundColor Green
Write-Host "Tables: outputs\results\   Plots: outputs\figures\   Log: outputs\results\run_log.txt"
Write-Host "NEXT: copy outputs\embeddings\*.npz to the work box, then run commands_work.ps1 there." -ForegroundColor Yellow
