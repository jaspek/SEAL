# commands.ps1 — HOME machine (RTX 3060): repair + finalize run.
# Includes BOTH fixes:
#   (1) ms1mv2 ArcFace preprocessing fix  -> rebuild the 5 broken *_ctrl embeddings
#   (2) seeded PCA/ITQ                    -> one clean re-run of every evaluation,
#                                            so all CSVs are reproducible
# Run with:  powershell -ExecutionPolicy Bypass -File .\commands.ps1
# (emb_lfw_ctrl.npz was already rebuilt on the fixed code and is skipped.)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "seminar-face-fusion\.venv\Scripts\python.exe"

Start-Transcript -Path "outputs\results\run_log.txt" -Append

function Step {
    Write-Host "`n=== python $($args -join ' ') ===" -ForegroundColor Cyan
    & $py @args
    if ($LASTEXITCODE -ne 0) { Stop-Transcript; throw "FAILED: python $($args -join ' ')" }
}

# --- 1. rebuild the five broken ctrl embeddings (fixed ms1mv2 preprocessing) --
foreach ($d in "cplfw","xqlfw","calfw","cfp_fp") {
    Step experiments/build_embeddings.py --dataset "${d}_ctrl" --bin "data/insightface_bins/$d.bin" --arcface ms1mv2 --force
}
Step experiments/build_embeddings.py --dataset sllfw_ctrl --source sllfw --arcface ms1mv2 --force

# --- 2. re-evaluate ALL ctrl datasets (fresh embeddings + seeded PCA/ITQ) ----
foreach ($d in "lfw","cplfw","xqlfw","calfw","cfp_fp","sllfw") {
    Step experiments/run_rate_distortion.py --emb "outputs/embeddings/emb_${d}_ctrl.npz" --out "rd_${d}_ctrl.csv" --plot
}

# --- 3. seeded re-run of the buffalo evaluations (embeddings unchanged) ------
foreach ($d in "cplfw","xqlfw","calfw","cfp_fp","sllfw") {
    Step experiments/run_rate_distortion.py --emb "outputs/embeddings/emb_$d.npz" --out "rd_$d.csv" --plot
}
Step experiments/run_rate_distortion.py --emb outputs/embeddings/lfw_embeddings.npz --num-distractors 10000 --out rd_lfw.csv --plot

# --- 4. refresh the error-overlap + redundancy summary ------------------------
Step experiments/run_error_overlap.py --all --out overlap_summary.csv

Stop-Transcript
Write-Host "`nALL DONE." -ForegroundColor Green
Write-Host "Every CSV in outputs\results\ is now reproducible (seeded) and ctrl is valid."
Write-Host "NEXT: git add -A ; git status ; git commit ; git push" -ForegroundColor Yellow
