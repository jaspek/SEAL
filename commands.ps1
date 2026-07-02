# commands.ps1 - HOME machine: final polish run.
# Regenerates every 1:1 rate-distortion table so all CSVs carry the new
# EER / TAR@FAR operating-point columns (embeddings unchanged, evals only).
# Runtime: ~10 min CPU. Run with:
#   powershell -ExecutionPolicy Bypass -File .\commands.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "seminar-face-fusion\.venv\Scripts\python.exe"

Start-Transcript -Path "outputs\results\run_log.txt" -Append

function Step {
    Write-Host "`n=== python $($args -join ' ') ===" -ForegroundColor Cyan
    & $py @args
    if ($LASTEXITCODE -ne 0) { Stop-Transcript; throw "FAILED: python $($args -join ' ')" }
}

# --- 1. buffalo evaluations (adds eer / tar_far columns) ---------------------
foreach ($d in "cplfw","xqlfw","calfw","cfp_fp","sllfw") {
    Step experiments/run_rate_distortion.py --emb "outputs/embeddings/emb_$d.npz" --out "rd_$d.csv" --plot
}
Step experiments/run_rate_distortion.py --emb outputs/embeddings/lfw_embeddings.npz --num-distractors 10000 --out rd_lfw.csv --plot

# --- 2. ms1mv2-control evaluations -------------------------------------------
foreach ($d in "lfw","cplfw","xqlfw","calfw","cfp_fp","sllfw") {
    Step experiments/run_rate_distortion.py --emb "outputs/embeddings/emb_${d}_ctrl.npz" --out "rd_${d}_ctrl.csv" --plot
}

Stop-Transcript
Write-Host ""
Write-Host 'ALL DONE - every 1:1 CSV now has acc + eer + tar_far columns.' -ForegroundColor Green
Write-Host 'NEXT (you): git add -A ; git status ; git commit -m "RaBitQ, learned fusion, EER/TAR-FAR columns" ; git push' -ForegroundColor Yellow
