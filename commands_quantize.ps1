# commands_quantize.ps1 - network INT8 quantization experiment.
# 1. quantize w600k_r50 -> int8   2. rebuild embeddings with the int8 net
# 3. evaluate: int8-arc single, int8 fused, fp32-arc single baseline
# Runtime: ~45-60 min (int8 arcface runs on CPU; magface/adaface on GPU).
# Run with:
#   powershell -ExecutionPolicy Bypass -File .\commands_quantize.ps1
# Prerequisite once: pip install onnx   (ORT quantization tooling needs it)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot "seminar-face-fusion\.venv\Scripts\python.exe"
$bins = "data\insightface_bins"   # = data.bin_pack in configs/paths.yaml

Start-Transcript -Path "outputs\results\run_log_quant.txt" -Append

function Step {
    Write-Host "`n=== python $($args -join ' ') ===" -ForegroundColor Cyan
    & $py @args
    if ($LASTEXITCODE -ne 0) { Stop-Transcript; throw "FAILED: python $($args -join ' ')" }
}

# --- 1. quantize the network --------------------------------------------------
# If the script warns about a low cosine check, rerun this step by hand with
# --reduce-range before continuing.
Step experiments/quantize_arcface.py

# --- 2. embeddings with the int8 network --------------------------------------
foreach ($d in "lfw","cplfw","xqlfw","calfw","cfp_fp") {
    Step experiments/build_embeddings.py --dataset "${d}_q8" --bin "$bins\$d.bin" --arcface int8
}
Step experiments/build_embeddings.py --dataset sllfw_q8 --source sllfw --arcface int8

# --- 3. evaluations ------------------------------------------------------------
foreach ($d in "lfw","cplfw","xqlfw","calfw","cfp_fp","sllfw") {
    # int8-network ArcFace alone, and int8-network fused template
    Step experiments/run_rate_distortion.py --emb "outputs/embeddings/emb_${d}_q8.npz" --models arcface --out "rd_${d}_q8arc.csv"
    Step experiments/run_rate_distortion.py --emb "outputs/embeddings/emb_${d}_q8.npz" --out "rd_${d}_q8.csv"
}
# fp32-network ArcFace-alone baselines from the EXISTING embeddings (cheap)
foreach ($d in "cplfw","xqlfw","calfw","cfp_fp","sllfw") {
    Step experiments/run_rate_distortion.py --emb "outputs/embeddings/emb_$d.npz" --models arcface --out "rd_${d}_arc.csv"
}
Step experiments/run_rate_distortion.py --emb outputs/embeddings/lfw_embeddings.npz --models arcface --out rd_lfw_arc.csv

Stop-Transcript
Write-Host "`nDONE - compare rd_<d>_arc.csv (fp32 net) vs rd_<d>_q8arc.csv (int8 net)" -ForegroundColor Green
