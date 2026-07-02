#!/usr/bin/env bash
# commands_work.sh — WORK machine (Linux, 32 GB RAM, no GPU): repair run.
# The first gallery sweep was invalid (Windows-path labels collapsed to one
# identity on Linux — fixed in facecomp.data.emb), so the sweep re-runs here.
# The TinyFace-full result was NOT affected (explicit labels) and is skipped
# if its CSV already exists; set FORCE_TINYFACE=1 to redo it anyway.
#
# Run with:  git pull && bash commands_work.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"

mkdir -p outputs/results outputs/figures
exec > >(tee -a outputs/results/run_log_work.txt) 2>&1

step() {
    echo ""
    echo "=== $PY $* ==="
    "$PY" "$@"
}

# --- 1. LFW gallery sweep, REDONE with fixed labels + seeded PCA/ITQ --------
if [[ -f outputs/embeddings/lfw_embeddings.npz ]]; then
    for N in 0 10000 100000 500000 1000000; do
        step experiments/run_rate_distortion.py --emb outputs/embeddings/lfw_embeddings.npz \
             --num-distractors "$N" --out "rd_lfw_N$N.csv"
    done
else
    echo "SKIP sweep: outputs/embeddings/lfw_embeddings.npz not found"
fi

# --- 2. TinyFace FULL evaluation (valid from the first run; skipped if done) -
if [[ -f outputs/results/rd_tinyface_full.csv && "${FORCE_TINYFACE:-0}" != "1" ]]; then
    echo "SKIP TinyFace-full: rd_tinyface_full.csv already exists (FORCE_TINYFACE=1 to redo)"
elif [[ -f outputs/embeddings/emb_tinyface_full.npz ]]; then
    step experiments/run_rate_distortion.py --emb outputs/embeddings/emb_tinyface_full.npz \
         --out rd_tinyface_full.csv --plot
else
    echo "SKIP TinyFace-full: outputs/embeddings/emb_tinyface_full.npz not found (build it on the home GPU first)"
fi

echo ""
echo "ALL DONE — tables in outputs/results/, log in outputs/results/run_log_work.txt"
echo "Sanity check: rd_lfw_N0.csv rank1 should be ~0.97-0.99, NOT exactly 1.0."
