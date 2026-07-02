#!/usr/bin/env bash
# commands_work.sh — WORK machine (Linux, i5 / 32 GB RAM, no GPU): everything
# that starts from cached emb_*.npz. Pure numpy/sklearn — no torch/insightface.
#
# One-time setup on this box:
#   git clone https://github.com/jaspek/SEAL.git && cd SEAL
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -e .
#   copy the emb_*.npz files (from the home box) into outputs/embeddings/
#
# Run with:  bash commands_work.sh
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

# --- 1. LFW gallery sweep incl. the 1M headline (needs the 32 GB) -----------
if [[ -f outputs/embeddings/lfw_embeddings.npz ]]; then
    for N in 0 10000 100000 500000 1000000; do
        step experiments/run_rate_distortion.py --emb outputs/embeddings/lfw_embeddings.npz \
             --num-distractors "$N" --out "rd_lfw_N$N.csv"
    done
else
    echo "SKIP sweep: outputs/embeddings/lfw_embeddings.npz not found"
fi

# --- 2. TinyFace FULL evaluation (153k real distractors, RAM-heavy) ---------
if [[ -f outputs/embeddings/emb_tinyface_full.npz ]]; then
    step experiments/run_rate_distortion.py --emb outputs/embeddings/emb_tinyface_full.npz \
         --out rd_tinyface_full.csv --plot
else
    echo "SKIP TinyFace-full: outputs/embeddings/emb_tinyface_full.npz not found (build it on the home GPU first)"
fi

echo ""
echo "ALL DONE — tables in outputs/results/, log in outputs/results/run_log_work.txt"
