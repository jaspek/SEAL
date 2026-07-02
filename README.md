# SEAL — face-recognition fusion & compression

Seminar project: **"Fusion and Compression of Face Recognition Embeddings"**
(ArcFace / MagFace / AdaFace) and its follow-up — a **rate–distortion study**
of embedding compression across six LFW-family benchmarks and 1:N retrieval.

Central question: compression that looks *free* on saturated LFW — does it stay
free as the data gets harder (LFW → XQLFW) and the gallery grows?

## Layout

```
src/facecomp/        the package (data / extract / fusion / select / compress /
                     network / evaluate / viz) — one module per pipeline stage
experiments/         runnable drivers (build_embeddings, run_rate_distortion, smoke_test)
configs/             paths.example.yaml -> copy to paths.yaml (gitignored) per machine
outputs/             results/*.csv + figures/*.png are committed;
                     embeddings/, models/ are generated and gitignored
docs/                pipeline docs, dataset download sources (DATA.md)
data/                datasets live here locally — gitignored, see docs
legacy/              original seminar scripts (pre-restructure, reference only)
encryption_attack/   separate strand: JPEG2000 selective-encryption attack lab
commands.ps1         full experiment run for the GPU machine
commands_work.ps1    CPU-only runs (gallery sweep) for the 32 GB machine
```

## Quick start

```bash
pip install -e .
python experiments/smoke_test.py            # end-to-end on synthetic data, no datasets needed
python experiments/build_embeddings.py --dataset cplfw          # needs GPU + models
python experiments/run_rate_distortion.py --emb outputs/embeddings/emb_cplfw.npz --out rd_cplfw.csv --plot
```

Embedding files follow one contract: `emb_<dataset>.npz` with per-image
`arcface/magface/adaface` (N,512) arrays, plus `pair_idx` (1:1 verification)
and/or `labels` (1:N identification).

Models and datasets are not in git — see `docs/DATA.md` and `docs/` for sources.
`--arcface ms1mv2` builds the same-training-set control (all three models MS1MV2).
