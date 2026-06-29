# Shape A — Extreme face-template compression as 1:N retrieval at scale

A CPU-first toolkit for the "efficiency-artifact" line: push compression
end-to-end (template **and** network) and evaluate it where compression actually
*matters* — **1:N identification against a large gallery**, where "free on LFW
1:1" stops being free. The deliverable is a **rate–distortion curve**
(bits/template vs Rank-1 / TAR@FAR) plus a deployable artifact (tiny model +
~32-byte template).

Everything in `facecomp/` + the two top-level scripts runs on your **6-core /
32 GB / no-GPU** box. Only three things want the **home RTX-3060** box.

---

## What runs where

| Experiment | Script | Machine |
|---|---|---|
| Template compression sweep (fp16/int8/binary/PCA/ITQ/PQ/OPQ) | `run_rate_distortion.py` | **CPU box** ✅ |
| 1:N (Rank-k / CMC / mAP) + 1:1 (EER / TAR@FAR) | `facecomp/evaluate.py` | **CPU box** ✅ |
| Large-gallery search (≤ ~1M distractors) | faiss-cpu inside the above | **CPU box** ✅ (32 GB is plenty) |
| Static INT8 PTQ of the ONNX recognizer (the real 4× on MagFace/AdaFace) | `network/quantize_onnx_static.py` | **CPU box** ✅ |
| One-shot structured pruning + param/FLOP counts (no retrain) | `network/prune_oneshot.py` | **CPU box** ✅ |
| Device-honest latency | `network/benchmark_latency.py` | both (quote within-device) |
| **2:4 sparsity speed-up** (+INT8), QAT, any fine-tune | `network/sparsity_24_gpu.py` | **home GPU box** ⛔ |

Rule of thumb: anything that *measures accuracy or storage* runs on the CPU box;
anything that *measures GPU speed-up or needs training* goes home.

---

## Install (CPU box)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_cpu.txt
python smoke_test.py        # validates the whole harness on synthetic data, ~15s
```

`smoke_test.py` needs no data and no models. If it prints a sane ordered table
and `OK: ...`, the harness is good.

---

## The one wiring step: produce the embeddings file

The toolkit is decoupled from extraction so you reuse your existing
ArcFace/MagFace/AdaFace extractor. The only change vs your current pipeline:
save **per-image** embeddings + an **identity id per image** (not pair lists).

Save an `emb.npz` like:

```python
np.savez("emb.npz",
         arcface=A,           # (N, 512) float32, one row per image
         magface=M,           # (N, 512)
         adaface=Ad,          # (N, 512)
         labels=ids,          # (N,) int   -> identity id (e.g. LFW folder name -> int)
         image_paths=paths)   # (N,) str   optional
```

For LFW: every image lives in `lfw/<Name>/<Name>_NNNN.jpg`; map `<Name>` to an
integer id. People with ≥2 images become gallery+probe; people with 1 image
become gallery distractors automatically. (For a *real* 1:N benchmark use a set
with native gallery/probe structure — e.g. TinyFace — same `.npz` contract.)

---

## Run the study (CPU box)

```bash
# single model
python run_rate_distortion.py --emb emb.npz --models arcface --out rd_arc.csv --plot

# fused (concat 3 models + renormalize) — the paper's 1536-d representation
python run_rate_distortion.py --emb emb.npz --models arcface magface adaface \
       --out rd_fused.csv --plot
```

### The headline figure: "compression stops being free at scale"

Run the *same* embeddings at growing gallery sizes and overlay Rank-1:

```bash
for N in 0 10000 100000 1000000; do
  python run_rate_distortion.py --emb emb.npz --models arcface magface adaface \
         --num-distractors $N --out rd_N$N.csv
done
```

Then plot Rank-1 vs bits/template, one line per gallery size. The prediction
that distinguishes this paper: the binary / low-bit templates that look *free* at
small N visibly **fall off as N grows**, while fp16/int8 hold — i.e. the
rate–distortion curve *steepens with gallery size*. That divergence is the
result; 1:1 LFW accuracy can't show it.

> RAM note: a 1M × 512 float32 gallery is ~2 GB; faiss flat search over it fits
> comfortably in 32 GB. 1M synthetic distractors are easy negatives (they only
> show the *geometry*); for an honest stress use real distractor embeddings from
> a large face set via `--num-distractors 0` + `data.append_distractors(...)`.

---

## Methods in the sweep (`facecomp/compress.py:default_zoo`)

| Method | bits/template (D=1536, k=256) | role |
|---|---|---|
| fp32 / fp16 / int8 | 49152 / 24576 / 12288 | precision baselines |
| binary (sign) | 1536 | the paper's 32× template |
| PCA-256 + {fp32,int8,binary} | 8192 / 2048 / **256** | dim-reduction + precision |
| ITQ-256 | **256** | learned rotation before sign — should beat PCA→sign |
| PQ / OPQ (M×8) | M×8 | the missing tool: usually beats binary at equal bits |

Add RaBitQ / Matryoshka here later as extra rows; the harness only cares that a
method exposes `approx_vectors()` and `.bits`.

---

## Network half (the "tiny model" artifact)

1. `network/quantize_onnx_static.py` — **CPU**. Static INT8 with calibration
   crops; gives the real ~4× on MagFace/AdaFace that dynamic INT8 missed. Re-run
   `run_rate_distortion.py` on embeddings from the INT8 model to confirm accuracy
   is unchanged.
2. `network/prune_oneshot.py` — **CPU**. One-shot structured pruning sweep (no
   retrain) via Torch-Pruning/DepGraph; tests "embedding redundancy ⇒ weight
   redundancy". Outputs params/FLOPs + pruned checkpoints.
3. `network/sparsity_24_gpu.py` + `benchmark_latency.py --gpu` — **home box**.
   2:4 sparsity + INT8 + within-device latency = the hardware-real speed-up.

End state to report: FP32 (~250 MB model, 192-byte float template) → INT8 +
(pruned/2:4) model + **256-bit template**, with the rate–distortion curve and a
within-device latency number — evaluated as 1:N at scale.

---

## Notes / honesty flags for the write-up

- **decode→cosine** is the single ranking metric for every method (PQ included),
  so rows differ only in bits. Production PQ uses ADC (faster, marginally
  different) — state this.
- Compressor fit (PCA/ITQ/PQ codebooks) is done on the gallery; pass a disjoint
  train set if you want to rule out optimism.
- Bit counts ignore the amortized codebook/rotation/scale overhead — fine at
  gallery scale, but say so.
- The 1M synthetic distractors are *easy*; real distractors are the honest test.
