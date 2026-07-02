# Data & Model Files — where to get them

None of the large files live in git (they exceed GitHub's 100 MB limit and are
listed in `.gitignore`). This document lists every file the code expects, where
to download it, and where to put it. Filenames/paths come from `config.py`.

> **Heads-up:** `config.py` currently uses hard-coded absolute paths
> (`C:\Users\jasko\...\Program_3`, `...\Program_6`, `...\Libraries\...`).
> On another machine, **edit `config.py`** to point at wherever you place these
> files. The relative locations below are a suggested clean layout.

---

## 1. Datasets

### LFW (Labeled Faces in the Wild)
Used for face-verification evaluation.

| File / folder | Source | Notes |
|---------------|--------|-------|
| `lfw_raw/` (raw images) | http://vis-www.cs.umass.edu/lfw/lfw.tgz | Official UMass distribution (~173 MB) |
| `pairs.txt` (eval protocol) | http://vis-www.cs.umass.edu/lfw/pairs.txt | Standard 6,000-pair protocol |
| `lfw_aligned_png/` | generated | 112×112 aligned crops produced by `extract_embeddings.py` (InsightFace alignment) |
| `pairs.csv` | generated/converted | the project's CSV form of `pairs.txt` |

Project homepage (mirrors, deep-funneled versions, citation):
http://vis-www.cs.umass.edu/lfw/

`config.py` expects (edit to your machine):
```
LFW_RAW_DIR     = .../data/lfw_raw
LFW_ALIGNED_DIR = .../data/lfw_aligned_png
PAIRS_CSV       = .../data/pairs.csv
```

---

## 2. Pretrained face-recognition models

### ArcFace (R100, MS1MV2) — ONNX
- Expected file: `pretrained/ms1mv2_arcface_r100.onnx`
- Source: InsightFace model zoo / ArcFace releases —
  https://github.com/deepinsight/insightface/tree/master/model_zoo
- Alternative: the InsightFace `buffalo_l` pack auto-downloads to
  `~/.insightface/models/buffalo_l/` (contains `w600k_r50.onnx`) and is used
  directly by `extract_embeddings.py` for detection + alignment — no manual
  download needed for that part.

### MagFace (iResNet100, MS1MV2)
- Expected file: `checkpoints/magface_epoch_00025.pth`
- Repo: https://github.com/IrvingMeng/MagFace
- Download: see the **"Pretrained Model"** table in that README
  (Google Drive / Baidu links). Pick the **iResNet100 / MS1MV2** checkpoint.

### AdaFace (IR-101, MS1MV2)
- Expected file: `pretrained/adaface_ir101_ms1mv2.ckpt`
- Repo: https://github.com/mk-minchul/AdaFace
- Download: see the **"Pretrained Models"** table in that README
  (Google Drive links). Pick **IR-101, trained on MS1MV2**.

`config.py` expects (edit to your machine):
```
ARCFACE_DIR  = .../Libraries/ArcFace   ->  pretrained/ms1mv2_arcface_r100.onnx
MAGFACE_DIR  = .../Libraries/MagFace   ->  checkpoints/magface_epoch_00025.pth
ADAFACE_DIR  = .../Libraries/AdaFace   ->  pretrained/adaface_ir101_ms1mv2.ckpt
```

> The MagFace and AdaFace **folders are also code repos** you must clone and add
> to `PYTHONPATH` (the code does `import net`, `from inference.network_inf import
> builder_inf`). See `README.md` step 4.

---

## 3. Generated files (no download — produced by the scripts)

These are recreated locally and stay out of git:

| Folder / file | Produced by | Approx size |
|---------------|-------------|-------------|
| `embeddings/lfw_embeddings.npz` | `extract_embeddings.py` | ~42 MB |
| `embeddings/lfw_aligned_cache.npz` | `extract_embeddings.py` | ~238 MB |
| `quantized_models/*.pt`, `*.onnx` | `quantize_and_bench.py`, `fix_arcface_quantisation.py` | ~1.3 GB total |
| `error_crops/*.png` | `dump_error_crops.py` | small |

To rebuild from scratch on a new machine: download the datasets + pretrained
models above, fix the paths in `config.py`, then run
`extract_embeddings.py` → `quantize_and_bench.py` → `run_evaluation.py`.

---

## Suggested clean layout

```
SEAL/seminar-face-fusion/
├── data/
│   ├── lfw_raw/
│   ├── lfw_aligned_png/   (generated)
│   ├── pairs.txt
│   └── pairs.csv
├── Libraries/
│   ├── ArcFace/pretrained/ms1mv2_arcface_r100.onnx
│   ├── MagFace/   (cloned repo) checkpoints/magface_epoch_00025.pth
│   └── AdaFace/   (cloned repo) pretrained/adaface_ir101_ms1mv2.ckpt
├── embeddings/        (generated, gitignored)
└── quantized_models/  (generated, gitignored)
```
