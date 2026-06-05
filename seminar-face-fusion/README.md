# Seminar — Face Embedding Fusion

Multimedia seminar, Topic A: combining face-recognition embeddings (ArcFace,
MagFace, AdaFace) and reducing model complexity via quantization / pruning.

---

## Running on a fresh computer (only PyCharm installed)

### 1. Get the code

```bash
git clone https://github.com/jaspek/SEAL.git
```

Open the `SEAL/seminar-face-fusion` folder as a project in PyCharm.

> Make sure you have **Python 3.10** installed on the machine. PyCharm uses it
> to create the virtual environment in the next step.

### 2. Create a virtual environment + install dependencies

PyCharm usually offers "Create venv from requirements.txt" automatically when
it sees `requirements.txt` — accept it. To do it manually instead:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

This installs the **CPU** builds of PyTorch and ONNX Runtime, which run on any
machine. For an NVIDIA GPU, see "GPU setup" below.

### 3. Get the things that are NOT in git

The repository **only contains source code**. Three things are deliberately
excluded (see `.gitignore`) and must be obtained separately on each machine:

| What | Why it's not in git | How to get it |
|------|---------------------|---------------|
| `quantized_models/`, `embeddings/` | 1.6 GB of generated files, exceed GitHub's 100 MB limit | Re-generate by running the scripts (see below), or copy from a USB / your other PC |
| **MagFace** repo + weights | External research repo | `git clone https://github.com/IrvingMeng/MagFace` + download checkpoint |
| **AdaFace** repo + weights | External research repo | `git clone https://github.com/mk-minchul/AdaFace` + download checkpoint |
| **LFW dataset** | Large; InsightFace caches it | InsightFace eval pack, or your existing `~/.insightface` cache |

**See [`DATA.md`](DATA.md) for exact download links, filenames, and target paths for every dataset and model file.**

### 4. Make MagFace / AdaFace importable

The code does `import net` and `from inference.network_inf import builder_inf`,
which come from those two cloned repos. Add their folders to **PYTHONPATH**
(not PATH). In PyCharm: *Settings → Project → Python Interpreter → ⚙ → Show All
→ select interpreter → Paths* — add the MagFace and AdaFace folders there.

Or set an env var before running:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "C:\path\to\MagFace;C:\path\to\AdaFace"
```

The scripts also `sys.path.append(...)` these paths, so editing the paths near
the top of `main.py` / the extractor files is an alternative.

### 5. Run

```bash
python main.py               # sanity-check that all models load
python extract_embeddings.py # build embeddings cache
python run_evaluation.py     # LFW verification / fusion results
python quantize_and_bench.py # quantization experiments
```

---

## GPU setup (optional, NVIDIA only)

The CPU builds work everywhere but are slow. For a CUDA GPU:

```bash
pip uninstall torch torchvision onnxruntime
# pick the CUDA version matching your driver from https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install onnxruntime-gpu==1.23.2
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

---

## Note on `termcolor`

MagFace's code imports `termcolor`. If you hit `ModuleNotFoundError: No module
named 'termcolor'`, run `pip install termcolor`.
