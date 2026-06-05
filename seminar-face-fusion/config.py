from pathlib import Path

# === Paths ===
LFW_RAW_DIR = Path(r"C:\Users\jasko\Programming\Python\Program_3\data\lfw_raw")
LFW_ALIGNED_DIR = Path(r"C:\Users\jasko\Programming\Python\Program_3\data\lfw_aligned_png")
PAIRS_CSV       = Path(r"C:\Users\jasko\Programming\Python\Program_3\data\pairs.csv")

EMBEDDINGS_DIR  = Path(r"C:\Users\jasko\Programming\Python\Program_6\embeddings")
EMBEDDINGS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "lfw_embeddings.npz"

# === Library paths ===
MAGFACE_DIR = r"C:\Users\jasko\Programming\Python\Libraries\MagFace"
ADAFACE_DIR = r"C:\Users\jasko\Programming\Python\Libraries\AdaFace"
ARCFACE_DIR = r"C:\Users\jasko\Programming\Python\Libraries\ArcFace"

# === Model checkpoints ===
ARCFACE_ONNX  = rf"{ARCFACE_DIR}\pretrained\ms1mv2_arcface_r100.onnx"
MAGFACE_PTH   = rf"{MAGFACE_DIR}\checkpoints\magface_epoch_00025.pth"
ADAFACE_CKPT  = rf"{ADAFACE_DIR}\pretrained\adaface_ir101_ms1mv2.ckpt"