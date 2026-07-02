"""
Quantize the actual working ArcFace ONNX (buffalo_l/w600k_r50) and re-evaluate.
"""
import sys, os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from onnxruntime.quantization import quantize_dynamic, QuantType

import config as C

QUANT_DIR = C.EMBEDDINGS_DIR.parent / "quantized_models"
ALIGN_CACHE = C.EMBEDDINGS_DIR / "lfw_aligned_cache.npz"

# The actual ArcFace recognition model that buffalo_l uses
BUFFALO_REC_FP32 = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
BUFFALO_REC_INT8 = QUANT_DIR / "w600k_r50_int8.onnx"


def l2norm(v):
    return v / (np.linalg.norm(v) + 1e-10)

def evaluate(emb, pair_idx):
    pair_y = pair_idx[:, 2]
    scores = np.sum(emb[pair_idx[:, 0]] * emb[pair_idx[:, 1]], axis=1)
    accs = []
    for tr, te in KFold(n_splits=10, shuffle=False).split(pair_idx):
        best_acc, best_thr = 0, 0
        for thr in np.arange(-1, 1, 0.005):
            a = ((scores[tr] > thr) == pair_y[tr]).mean()
            if a > best_acc: best_acc, best_thr = a, thr
        accs.append(((scores[te] > best_thr) == pair_y[te]).mean())
    return np.mean(accs), np.std(accs)


def file_size_mb(p):
    return os.path.getsize(p) / (1024 * 1024)


def extract_via_rec(rec, images):
    """Use the high-level rec.get_feat() (handles preprocessing internally)."""
    N = len(images)
    embs = np.zeros((N, 512), dtype=np.float32)
    for i in tqdm(range(N)):
        if not images[i].any(): continue
        feat = rec.get_feat(images[i]).squeeze().astype(np.float32)
        embs[i] = l2norm(feat)
    return embs


def extract_via_onnx_session(session, images):
    """Direct ONNX call with the same preprocessing rec.get_feat uses internally."""
    inp = session.get_inputs()[0].name
    N = len(images)
    embs = np.zeros((N, 512), dtype=np.float32)
    for i in tqdm(range(N)):
        if not images[i].any(): continue
        # InsightFace's ArcFaceONNX preprocessing: BGR, (x-127.5)/127.5
        x = (images[i].astype(np.float32) - 127.5) / 127.5
        x = np.transpose(x, (2, 0, 1))[None, ...]
        out = session.run(None, {inp: x})[0].squeeze().astype(np.float32)
        embs[i] = l2norm(out)
    return embs


def main():
    # Sanity check on file
    if not BUFFALO_REC_FP32.exists():
        print(f"!! Couldn't find buffalo_l recognition ONNX at:\n   {BUFFALO_REC_FP32}")
        print("   Make sure InsightFace's buffalo_l pack has been downloaded "
              "(it auto-downloads on first FaceAnalysis(name='buffalo_l').prepare())")
        return

    print(f"FP32 ArcFace ONNX:  {BUFFALO_REC_FP32}")
    print(f"  size: {file_size_mb(BUFFALO_REC_FP32):.1f} MB")

    # Quantize
    if not BUFFALO_REC_INT8.exists():
        print("\nQuantizing to INT8 (this takes ~30s)...")
        quantize_dynamic(str(BUFFALO_REC_FP32), str(BUFFALO_REC_INT8),
                         weight_type=QuantType.QUInt8)
    print(f"INT8 ArcFace ONNX:  {BUFFALO_REC_INT8}")
    print(f"  size: {file_size_mb(BUFFALO_REC_INT8):.1f} MB")

    # Load aligned cache
    if not ALIGN_CACHE.exists():
        print("!! No alignment cache found. Run eval_quantized.py first.")
        return
    d = np.load(ALIGN_CACHE, allow_pickle=True)
    images, pair_idx = d["images"], d["pair_idx"]

    # ---- FP32 baseline (via FaceAnalysis -> rec.get_feat) ----
    print("\n=== ArcFace FP32 (buffalo_l rec) ===")
    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    rec = app.models["recognition"]
    emb = extract_via_rec(rec, images)
    acc_fp32, _ = evaluate(emb, pair_idx)
    print(f"  -> {acc_fp32*100:.2f}%")

    # ---- INT8 ----
    print("\n=== ArcFace INT8 (buffalo_l rec, quantized) ===")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(BUFFALO_REC_INT8),
                                providers=["CPUExecutionProvider"])
    emb = extract_via_onnx_session(sess, images)
    acc_int8, _ = evaluate(emb, pair_idx)
    print(f"  -> {acc_int8*100:.2f}%")

    # Summary
    print("\n=== ArcFace Quantization Summary ===")
    print(f"  FP32 ({file_size_mb(BUFFALO_REC_FP32):.1f} MB):  {acc_fp32*100:.2f}%")
    print(f"  INT8 ({file_size_mb(BUFFALO_REC_INT8):.1f} MB):  {acc_int8*100:.2f}%")
    print(f"  Compression: {file_size_mb(BUFFALO_REC_FP32) / file_size_mb(BUFFALO_REC_INT8):.1f}x")
    print(f"  Accuracy delta: {(acc_int8 - acc_fp32)*100:+.2f}%")


if __name__ == "__main__":
    main()