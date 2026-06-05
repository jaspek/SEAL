"""
Phase 2.3 Part B (Accuracy): LFW accuracy of each quantized variant.
Caches aligned face crops once, then runs all model variants over the cache.

Run quantize_and_bench.py FIRST so quantized_models/arcface_int8.onnx exists.
"""
import sys, os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

import config as C
from extract_embeddings import (
    parse_pairs, align_from_raw,
    preprocess_magface, preprocess_adaface,
)

for d in (C.MAGFACE_DIR, C.ADAFACE_DIR, C.ARCFACE_DIR):
    if d not in sys.path:
        sys.path.append(d)

ALIGN_CACHE = C.EMBEDDINGS_DIR / "lfw_aligned_cache.npz"
QUANT_DIR = C.EMBEDDINGS_DIR.parent / "quantized_models"


# ---------------- Cache aligned faces ----------------
def build_alignment_cache():
    if ALIGN_CACHE.exists():
        print(f"Loading alignment cache from {ALIGN_CACHE}...")
        d = np.load(ALIGN_CACHE, allow_pickle=True)
        return d["images"], d["pair_idx"]

    print("Building alignment cache (one-time, ~7 min)...")
    pairs, all_imgs = parse_pairs(C.PAIRS_CSV)
    all_paths = [info[0] for info in all_imgs]
    path_to_idx = {p: i for i, p in enumerate(all_paths)}
    N = len(all_paths)
    images = np.zeros((N, 112, 112, 3), dtype=np.uint8)
    valid = np.zeros(N, dtype=bool)

    for i, (path, name, idx) in enumerate(tqdm(all_imgs, desc="Aligning")):
        img = align_from_raw(name, idx)
        if img is not None:
            images[i] = img
            valid[i] = True

    valid_pairs = [
        (path_to_idx[a], path_to_idx[b], lbl)
        for a, b, lbl in pairs
        if a in path_to_idx and b in path_to_idx
        and valid[path_to_idx[a]] and valid[path_to_idx[b]]
    ]
    pair_idx = np.array(valid_pairs, dtype=np.int32)
    np.savez_compressed(ALIGN_CACHE, images=images, pair_idx=pair_idx)
    print(f"  cached {valid.sum()} aligned images, {len(pair_idx)} valid pairs")
    return images, pair_idx


# ---------------- Helpers ----------------
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


# ---------------- Embedding extraction per variant ----------------
def extract_torch(model, images, preprocess_fn, device, half=False):
    N = len(images)
    embs = np.zeros((N, 512), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(N)):
            if not images[i].any(): continue
            t = preprocess_fn(images[i]).to(device)
            if half: t = t.half()
            out = model(t)
            if isinstance(out, tuple): out = out[0]
            embs[i] = l2norm(out.float().cpu().numpy().squeeze())
    return embs

def extract_onnx(session, images):
    import onnxruntime as ort
    N = len(images)
    embs = np.zeros((N, 512), dtype=np.float32)
    inp = session.get_inputs()[0].name
    for i in tqdm(range(N)):
        if not images[i].any(): continue
        x = (images[i].astype(np.float32) - 127.5) / 127.5
        x = np.transpose(x, (2, 0, 1))[None, ...]
        out = session.run(None, {inp: x})[0].squeeze()
        embs[i] = l2norm(out.astype(np.float32))
    return embs


# ---------------- Loaders ----------------
def load_magface(device, half=False, dynamic_int8=False):
    from inference.network_inf import builder_inf
    class Args:
        arch = "iresnet100"; embedding_size = 512
        cpu_mode = (device == "cpu"); resume = C.MAGFACE_PTH
    m = builder_inf(Args()).eval()
    if dynamic_int8:
        m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        return m
    m = m.to(device)
    return m.half() if half else m

def load_adaface(device, half=False, dynamic_int8=False):
    import net
    sd = torch.load(C.ADAFACE_CKPT, map_location="cpu")["state_dict"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
    m = net.build_model("ir_101"); m.load_state_dict(sd); m.eval()
    if dynamic_int8:
        m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        return m
    m = m.to(device)
    return m.half() if half else m


# ---------------- Main ----------------
def main():
    images, pair_idx = build_alignment_cache()
    print(f"\n{len(images)} aligned images, {len(pair_idx)} pairs\n")
    results = []

    # MagFace
    if torch.cuda.is_available():
        print("=== MagFace FP32 GPU ===")
        m = load_magface("cuda", half=False)
        emb = extract_torch(m, images, preprocess_magface, "cuda", half=False)
        acc, _ = evaluate(emb, pair_idx); results.append(("MagFace", "FP32 GPU", acc))
        print(f"  -> {acc*100:.2f}%"); del m; torch.cuda.empty_cache()

        print("=== MagFace FP16 GPU ===")
        m = load_magface("cuda", half=True)
        emb = extract_torch(m, images, preprocess_magface, "cuda", half=True)
        acc, _ = evaluate(emb, pair_idx); results.append(("MagFace", "FP16 GPU", acc))
        print(f"  -> {acc*100:.2f}%"); del m; torch.cuda.empty_cache()

    print("=== MagFace INT8 dyn CPU ===")
    m = load_magface("cpu", dynamic_int8=True)
    emb = extract_torch(m, images, preprocess_magface, "cpu", half=False)
    acc, _ = evaluate(emb, pair_idx); results.append(("MagFace", "INT8 CPU", acc))
    print(f"  -> {acc*100:.2f}%"); del m

    # AdaFace
    if torch.cuda.is_available():
        print("\n=== AdaFace FP32 GPU ===")
        a = load_adaface("cuda", half=False)
        emb = extract_torch(a, images, preprocess_adaface, "cuda", half=False)
        acc, _ = evaluate(emb, pair_idx); results.append(("AdaFace", "FP32 GPU", acc))
        print(f"  -> {acc*100:.2f}%"); del a; torch.cuda.empty_cache()

        print("=== AdaFace FP16 GPU ===")
        a = load_adaface("cuda", half=True)
        emb = extract_torch(a, images, preprocess_adaface, "cuda", half=True)
        acc, _ = evaluate(emb, pair_idx); results.append(("AdaFace", "FP16 GPU", acc))
        print(f"  -> {acc*100:.2f}%"); del a; torch.cuda.empty_cache()

    print("=== AdaFace INT8 dyn CPU ===")
    a = load_adaface("cpu", dynamic_int8=True)
    emb = extract_torch(a, images, preprocess_adaface, "cpu", half=False)
    acc, _ = evaluate(emb, pair_idx); results.append(("AdaFace", "INT8 CPU", acc))
    print(f"  -> {acc*100:.2f}%"); del a

    # ArcFace
    import onnxruntime as ort
    print("\n=== ArcFace FP32 ONNX ===")
    sess = ort.InferenceSession(C.ARCFACE_ONNX,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    emb = extract_onnx(sess, images)
    acc, _ = evaluate(emb, pair_idx); results.append(("ArcFace", "FP32 ONNX", acc))
    print(f"  -> {acc*100:.2f}%")

    print("=== ArcFace INT8 ONNX ===")
    int8_path = str(QUANT_DIR / "arcface_int8.onnx")
    sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    emb = extract_onnx(sess, images)
    acc, _ = evaluate(emb, pair_idx); results.append(("ArcFace", "INT8 ONNX", acc))
    print(f"  -> {acc*100:.2f}%")

    # Summary
    print("\n\n=== Final Accuracy Table ===")
    print(f"  {'Model':10s} | {'Precision':12s} | {'Accuracy':>10s}")
    print("-" * 40)
    for model, prec, acc in results:
        print(f"  {model:10s} | {prec:12s} | {acc*100:9.2f}%")


if __name__ == "__main__":
    main()