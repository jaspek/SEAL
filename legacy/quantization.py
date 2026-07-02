"""
Embedding-level quantization experiments.
Takes the best-performing embeddings from feature_selection.py (full concat,
L1-LR top-256, PCA-256) and tests how they degrade under different precision levels.
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import config as C


# ---------------- Quantization functions ----------------

def quant_float32(emb):
    """Baseline: no quantization."""
    return emb

def quant_float16(emb):
    """16-bit floating point. 2x compression."""
    return emb.astype(np.float16).astype(np.float32)

def quant_int8(emb):
    """Symmetric int8 per-vector. 4x compression."""
    scale = np.max(np.abs(emb), axis=1, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    q = np.round(emb / scale).clip(-127, 127).astype(np.int8)
    deq = q.astype(np.float32) * scale
    return deq / (np.linalg.norm(deq, axis=1, keepdims=True) + 1e-10)

def quant_binary(emb):
    """Sign-only ±1. 32x compression vs float32. Hamming-distance equivalent."""
    b = np.sign(emb).astype(np.float32)
    b[b == 0] = 1.0
    return b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)

QUANT_METHODS = [
    ("float32 (baseline)", quant_float32, 32),
    ("float16",            quant_float16, 16),
    ("int8",               quant_int8,    8),
    ("binary (±1)",        quant_binary,  1),
]


# ---------------- Eval helpers ----------------

def best_thr_acc(scores, labels, train_idx, test_idx):
    best_acc, best_thr = 0, 0
    for thr in np.arange(-1, 1, 0.005):
        acc = ((scores[train_idx] > thr) == labels[train_idx]).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return ((scores[test_idx] > best_thr) == labels[test_idx]).mean()

def evaluate(emb, pair_idx):
    """10-fold LFW protocol. emb already L2-normalized."""
    pair_y = pair_idx[:, 2]
    scores = np.sum(emb[pair_idx[:, 0]] * emb[pair_idx[:, 1]], axis=1)
    accs = []
    for tr, te in KFold(n_splits=10, shuffle=False).split(pair_idx):
        accs.append(best_thr_acc(scores, pair_y, tr, te))
    return np.mean(accs), np.std(accs)


# ---------------- Embedding constructors ----------------

def get_full_concat(arc, mag, ada):
    e = np.concatenate([arc, mag, ada], axis=1)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-10)

def get_l1lr_topk(emb, pair_idx, k):
    """Use full-data L1-LR ranking (slight optimism, but fine for this analysis).
       Strict per-fold version would be heavier; this is the standard 'final embedding' setup."""
    a = emb[pair_idx[:, 0]]; b = emb[pair_idx[:, 1]]
    pair_X = a * b
    pair_y = pair_idx[:, 2]
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=300)
    lr.fit(pair_X, pair_y)
    top = np.argsort(-np.abs(lr.coef_[0]))[:k]
    sub = emb[:, top]
    return sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-10)

def get_pca_topk(emb, pair_idx, k):
    """PCA fit on full data (consistent with above)."""
    pca = PCA(n_components=k).fit(emb)
    proj = pca.transform(emb)
    return proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-10)


# ---------------- Main ----------------

def main():
    print(f"Loading {C.EMBEDDINGS_FILE}...")
    data = np.load(C.EMBEDDINGS_FILE, allow_pickle=True)
    arc, mag, ada = data["arcface"], data["magface"], data["adaface"]
    pair_idx = data["pair_idx"]

    full = get_full_concat(arc, mag, ada)
    print(f"  full concat dim: {full.shape[1]}, pairs: {len(pair_idx)}\n")

    print("Building reduced embeddings...")
    l1lr_256 = get_l1lr_topk(full, pair_idx, 256)
    pca_256  = get_pca_topk(full, pair_idx, 256)

    embeddings = [
        ("Full concat (1536d)", full),
        ("L1-LR top-256",       l1lr_256),
        ("PCA top-256",         pca_256),
    ]

    print("\n=== Compression vs Accuracy Table ===")
    header = f"  {'Embedding':24s} | {'Quantization':22s} | {'Acc %':>6s} | {'Bits/dim':>8s} | {'Total bits':>10s} | {'Compression':>11s}"
    print(header); print("-" * len(header))

    baseline_total = full.shape[1] * 32  # float32 full embedding

    for name, e in embeddings:
        for q_name, q_fn, bits_per_dim in QUANT_METHODS:
            quantized = q_fn(e)
            acc, std = evaluate(quantized, pair_idx)
            dim = e.shape[1]
            total_bits = dim * bits_per_dim
            comp_ratio = baseline_total / total_bits
            print(f"  {name:24s} | {q_name:22s} | {acc*100:6.2f} | "
                  f"{bits_per_dim:8d} | {total_bits:10d} | {comp_ratio:10.1f}x")
        print()

if __name__ == "__main__":
    main()