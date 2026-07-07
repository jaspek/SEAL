"""Template compression. Every method decodes back to a float vector so the
downstream cosine is identical and rows differ ONLY in the bits they survived."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA


def _l2(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def fp32(emb):
    return emb.astype(np.float32)


def fp16(emb):
    return emb.astype(np.float16).astype(np.float32)


def int8(emb):
    scale = np.max(np.abs(emb), axis=1, keepdims=True) / 127.0
    scale[scale == 0] = 1.0
    q = np.round(emb / scale).clip(-127, 127).astype(np.int8)
    return _l2(q.astype(np.float32) * scale)


def binary(emb):
    b = np.sign(emb).astype(np.float32)
    b[b == 0] = 1.0
    return _l2(b)


def intb(emb, bits):
    """Symmetric per-vector B-bit scalar quantization. Generalizes int8 (bits=8)
    and binary (bits=1): qmax = 2**(bits-1)-1 levels per sign. Fills the 2-7 bit
    range the paper leaves empty between binary and int8."""
    if bits <= 1:
        return binary(emb)
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(emb), axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    q = np.round(emb / scale).clip(-qmax, qmax)
    return _l2(q.astype(np.float32) * scale)


BITS_PER_DIM = {"fp32": 32, "fp16": 16, "int8": 8, "binary": 1}
_PRECISION = {"fp32": fp32, "fp16": fp16, "int8": int8, "binary": binary}
# int2..int7 fill the bit range between binary (1) and int8 (8):
for _b in (2, 3, 4, 5, 6, 7):
    BITS_PER_DIM[f"int{_b}"] = _b
    _PRECISION[f"int{_b}"] = (lambda bb: (lambda e: intb(e, bb)))(_b)

# scalar-quantized methods also store one float32 per-vector scale (uncounted in
# the paper's bit budgets); count_scale=True in compress() adds it back.
_SCALED = {f"int{b}" for b in (2, 3, 4, 5, 6, 7, 8)}


class PCAReducer:
    def __init__(self, k, seed=0):
        self.k = k
        # random_state pins sklearn's randomized SVD — otherwise PCA/ITQ rows
        # vary run-to-run by whole points on hard datasets (seen on XQLFW).
        self.pca = PCA(n_components=k, random_state=seed)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return _l2(self.pca.transform(X))


class ITQReducer:
    """PCA -> orthogonal rotation minimizing sign-quantization error (Gong & Lazebnik).
    Pair with `binary`; beats raw PCA->sign at equal bits."""
    def __init__(self, k, n_iter=50, seed=0):
        self.k, self.n_iter, self.seed = k, n_iter, seed

    def fit(self, X):
        self.pca = PCA(n_components=self.k, random_state=self.seed).fit(X)
        V = self.pca.transform(X)
        rng = np.random.default_rng(self.seed)
        R, _ = np.linalg.qr(rng.standard_normal((self.k, self.k)))
        for _ in range(self.n_iter):
            B = np.sign(V @ R)
            B[B == 0] = 1.0
            U, _, Wt = np.linalg.svd(V.T @ B)   # Procrustes: max tr(R^T V^T B)
            R = U @ Wt
        self.R = R
        return self

    def transform(self, X):
        return _l2(self.pca.transform(X) @ self.R)


def compress(emb, method="fp32", dim_reducer=None, count_scale=False):
    """Return (decoded_float_emb, bits_per_template). `dim_reducer` must be pre-fit.
    count_scale=True adds the 32-bit per-vector scale for int* methods (honest
    bytes accounting); default False so existing tables are unchanged."""
    x, d = emb, emb.shape[1]
    if dim_reducer is not None:
        x, d = dim_reducer.transform(x), dim_reducer.k
    bits = d * BITS_PER_DIM[method]
    if count_scale and method in _SCALED:
        bits += 32
    return _PRECISION[method](x), bits
