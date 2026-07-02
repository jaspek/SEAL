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


BITS_PER_DIM = {"fp32": 32, "fp16": 16, "int8": 8, "binary": 1}
_PRECISION = {"fp32": fp32, "fp16": fp16, "int8": int8, "binary": binary}


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


def compress(emb, method="fp32", dim_reducer=None):
    """Return (decoded_float_emb, bits_per_template). `dim_reducer` must be pre-fit."""
    x, d = emb, emb.shape[1]
    if dim_reducer is not None:
        x, d = dim_reducer.transform(x), dim_reducer.k
    return _PRECISION[method](x), d * BITS_PER_DIM[method]
