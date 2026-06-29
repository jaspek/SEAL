"""
Template compressors for the rate-distortion sweep.

Design: every compressor turns a matrix of embeddings X (N, D) into *decoded*
float vectors (`approx_vectors`). The evaluator then L2-normalizes those vectors
and ranks by inner product (== cosine). This "decode -> cosine" protocol keeps
ALL methods on one identical evaluation path (binary, int8, PCA, ITQ, PQ, OPQ),
so the only thing that changes across rows of the rate-distortion table is the
number of bits each method spends per template.

Bits are reported by `.bits` (set during `.fit`). For methods that work in a
reduced k-dim space the stored object is the k-dim code; for PQ/OPQ it is the
M*nbits code. We never count the (tiny, amortized) codebook / rotation / scale
overhead — state that assumption in the paper.

NOTE on PQ/OPQ: production systems search PQ codes with asymmetric distance
(ADC) which is faster and marginally different from "decode then cosine". We use
decode-then-cosine for a uniform, exact-on-the-decoded-vectors comparison. Worth
one sentence in the write-up.
"""
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:                                    # pragma: no cover
    _HAS_FAISS = False

from sklearn.decomposition import PCA


def l2norm(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class Compressor:
    name = "base"

    def fit(self, X):
        self.bits = 32 * X.shape[1]
        return self

    def approx_vectors(self, X):
        """Return decoded float32 vectors (NOT yet L2-normalized)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Precision-reduction (full dimensionality)
# --------------------------------------------------------------------------- #
class Float32(Compressor):
    name = "fp32"

    def fit(self, X):
        self.bits = 32 * X.shape[1]
        return self

    def approx_vectors(self, X):
        return np.asarray(X, dtype=np.float32)


class Float16(Compressor):
    name = "fp16"

    def fit(self, X):
        self.bits = 16 * X.shape[1]
        return self

    def approx_vectors(self, X):
        return np.asarray(X, dtype=np.float16).astype(np.float32)


class Int8(Compressor):
    """Per-vector symmetric int8. After L2-normalization at eval time the
    per-vector scale cancels, so this is exactly 'store 8-bit direction codes,
    compare by cosine'."""
    name = "int8"

    def fit(self, X):
        self.bits = 8 * X.shape[1]
        return self

    def approx_vectors(self, X):
        X = np.asarray(X, dtype=np.float32)
        scale = np.max(np.abs(X), axis=1, keepdims=True) + 1e-12
        q = np.round(X / scale * 127.0).clip(-127, 127)
        return (q * (scale / 127.0)).astype(np.float32)


class BinarySign(Compressor):
    """1 bit/dim sign quantization. IP of +/-1 vectors == (D - 2*Hamming)."""
    name = "binary"

    def fit(self, X):
        self.bits = 1 * X.shape[1]
        return self

    def approx_vectors(self, X):
        s = np.sign(np.asarray(X, dtype=np.float32))
        s[s == 0] = 1.0
        return s


# --------------------------------------------------------------------------- #
# Dimensionality reduction + (optional) precision reduction
# --------------------------------------------------------------------------- #
_SUBQUANT = {
    "fp32":   (lambda v: v.astype(np.float32),                          32),
    "fp16":   (lambda v: v.astype(np.float16).astype(np.float32),       16),
    "int8":   (None,                                                     8),   # handled inline
    "binary": (lambda v: np.where(v >= 0, 1.0, -1.0).astype(np.float32), 1),
}


class PCAReduce(Compressor):
    """PCA to k dims, then a per-dim sub-quantizer in {fp32,fp16,int8,binary}."""

    def __init__(self, k=256, sub="binary"):
        self.k = k
        self.sub = sub
        self.name = f"pca{k}-{sub}"

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        k = min(self.k, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=k, random_state=0).fit(X)
        self.k_eff = k
        self.bits = _SUBQUANT[self.sub][1] * k
        return self

    def approx_vectors(self, X):
        v = self.pca.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        if self.sub == "int8":
            scale = np.max(np.abs(v), axis=1, keepdims=True) + 1e-12
            q = np.round(v / scale * 127.0).clip(-127, 127)
            return (q * (scale / 127.0)).astype(np.float32)
        return _SUBQUANT[self.sub][0](v)


class ITQ(Compressor):
    """Iterative Quantization: PCA to k, learn an orthogonal rotation that
    minimizes the sign-quantization error, then take the sign. k bits/template.
    Should beat raw PCA->sign at the same bit budget."""

    def __init__(self, k=256, n_iter=50, seed=0):
        self.k = k
        self.n_iter = n_iter
        self.seed = seed
        self.name = f"itq{k}"

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        k = min(self.k, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=k, random_state=0).fit(X)
        V = self.pca.transform(X).astype(np.float64)
        rng = np.random.default_rng(self.seed)
        R = np.linalg.qr(rng.standard_normal((k, k)))[0]
        for _ in range(self.n_iter):
            Z = V @ R
            B = np.sign(Z)
            B[B == 0] = 1.0
            # orthogonal Procrustes: minimize ||V R - B|| -> R = U Vt, SVD(V^T B)
            U, _, Vt = np.linalg.svd(V.T @ B)
            R = U @ Vt
        self.R = R.astype(np.float32)
        self.k_eff = k
        self.bits = k
        return self

    def approx_vectors(self, X):
        V = self.pca.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        s = np.sign(V @ self.R)
        s[s == 0] = 1.0
        return s.astype(np.float32)


# --------------------------------------------------------------------------- #
# Product quantization (needs faiss)
# --------------------------------------------------------------------------- #
class PQ(Compressor):
    """Product Quantization via faiss standalone codec. M subvectors x nbits."""

    def __init__(self, m=32, nbits=8, opq=False):
        self.m = m
        self.nbits = nbits
        self.opq = opq
        self.name = f"{'opq' if opq else 'pq'}{m}x{nbits}"

    def fit(self, X, train_cap=50_000, opq_train_cap=20_000, opq_niter=10,
            seed=0):
        if not _HAS_FAISS:
            raise RuntimeError("PQ/OPQ require faiss-cpu (pip install faiss-cpu)")
        X = np.ascontiguousarray(X, dtype=np.float32)
        d = X.shape[1]
        pq = faiss.IndexPQ(d, self.m, self.nbits)
        if self.opq:
            opq = faiss.OPQMatrix(d, self.m)
            opq.niter = opq_niter                      # OPQ on CPU is the slow part
            self.codec = faiss.IndexPreTransform(opq, pq)
            cap = opq_train_cap
        else:
            self.codec = pq
            cap = train_cap
        # codebook training does not need the whole gallery; subsample.
        if len(X) > cap:
            idx = np.random.default_rng(seed).choice(len(X), cap, replace=False)
            Xtrain = np.ascontiguousarray(X[idx])
        else:
            Xtrain = X
        self.codec.train(Xtrain)
        self.bits = self.m * self.nbits
        return self

    def approx_vectors(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        codes = self.codec.sa_encode(X)
        return self.codec.sa_decode(codes).astype(np.float32)


# --------------------------------------------------------------------------- #
# Default sweep
# --------------------------------------------------------------------------- #
def default_zoo(k=256):
    """A representative rate-distortion sweep. Edit freely."""
    zoo = [
        Float32(),
        Float16(),
        Int8(),
        BinarySign(),
        PCAReduce(k, "fp32"),
        PCAReduce(k, "int8"),
        PCAReduce(k, "binary"),
        ITQ(k),
    ]
    if _HAS_FAISS:
        zoo += [PQ(m=64, nbits=8), PQ(m=32, nbits=8)]
        # OPQ is intentionally NOT in the default sweep: OPQ codebook training is
        # very slow on CPU at high dim (minutes on 1536-d). Opt in explicitly if
        # you want it, e.g. zoo += [PQ(32, 8, opq=True)] and expect a long fit.
    return zoo
