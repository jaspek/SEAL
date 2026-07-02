"""RaBitQ-style binary templates (Gao & Long, SIGMOD 2024), practical version.

Encode: project to k dims with a seeded random orthonormal map (data-independent,
unlike PCA/ITQ), L2-normalize, take sign codes x = sign(v)/sqrt(k), and store the
per-vector correction c = <v, x>  (how well the code aligns with the vector).

Score:
  asymmetric (stored template quantized, probe full precision — the paper's
  setting):     <probe_rot, x> / c
  symmetric (both templates quantized):  <x_a, x_b> / (c_a * c_b)

Storage per template: k bits + one float32 for c  ->  k + 32 bits.
An optional PCA front-end packs signal into the k dims first (Extended-RaBitQ
style low-bit budgets)."""
from __future__ import annotations
import numpy as np

from .template import PCAReducer


def _l2(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


class RaBitQ:
    def __init__(self, k, seed=0, pca_front=False):
        self.k, self.seed, self.pca_front = k, seed, pca_front
        self.pca = None

    def fit(self, X):
        d = X.shape[1]
        if self.pca_front:
            self.pca = PCAReducer(self.k, seed=self.seed).fit(X)
            d = self.k
        rng = np.random.default_rng(self.seed)
        q, _ = np.linalg.qr(rng.standard_normal((d, self.k)).astype(np.float64))
        self.P = q.astype(np.float32)              # (d, k), orthonormal columns
        return self

    def rotate(self, X):
        if self.pca is not None:
            X = self.pca.transform(X)
        return _l2(X @ self.P)

    def encode(self, X):
        """Returns (codes (N,k) as +/-1/sqrt(k) float32, corrections c (N,))."""
        v = self.rotate(X)
        codes = np.where(v >= 0, 1.0, -1.0).astype(np.float32) / np.sqrt(self.k)
        c = np.einsum("ij,ij->i", v, codes).astype(np.float32)
        c = np.clip(c, 1e-6, None)
        return codes, c

    def bits(self):
        return self.k + 32                          # code + float32 correction

    def score_sym(self, codes, c, pair_idx):
        a, b = pair_idx[:, 0], pair_idx[:, 1]
        return np.einsum("ij,ij->i", codes[a], codes[b]) / (c[a] * c[b])

    def gallery_effective(self, codes, c):
        """codes/c so a plain dot with a rotated full-precision probe gives the
        asymmetric RaBitQ estimate — plugs into the existing 1:N ranking code."""
        return codes / c[:, None]
