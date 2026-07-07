"""TurboQuant-style data-oblivious quantization (Zandieh et al., 2025,
arXiv:2504.19874), practical version.

No calibration data: a seeded random rotation concentrates coordinates, then a
per-coordinate uniform scalar quantizer is applied. The two-stage variant adds a
1-bit residual code for an ~unbiased inner-product estimate. Serves as the
data-oblivious CONTROL against the data-fit PCA/ITQ/RaBitQ reducers — it exposes
`.transform` and `.k`, so it plugs into the existing cosine path exactly like the
other reducers:  T.compress(X, method="fp32", dim_reducer=TurboQuant(bits=4).fit(X))
(bits accounted separately: bits/template = dim * bits, + one float32 scale)."""
from __future__ import annotations
import numpy as np


def _l2(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


class TurboQuant:
    def __init__(self, bits=4, seed=0, two_stage=False):
        self.bits, self.seed, self.two_stage = bits, seed, two_stage
        self.R = None

    def fit(self, X):
        d = X.shape[1]
        rng = np.random.default_rng(self.seed)
        R, _ = np.linalg.qr(rng.standard_normal((d, d)))      # random rotation
        self.R = R.astype(np.float32)
        return self

    def _rotate(self, X):
        return _l2(X) @ self.R

    def transform(self, X):
        """Decode back to a float vector so it plugs into the cosine path."""
        v = self._rotate(X)
        d = v.shape[1]
        rng = 3.0 / np.sqrt(d)                # +/-3 std of the concentrated coords
        lo, hi = -rng, rng
        levels = 2 ** self.bits
        step = (hi - lo) / (levels - 1)
        q = np.clip(np.round((v - lo) / step), 0, levels - 1)
        deq = q * step + lo
        if self.two_stage:                    # 1-bit residual -> less biased IP
            deq = deq + np.sign(v - deq) * (step / 4.0)
        return _l2(deq.astype(np.float32))

    @property
    def k(self):
        return self.R.shape[0] if self.R is not None else None

    def bits_per_template(self):
        return (self.k or 0) * self.bits + 32   # code + one float32 scale
