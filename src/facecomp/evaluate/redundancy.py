"""Redundancy metrics from the eigenspectrum of the embedding covariance.

effective_rank  exp of the spectral entropy (Roy & Vetterli 2007) — "how many
                dimensions does this embedding effectively use"
participation_ratio  (sum lam)^2 / sum lam^2 — alternative spread measure
k95 / k99       components needed for 95% / 99% of the variance

Computed via the DxD covariance (never the NxN Gram), so it stays cheap even
for the 161k-image TinyFace matrix."""
from __future__ import annotations
import numpy as np


def spectrum_metrics(X):
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc).astype(np.float64) / max(n - 1, 1)
    lam = np.linalg.eigvalsh(cov)[::-1]
    lam = np.clip(lam, 0.0, None)
    tot = lam.sum()
    p = lam / tot
    pr = float(tot ** 2 / (lam ** 2).sum())
    erank = float(np.exp(-(p * np.log(p + 1e-300)).sum()))
    cum = np.cumsum(p)
    return {
        "effective_rank": round(erank, 1),
        "participation_ratio": round(pr, 1),
        "k95": int(np.searchsorted(cum, 0.95) + 1),
        "k99": int(np.searchsorted(cum, 0.99) + 1),
        "dim": d,
    }
