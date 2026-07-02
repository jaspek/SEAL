"""1:1 verification: LFW 10-fold accuracy + EER / TAR@FAR. emb assumed L2-normalized."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold


def cosine_scores(emb, pair_idx):
    return np.sum(emb[pair_idx[:, 0]] * emb[pair_idx[:, 1]], axis=1)


def evaluate_lfw(emb, pair_idx, thr_grid=None):
    if thr_grid is None:
        thr_grid = np.arange(-1.0, 1.0, 0.005)
    scores, labels = cosine_scores(emb, pair_idx), pair_idx[:, 2]
    accs = []
    for tr, te in KFold(n_splits=10, shuffle=False).split(scores):
        best_acc, best_thr = -1.0, 0.0
        for thr in thr_grid:
            a = ((scores[tr] > thr) == labels[tr]).mean()
            if a > best_acc:
                best_acc, best_thr = a, thr
        accs.append(((scores[te] > best_thr) == labels[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def roc_metrics(emb, pair_idx, fars=(1e-2, 1e-3)):
    """Pooled-pair operating-point metrics. With ~3000 imposters per benchmark,
    FAR=1e-2 is the reliable point; FAR=1e-3 rests on ~3 imposters — report it
    with that caveat. Keys are CSV-safe: eer, tar_far0.01, tar_far0.001."""
    from sklearn.metrics import roc_curve
    scores, labels = cosine_scores(emb, pair_idx), pair_idx[:, 2]
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    out = {"eer": round(float((fpr[i] + fnr[i]) / 2), 6)}
    for far in fars:
        j = max(np.searchsorted(fpr, far, side="right") - 1, 0)
        out[f"tar_far{far:g}"] = round(float(tpr[j]), 6)
    return out
