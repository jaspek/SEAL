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


def tar_at_far(scores, labels, far):
    """TAR at a target FAR from pooled pair scores/labels (labels: 1=genuine)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    j = max(np.searchsorted(fpr, far, side="right") - 1, 0)
    return float(tpr[j])


def paired_tar_bootstrap(scores_a, scores_b, labels, far=0.01, n_boot=1000, seed=0):
    """Paired pair-level bootstrap CI on TAR@FAR(A) - TAR@FAR(B). Pure numpy on
    cached score vectors — the uncertainty the paper's TAR claims currently lack.
    Returns delta, 95% CI, and a two-sided bootstrap p-value for delta != 0."""
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(labels)
    base = tar_at_far(scores_a, labels, far) - tar_at_far(scores_b, labels, far)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        la = labels[idx]
        d[i] = tar_at_far(scores_a[idx], la, far) - tar_at_far(scores_b[idx], la, far)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return {"delta": round(base, 5), "ci_lo": round(float(lo), 5),
            "ci_hi": round(float(hi), 5), "p_boot": round(float(min(p, 1.0)), 5)}
