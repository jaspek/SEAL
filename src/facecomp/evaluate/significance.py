"""Statistical tests for paired accuracy comparisons under the 10-fold protocol.

McNemar (exact binomial) conditions on the discordant pairs only — ideal here,
since methods agree on the vast majority of pairs (Dietterich 1998).
Nadeau-Bengio corrects the paired fold t-test for the variance inflation of
overlapping cross-validation training sets (Nadeau & Bengio 2003); for k-fold
the test/train ratio is 1/(k-1)."""
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold

_GRID = np.arange(-1.0, 1.0, 0.005)


def kfold_eval(scores, labels, n_splits=10):
    """Per-pair correctness under the fold-tuned-threshold protocol.
    Returns (correct bool array over all pairs, per-fold accuracies)."""
    correct = np.zeros(len(labels), dtype=bool)
    fold_accs = []
    for tr, te in KFold(n_splits=n_splits, shuffle=False).split(scores):
        best_acc, best_thr = -1.0, 0.0
        for thr in _GRID:
            a = ((scores[tr] > thr) == labels[tr]).mean()
            if a > best_acc:
                best_acc, best_thr = a, thr
        c = (scores[te] > best_thr) == labels[te]
        correct[te] = c
        fold_accs.append(float(c.mean()))
    return correct, np.array(fold_accs)


def mcnemar_exact(correct_a, correct_b):
    """Exact binomial McNemar. Returns (b, c, p): b = A right & B wrong,
    c = A wrong & B right."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    p = 1.0 if n == 0 else float(stats.binomtest(min(b, c), n, 0.5).pvalue)
    return b, c, p


def nadeau_bengio(fold_accs_a, fold_accs_b, test_train_ratio=None):
    """Corrected resampled paired t-test over fold accuracies. Returns (t, p)."""
    d = np.asarray(fold_accs_a, float) - np.asarray(fold_accs_b, float)
    k = len(d)
    if test_train_ratio is None:
        test_train_ratio = 1.0 / (k - 1)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / k + test_train_ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)
