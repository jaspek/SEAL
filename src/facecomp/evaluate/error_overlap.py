"""Error-overlap across the three models + fusion ceilings, per dataset.

Extends the seminar's Table-3 analysis (LFW: 111/112 errors shared) to any
emb_<dataset>.npz. Overlap categories use one global best threshold per model
(same diagnostic as the seminar's error_overlap.py); the accuracy numbers use
the standard 10-fold protocol so they are comparable to the rate-distortion
tables."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold

from .verification import cosine_scores, evaluate_lfw

_GRID = np.arange(-1.0, 1.0, 0.005)
MODELS = ("arcface", "magface", "adaface")


def _best_global_mask(scores, labels):
    best_acc, best_thr = -1.0, 0.0
    for thr in _GRID:
        a = ((scores > thr) == labels).mean()
        if a > best_acc:
            best_acc, best_thr = a, thr
    return (scores > best_thr) == labels


def _kfold_acc(scores, labels):
    accs = []
    for tr, te in KFold(n_splits=10, shuffle=False).split(scores):
        best_acc, best_thr = -1.0, 0.0
        for thr in _GRID:
            a = ((scores[tr] > thr) == labels[tr]).mean()
            if a > best_acc:
                best_acc, best_thr = a, thr
        accs.append(((scores[te] > best_thr) == labels[te]).mean())
    return float(np.mean(accs))


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def analyze(per_model, pair_idx):
    """Full overlap + fusion analysis. Returns dict with 'summary' (flat scalars
    for CSV) and 'categories' (Table-3-style error breakdown)."""
    labels = pair_idx[:, 2]
    present = [m for m in MODELS if m in per_model]

    scores = {m: cosine_scores(per_model[m], pair_idx) for m in present}
    wrong = {m: ~_best_global_mask(scores[m], labels) for m in present}

    n = len(labels)
    z = np.zeros(n, dtype=bool)
    wa, wm, wd = (wrong.get(m, z) for m in MODELS)

    categories = {
        "arcface only": int((wa & ~wm & ~wd).sum()),
        "magface only": int((~wa & wm & ~wd).sum()),
        "adaface only": int((~wa & ~wm & wd).sum()),
        "arcface+magface": int((wa & wm & ~wd).sum()),
        "arcface+adaface": int((wa & ~wm & wd).sum()),
        "magface+adaface": int((~wa & wm & wd).sum()),
        "all three": int((wa & wm & wd).sum()),
    }
    wrong_all = wa & wm & wd
    shared_fr = int((wrong_all & (labels == 1)).sum())   # false rejects
    shared_fa = int((wrong_all & (labels == 0)).sum())   # false accepts

    votes = sum((~wrong[m]).astype(int) for m in present)
    majority = float((votes >= 2).mean())
    oracle = float((votes >= 1).mean())

    per_errors = {m: int(wrong[m].sum()) for m in present}
    mean_err = float(np.mean(list(per_errors.values())))
    shared_frac = categories["all three"] / mean_err if mean_err > 0 else float("nan")

    # accuracies under the proper 10-fold protocol
    single = {m: _kfold_acc(scores[m], labels) for m in present}
    concat = _l2(np.concatenate([per_model[m] for m in present], axis=1))
    concat_acc, _ = evaluate_lfw(concat, pair_idx)
    smean_acc = _kfold_acc(np.mean([scores[m] for m in present], axis=0), labels)
    best_single = max(single.values())

    summary = {
        "n_pairs": n,
        **{f"err_{m}": per_errors.get(m, 0) for m in MODELS},
        "err_shared": categories["all three"],
        "shared_frac": round(shared_frac, 4),
        "shared_false_rejects": shared_fr,
        "shared_false_accepts": shared_fa,
        **{f"acc_{m}": round(single.get(m, float("nan")), 6) for m in MODELS},
        "acc_best_single": round(best_single, 6),
        "acc_concat": round(concat_acc, 6),
        "acc_scoremean": round(smean_acc, 6),
        "fusion_gain": round(max(concat_acc, smean_acc) - best_single, 6),
        "acc_majority_vote": round(majority, 6),
        "acc_oracle": round(oracle, 6),
    }
    return {"summary": summary, "categories": categories}


def format_report(name, res):
    s, c = res["summary"], res["categories"]
    lines = [f"\n=== {name}: error overlap ({s['n_pairs']} pairs) ==="]
    for k, v in c.items():
        lines.append(f"  wrong by {k:18s} {v:6d}")
    lines.append(f"  shared errors = {s['shared_frac']:.1%} of a model's errors "
                 f"(FR {s['shared_false_rejects']} / FA {s['shared_false_accepts']})")
    lines.append(f"  acc: arc {s['acc_arcface']:.4f}  mag {s['acc_magface']:.4f}  "
                 f"ada {s['acc_adaface']:.4f}")
    lines.append(f"  fusion: concat {s['acc_concat']:.4f}  score-mean {s['acc_scoremean']:.4f}  "
                 f"gain vs best single {s['fusion_gain']:+.4f}")
    lines.append(f"  ceilings: majority {s['acc_majority_vote']:.4f}  oracle {s['acc_oracle']:.4f}")
    return "\n".join(lines)
