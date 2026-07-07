"""Learned score-level fusion: can a weighted combination exploit the error
diversity that naive score-mean cannot?

Per fold (standard 10-fold, thresholds/weights fit on the 9 train folds only):
  best-single : the model with the highest TRAIN accuracy, applied to test
  score-mean  : unweighted mean of the three cosine scores
  learned (LR): logistic regression over [s_arc, s_mag, s_ada]
  learned_aug : LR over scores + pairwise |disagreements| + products + min/max
  mlp         : small MLP over the three scores

learned_aug/mlp test the paper's claim that the error diversity is "not
accessible from scores alone": disagreement features + nonlinearity are exactly
what a score-level gate would exploit, so if they still cannot beat best-single
on hard data the claim gets materially stronger.

  python experiments/run_learned_fusion.py --buffalo
Writes outputs/results/learned_fusion_summary.csv"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.evaluate import significance as S
from facecomp.evaluate.verification import cosine_scores

_GRID = np.arange(-1.0, 1.0, 0.005)


def _best_thr(scores, labels):
    best_acc, best_thr = -1.0, 0.0
    for thr in _GRID:
        a = ((scores > thr) == labels).mean()
        if a > best_acc:
            best_acc, best_thr = a, thr
    return best_thr, best_acc


def _augment(M):
    """scores + pairwise |disagreements| + products + min/max — the features a
    score-level gate would use to route/veto."""
    cols = [M]
    kk = M.shape[1]
    for i in range(kk):
        for j in range(i + 1, kk):
            cols.append(np.abs(M[:, i:i + 1] - M[:, j:j + 1]))
            cols.append(M[:, i:i + 1] * M[:, j:j + 1])
    cols += [M.min(axis=1, keepdims=True), M.max(axis=1, keepdims=True)]
    return np.hstack(cols)


def analyze_file(path):
    name = Path(path).stem
    for pre, post in (("emb_", ""), ("_embeddings", "")):
        name = name.replace(pre, post)
    b = E.load_emb(path)
    if b.pair_idx is None:
        print(f"SKIP {name}: no pair_idx")
        return None
    labels = b.pair_idx[:, 2]
    models = [m for m in E.MODELS if m in b.per_model]
    s = {m: cosine_scores(b.per_model[m], b.pair_idx) for m in models}
    X = np.stack([s[m] for m in models], axis=1)

    n = len(labels)
    keys = ("best_single", "score_mean", "learned", "learned_aug", "mlp")
    correct = {k: np.zeros(n, bool) for k in keys}
    weights = []
    for tr, te in KFold(n_splits=10, shuffle=False).split(labels):
        # fold-honest best single: pick the model by TRAIN accuracy
        train_accs = {}
        thrs = {}
        for m in models:
            thr, acc = _best_thr(s[m][tr], labels[tr])
            thrs[m], train_accs[m] = thr, acc
        best = max(train_accs, key=train_accs.get)
        correct["best_single"][te] = (s[best][te] > thrs[best]) == labels[te]

        mean_scores = X.mean(axis=1)
        thr, _ = _best_thr(mean_scores[tr], labels[tr])
        correct["score_mean"][te] = (mean_scores[te] > thr) == labels[te]

        lr = LogisticRegression(max_iter=1000).fit(X[tr], labels[tr])
        correct["learned"][te] = lr.predict(X[te]) == labels[te]
        weights.append(lr.coef_[0])

        # nonlinear / disagreement-feature variants
        lr_a = LogisticRegression(max_iter=2000).fit(_augment(X[tr]), labels[tr])
        correct["learned_aug"][te] = lr_a.predict(_augment(X[te])) == labels[te]

        mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000,
                            random_state=0).fit(X[tr], labels[tr])
        correct["mlp"][te] = mlp.predict(X[te]) == labels[te]

    w = np.mean(weights, axis=0)
    w = w / (np.abs(w).sum() + 1e-12)
    accs = {k: float(v.mean()) for k, v in correct.items()}
    _, _, p_vs_single = S.mcnemar_exact(correct["learned"], correct["best_single"])
    _, _, p_vs_mean = S.mcnemar_exact(correct["learned"], correct["score_mean"])
    _, _, p_aug = S.mcnemar_exact(correct["learned_aug"], correct["best_single"])
    _, _, p_mlp = S.mcnemar_exact(correct["mlp"], correct["best_single"])

    row = {"dataset": name,
           **{f"acc_{k}": round(v, 6) for k, v in accs.items()},
           # original columns (kept verbatim so make_tables.py still works):
           "gain_vs_best_single": round(accs["learned"] - accs["best_single"], 6),
           "gain_vs_score_mean": round(accs["learned"] - accs["score_mean"], 6),
           "p_vs_best_single": p_vs_single, "p_vs_score_mean": p_vs_mean,
           # new nonlinear / disagreement-feature columns:
           "gain_aug_vs_single": round(accs["learned_aug"] - accs["best_single"], 6),
           "gain_mlp_vs_single": round(accs["mlp"] - accs["best_single"], 6),
           "p_aug_vs_single": p_aug, "p_mlp_vs_single": p_mlp,
           **{f"w_{m}": round(float(wi), 3) for m, wi in zip(models, w)}}
    print(f"{name:>10}: single {accs['best_single']:.4f}  mean {accs['score_mean']:.4f}  "
          f"learned {accs['learned']:.4f}  aug {accs['learned_aug']:.4f}  "
          f"mlp {accs['mlp']:.4f}  (gain_mlp {row['gain_mlp_vs_single']:+.4f} p={p_mlp:.2g})")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*", default=None)
    ap.add_argument("--buffalo", action="store_true")
    ap.add_argument("--out", default="learned_fusion_summary.csv")
    args = ap.parse_args()

    paths = [Path(p) for p in (args.emb or [])]
    if args.buffalo:
        paths += [C.EMBEDDINGS_DIR / "lfw_embeddings.npz"] + \
                 [C.EMBEDDINGS_DIR / f"emb_{d}.npz" for d in
                  ("sllfw", "cplfw", "xqlfw", "calfw", "cfp_fp")]
    if not paths:
        ap.error("give --emb <files> or --buffalo")

    rows = [r for p in paths if (r := analyze_file(p)) is not None]
    C.ensure_dirs()
    out = Path(args.out)
    out = out if out.is_absolute() else C.RESULTS_DIR / out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
