"""Learned score-level fusion: can a weighted combination exploit the error
diversity that naive score-mean cannot?

Per fold (standard 10-fold, thresholds/weights fit on the 9 train folds only):
  best-single : the model with the highest TRAIN accuracy, applied to test
  score-mean  : unweighted mean of the three cosine scores
  learned (LR): logistic regression over [s_arc, s_mag, s_ada]

  python experiments/run_learned_fusion.py --buffalo
Writes outputs/results/learned_fusion_summary.csv"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    correct = {k: np.zeros(n, bool) for k in ("best_single", "score_mean", "learned")}
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

    w = np.mean(weights, axis=0)
    w = w / (np.abs(w).sum() + 1e-12)
    accs = {k: float(v.mean()) for k, v in correct.items()}
    _, _, p_vs_single = S.mcnemar_exact(correct["learned"], correct["best_single"])
    _, _, p_vs_mean = S.mcnemar_exact(correct["learned"], correct["score_mean"])

    row = {"dataset": name,
           **{f"acc_{k}": round(v, 6) for k, v in accs.items()},
           "gain_vs_best_single": round(accs["learned"] - accs["best_single"], 6),
           "gain_vs_score_mean": round(accs["learned"] - accs["score_mean"], 6),
           "p_vs_best_single": p_vs_single, "p_vs_score_mean": p_vs_mean,
           **{f"w_{m}": round(float(wi), 3) for m, wi in zip(models, w)}}
    print(f"{name:>10}: single {accs['best_single']:.4f}  mean {accs['score_mean']:.4f}  "
          f"learned {accs['learned']:.4f}  gain {row['gain_vs_best_single']:+.4f} "
          f"(p={p_vs_single:.2g})  weights arc/mag/ada = "
          + "/".join(f"{x:+.2f}" for x in w))
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
