"""Per-input gated fusion using MagFace/AdaFace feature magnitude as a FREE
quality signal (the paper cites this property but normalizes it away before
saving; build_embeddings now stores <model>_norm arrays). Fold-honest: the gate
rule is applied on test folds only, compared against best-single and the
fold-honest oracle ceiling.

Gates evaluated:
  qgate    : route to the model whose pair has the highest quality proxy
  maxscore : route to the model with the highest |cosine| (calibration-free)
Ceilings:
  oracle   : right if ANY model is right at its own fold threshold

  python experiments/run_quality_gate.py --buffalo
Writes outputs/results/quality_gate.csv"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.evaluate.verification import cosine_scores

_GRID = np.arange(-1.0, 1.0, 0.005)


def _thr(s, y):
    best_a, best_t = -1.0, 0.0
    for t in _GRID:
        a = ((s > t) == y).mean()
        if a > best_a:
            best_a, best_t = a, t
    return best_t


def _l2(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def analyze(path):
    d = np.load(path, allow_pickle=True)
    if "pair_idx" not in d.files:
        return None
    name = Path(path).stem.replace("emb_", "").replace("_embeddings", "")
    pair = d["pair_idx"].astype(int)
    y = pair[:, 2]
    models = [m for m in E.MODELS if m in d.files]
    S = {m: cosine_scores(_l2(d[m].astype(np.float32)), pair) for m in models}
    # per-pair quality = min of the two images' magnitudes (missing -> ones)
    Q = {}
    for m in models:
        nm = f"{m}_norm"
        q = d[nm].astype(np.float32) if nm in d.files else np.ones(d[m].shape[0], np.float32)
        Q[m] = np.minimum(q[pair[:, 0]], q[pair[:, 1]])
    have_norm = any(f"{m}_norm" in d.files for m in models)
    X = np.stack([S[m] for m in models], 1)

    keys = ("best", "qgate", "maxscore", "oracle")
    correct = {k: np.zeros(len(y), bool) for k in keys}
    for tr, te in KFold(10, shuffle=False).split(y):
        accs = {m: ((S[m][tr] > _thr(S[m][tr], y[tr])) == y[tr]).mean() for m in models}
        best = max(accs, key=accs.get)
        thr = {m: _thr(S[m][tr], y[tr]) for m in models}
        correct["best"][te] = (S[best][te] > thr[best]) == y[te]
        # quality-routed gate
        Qte = np.stack([Q[m][te] for m in models], 1)
        pick = np.argmax(Qte, 1)
        pred_q = np.array([S[models[pick[i]]][te][i] > thr[models[pick[i]]]
                           for i in range(len(te))])
        correct["qgate"][te] = pred_q == y[te]
        # max-|score| routed gate (calibration-free baseline)
        pickm = np.argmax(np.abs(X[te]), 1)
        pred_m = np.array([X[te][i, pickm[i]] > thr[models[pickm[i]]]
                           for i in range(len(te))])
        correct["maxscore"][te] = pred_m == y[te]
        # fold-honest oracle
        anyright = np.zeros(len(te), bool)
        for m in models:
            anyright |= ((S[m][te] > thr[m]) == y[te])
        correct["oracle"][te] = anyright

    row = {"dataset": name, "has_norm": have_norm,
           **{f"acc_{k}": round(float(v.mean()), 5) for k, v in correct.items()}}
    row["qgate_gain_vs_best"] = round(row["acc_qgate"] - row["acc_best"], 5)
    row["oracle_headroom"] = round(row["acc_oracle"] - row["acc_best"], 5)
    print(f"  {name:8s} best {row['acc_best']:.4f}  qgate {row['acc_qgate']:.4f} "
          f"({row['qgate_gain_vs_best']:+.4f})  maxscore {row['acc_maxscore']:.4f}  "
          f"oracle {row['acc_oracle']:.4f}  norm={have_norm}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*", default=None)
    ap.add_argument("--buffalo", action="store_true")
    ap.add_argument("--out", default="quality_gate.csv")
    args = ap.parse_args()
    paths = [Path(p) for p in (args.emb or [])]
    if args.buffalo:
        paths += [C.EMBEDDINGS_DIR / f"emb_{x}.npz" for x in
                  ("lfw", "sllfw", "cplfw", "xqlfw", "calfw", "cfp_fp")]
    if not paths:
        ap.error("give --emb <files> or --buffalo")
    rows = [r for p in paths if (r := analyze(p)) is not None]
    C.ensure_dirs()
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / args.out, index=False)
    print("wrote", C.RESULTS_DIR / args.out)


if __name__ == "__main__":
    main()
