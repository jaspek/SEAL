"""Threshold portability: the paper re-tunes the decision threshold per benchmark
AND per compression, silently absorbing any score-distribution shift a compressor
causes. Here we tune the threshold ONCE on the fp32 template (per fold) and apply
that frozen threshold to each compressed variant -- the deployment reality where
one operating point is calibrated once and reused.

Reports frozen-threshold accuracy vs the paper's re-tuned accuracy (the
portability gap), plus the resulting FMR/FNMR drift.

  python experiments/run_threshold_portability.py --buffalo
Writes outputs/results/threshold_portability.csv"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.compress import template as T
from facecomp.evaluate.verification import cosine_scores

_GRID = np.arange(-1.0, 1.0, 0.005)
SPECS = [("none", None, "int8"), ("none", None, "binary"),
         ("itq", 256, "binary"), ("itq", 128, "binary")]


def _fold_thr(scores, labels):
    best_a, best_t = -1.0, 0.0
    for t in _GRID:
        a = ((scores > t) == labels).mean()
        if a > best_a:
            best_a, best_t = a, t
    return best_t


def analyze(path):
    b = E.load_emb(path)
    if b.pair_idx is None:
        return []
    name = Path(path).stem.replace("emb_", "").replace("_embeddings", "")
    fused = b.fuse()
    labels = b.pair_idx[:, 2]
    s_fp = cosine_scores(fused, b.pair_idx)
    rows = []
    print(f"\n=== {name} ===")
    for kind, k, prec in SPECS:
        red = None if kind == "none" else (T.PCAReducer(k) if kind == "pca"
                                           else T.ITQReducer(k)).fit(fused)
        comp, _ = T.compress(fused, prec, red)
        s_c = cosine_scores(comp, b.pair_idx)
        frozen_acc, retuned_acc, fmr, fnmr = [], [], [], []
        for tr, te in KFold(10, shuffle=False).split(labels):
            t_fp = _fold_thr(s_fp[tr], labels[tr])          # frozen: tuned on fp32 scores
            t_re = _fold_thr(s_c[tr], labels[tr])           # paper: re-tuned per variant
            pred_frozen = s_c[te] > t_fp
            frozen_acc.append((pred_frozen == labels[te]).mean())
            retuned_acc.append(((s_c[te] > t_re) == labels[te]).mean())
            imp, gen = labels[te] == 0, labels[te] == 1
            fmr.append(pred_frozen[imp].mean() if imp.any() else np.nan)     # false match
            fnmr.append((~pred_frozen[gen]).mean() if gen.any() else np.nan)  # false non-match
        tag = prec if kind == "none" else f"{kind}{k}-{prec}"
        row = {"dataset": name, "variant": tag,
               "acc_frozen_fp32thr": round(float(np.mean(frozen_acc)), 5),
               "acc_retuned": round(float(np.mean(retuned_acc)), 5),
               "portability_gap": round(float(np.mean(retuned_acc) - np.mean(frozen_acc)), 5),
               "fmr": round(float(np.nanmean(fmr)), 5),
               "fnmr": round(float(np.nanmean(fnmr)), 5)}
        rows.append(row)
        print(f"  {tag:16s} frozen {row['acc_frozen_fp32thr']:.4f}  "
              f"retuned {row['acc_retuned']:.4f}  gap {row['portability_gap']:+.4f}  "
              f"(FMR {row['fmr']:.4f} / FNMR {row['fnmr']:.4f})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*", default=None)
    ap.add_argument("--buffalo", action="store_true")
    ap.add_argument("--out", default="threshold_portability.csv")
    args = ap.parse_args()
    paths = [Path(p) for p in (args.emb or [])]
    if args.buffalo:
        paths += [C.EMBEDDINGS_DIR / "lfw_embeddings.npz"] + \
                 [C.EMBEDDINGS_DIR / f"emb_{d}.npz" for d in
                  ("sllfw", "cplfw", "xqlfw", "calfw", "cfp_fp")]
    if not paths:
        ap.error("give --emb <files> or --buffalo")
    rows = [r for p in paths for r in analyze(p)]
    C.ensure_dirs()
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / args.out, index=False)
    print("\nwrote", C.RESULTS_DIR / args.out)


if __name__ == "__main__":
    main()
