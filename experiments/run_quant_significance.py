"""Significance of NETWORK quantization: fp32 backbone vs static-INT8 backbone.

Pairs each dataset's two builds from the same .bin (emb_<d>.npz vs
emb_<d>_q8.npz; LFW uses emb_lfw_bin.npz so both sides share the official
crops), scores the full-precision template, and runs the same exact-McNemar +
Nadeau-Bengio protocol as run_significance.py. Tested per dataset: the
quantized model alone (arcface) and the fused template containing it.

  python experiments/run_quant_significance.py                 # all six datasets
  python experiments/run_quant_significance.py --datasets xqlfw cplfw

Writes outputs/results/quant_significance.csv. A = fp32 net, B = int8 net;
positive diff means the fp32 network is better."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.evaluate import significance as S
from facecomp.evaluate.verification import cosine_scores

BASELINE = {"lfw": "emb_lfw_bin.npz"}   # official-crop fp32 baseline, not the re-aligned legacy npz
DATASETS = ["lfw", "sllfw", "cplfw", "xqlfw", "calfw", "cfp_fp"]


def compare(dataset):
    pa = C.EMBEDDINGS_DIR / BASELINE.get(dataset, f"emb_{dataset}.npz")
    pb = C.EMBEDDINGS_DIR / f"emb_{dataset}_q8.npz"
    for p in (pa, pb):
        if not p.exists():
            print(f"SKIP {dataset}: {p.name} not found")
            return []
    a, b = E.load_emb(pa), E.load_emb(pb)
    if a.pair_idx is None or b.pair_idx is None or not np.array_equal(a.pair_idx, b.pair_idx):
        print(f"SKIP {dataset}: pair lists differ between the two bundles")
        return []
    labels = a.pair_idx[:, 2]

    rows = []
    print(f"\n=== {dataset} ===")
    for config in ("arcface", "fused"):
        if config == "arcface":
            sa = cosine_scores(a.per_model["arcface"], a.pair_idx)
            sb = cosine_scores(b.per_model["arcface"], b.pair_idx)
        else:
            sa = cosine_scores(a.fuse(), a.pair_idx)
            sb = cosine_scores(b.fuse(), b.pair_idx)
        ca, fa = S.kfold_eval(sa, labels)
        cb, fb = S.kfold_eval(sb, labels)
        d_b, d_c, p_mc = S.mcnemar_exact(ca, cb)
        t_nb, p_nb = S.nadeau_bengio(fa, fb)
        row = {"dataset": dataset, "config": config,
               "acc_fp32net": round(float(ca.mean()), 6),
               "acc_int8net": round(float(cb.mean()), 6),
               "diff": round(float(ca.mean() - cb.mean()), 6),
               "discordant_fp32_wins": d_b, "discordant_int8_wins": d_c,
               "p_mcnemar": p_mc, "t_nb": round(t_nb, 3), "p_nb": p_nb}
        rows.append(row)
        mark = "**" if (p_mc < 0.05 and p_nb < 0.05) else ("* " if p_mc < 0.05 else "  ")
        print(f"  {mark}{config:>8}: fp32 {row['acc_fp32net']:.4f} vs int8 "
              f"{row['acc_int8net']:.4f}  diff {row['diff']:+.4f}  "
              f"McNemar p={p_mc:.2g} ({d_b}/{d_c})  NB p={p_nb:.2g}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--out", default="quant_significance.csv")
    args = ap.parse_args()

    rows = [r for d in args.datasets for r in compare(d)]
    C.ensure_dirs()
    out = Path(args.out)
    out = out if out.is_absolute() else C.RESULTS_DIR / out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}  (** = both tests <0.05, * = McNemar only)")


if __name__ == "__main__":
    main()
