"""Apply Holm (FWER) + Benjamini-Hochberg (FDR) correction to the p-values already
written by run_significance / run_quant_significance / run_fusion_ablation. The
paper runs dozens of McNemar tests (it flags the 18 leave-one-out tests) but
applies no multiplicity control.

  python experiments/correct_pvalues.py
Writes outputs/results/pvalues_corrected.csv and prints how many survive Holm."""
import argparse

import numpy as np
import pandas as pd

from facecomp import config as C

# (csv filename, explicit p-value column or None to auto-detect p* columns)
FAMILIES = [
    ("significance_summary.csv", "p_mcnemar"),
    ("significance_summary.csv", "p_nb"),
    ("quant_significance.csv", "p_mcnemar"),
    ("quant_significance.csv", "p_nb"),
    ("fusion_ablation_significance.csv", None),
]


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pvalues_corrected.csv")
    args = ap.parse_args()

    out_rows = []
    seen = {}
    for fname, pcol in FAMILIES:
        path = C.RESULTS_DIR / fname
        if not path.exists():
            print(f"skip {fname} (not found)")
            continue
        df = seen.get(fname)
        if df is None:
            df = pd.read_csv(path)
            df["source_csv"] = fname
            seen[fname] = df
        pcols = [pcol] if pcol else [c for c in df.columns
                                     if c.lower().startswith("p") and
                                     pd.api.types.is_numeric_dtype(df[c])]
        for pc in pcols:
            if pc not in df.columns:
                continue
            fam = df[pc].dropna()
            if fam.empty:
                continue
            df.loc[fam.index, f"{pc}_holm"] = holm(fam.values)
            df.loc[fam.index, f"{pc}_bh"] = bh(fam.values)
            n_raw = int((fam < 0.05).sum())
            n_holm = int((df.loc[fam.index, f"{pc}_holm"] < 0.05).sum())
            n_bh = int((df.loc[fam.index, f"{pc}_bh"] < 0.05).sum())
            print(f"  {fname}:{pc}  n={len(fam)}  raw<.05={n_raw}  "
                  f"Holm={n_holm}  BH={n_bh}")

    if seen:
        pd.concat(seen.values(), ignore_index=True).to_csv(
            C.RESULTS_DIR / args.out, index=False)
        print("\nwrote", C.RESULTS_DIR / args.out)
    else:
        print("no p-value CSVs found in", C.RESULTS_DIR)


if __name__ == "__main__":
    main()
