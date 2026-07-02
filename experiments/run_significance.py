"""Significance tests for the headline comparisons, per 1:1 dataset:
  compression vs fp32, ITQ vs PCA at equal bits, and model vs model.

  python experiments/run_significance.py --emb outputs/embeddings/emb_xqlfw.npz
  python experiments/run_significance.py --buffalo     # the six main datasets

Writes outputs/results/significance_summary.csv. Read `p_mcnemar`/`p_nb`:
both < 0.05 = solid; only McNemar < 0.05 = pair-level real but fold-fragile."""
import argparse
from pathlib import Path

import pandas as pd

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.compress import template as T
from facecomp.evaluate import significance as S
from facecomp.evaluate.verification import cosine_scores

SPECS = [("none", None, "fp32"), ("none", None, "int8"), ("none", None, "binary"),
         ("pca", 128, "binary"), ("itq", 128, "binary"),
         ("pca", 256, "binary"), ("itq", 256, "binary")]

COMPARISONS = [                      # (A, B) -> tests A vs B
    ("int8-1536", "fp32-1536"),
    ("binary-1536", "fp32-1536"),
    ("itq256-binary", "fp32-1536"),  # the 256-bit-template cost claim
    ("itq128-binary", "fp32-1536"),
    ("itq128-binary", "pca128-binary"),
    ("itq256-binary", "pca256-binary"),
    ("arcface", "magface"),
    ("arcface", "adaface"),
    ("adaface", "magface"),
]


def dataset_scores(b):
    fused = b.fuse()
    scores, reducers = {}, {}
    for kind, k, prec in SPECS:
        red = None
        if kind != "none":
            if (kind, k) not in reducers:
                cls = T.PCAReducer if kind == "pca" else T.ITQReducer
                reducers[(kind, k)] = cls(k).fit(fused)
            red = reducers[(kind, k)]
        comp, _ = T.compress(fused, method=prec, dim_reducer=red)
        name = f"{prec}-1536" if kind == "none" else f"{kind}{k}-binary"
        scores[name] = cosine_scores(comp, b.pair_idx)
    for m in E.MODELS:
        if m in b.per_model:
            scores[m] = cosine_scores(b.per_model[m], b.pair_idx)
    return scores


def analyze_file(path):
    name = Path(path).stem
    for pre, post in (("emb_", ""), ("_embeddings", "")):
        name = name.replace(pre, post)
    b = E.load_emb(path)
    if b.pair_idx is None:
        print(f"SKIP {name}: no pair_idx (1:N dataset)")
        return []
    labels = b.pair_idx[:, 2]
    scores = dataset_scores(b)
    evals = {k: S.kfold_eval(v, labels) for k, v in scores.items()}

    rows = []
    print(f"\n=== {name} ===")
    for a, bb in COMPARISONS:
        if a not in evals or bb not in evals:
            continue
        (ca, fa), (cb, fb) = evals[a], evals[bb]
        d_b, d_c, p_mc = S.mcnemar_exact(ca, cb)
        t_nb, p_nb = S.nadeau_bengio(fa, fb)
        row = {"dataset": name, "A": a, "B": bb,
               "acc_A": round(float(ca.mean()), 6), "acc_B": round(float(cb.mean()), 6),
               "diff": round(float(ca.mean() - cb.mean()), 6),
               "discordant_A_wins": d_b, "discordant_B_wins": d_c,
               "p_mcnemar": p_mc, "t_nb": round(t_nb, 3), "p_nb": p_nb}
        rows.append(row)
        mark = "**" if (p_mc < 0.05 and p_nb < 0.05) else ("* " if p_mc < 0.05 else "  ")
        print(f"  {mark}{a:>14} vs {bb:<14} diff {row['diff']:+.4f}  "
              f"McNemar p={p_mc:.2g} ({d_b}/{d_c})  NB p={p_nb:.2g}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*", default=None)
    ap.add_argument("--buffalo", action="store_true",
                    help="the six main datasets (lfw sllfw cplfw xqlfw calfw cfp_fp)")
    ap.add_argument("--out", default="significance_summary.csv")
    args = ap.parse_args()

    paths = [Path(p) for p in (args.emb or [])]
    if args.buffalo:
        paths += [C.EMBEDDINGS_DIR / "lfw_embeddings.npz"] + \
                 [C.EMBEDDINGS_DIR / f"emb_{d}.npz" for d in
                  ("sllfw", "cplfw", "xqlfw", "calfw", "cfp_fp")]
    if not paths:
        ap.error("give --emb <files> or --buffalo")

    rows = [r for p in paths for r in analyze_file(p)]
    C.ensure_dirs()
    out = Path(args.out)
    out = out if out.is_absolute() else C.RESULTS_DIR / out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}  (** = both tests <0.05, * = McNemar only)")


if __name__ == "__main__":
    main()
