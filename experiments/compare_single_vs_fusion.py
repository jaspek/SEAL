"""The 'compress the single strong model, not the fusion' finding already sitting
in your CSVs: at equal bits, compressed ArcFace-alone vs the compressed fused
template. Reads rd_<d>.csv (fused) and rd_<d>_arc.csv (arcface-only) and reports
the per-configuration gap.

  python experiments/compare_single_vs_fusion.py --metric tar_far0.01
Writes outputs/results/single_vs_fusion.csv"""
import argparse

import pandas as pd

from facecomp import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["lfw", "sllfw", "cfp_fp", "calfw", "cplfw", "xqlfw"])
    ap.add_argument("--metric", default="acc",
                    choices=["acc", "tar_far0.01", "tar_far0.001", "rank1"])
    ap.add_argument("--out", default="single_vs_fusion.csv")
    args = ap.parse_args()
    col = args.metric
    rows = []
    for d in args.datasets:
        f = C.RESULTS_DIR / f"rd_{d}.csv"
        a = C.RESULTS_DIR / f"rd_{d}_arc.csv"
        if not (f.exists() and a.exists()):
            print(f"skip {d} (need {f.name} and {a.name})")
            continue
        fdf, adf = pd.read_csv(f), pd.read_csv(a)
        m = pd.merge(fdf, adf, on=["reducer", "dim", "precision"],
                     suffixes=("_fused", "_arc"))
        cf, ca = f"{col}_fused", f"{col}_arc"
        if cf not in m.columns or ca not in m.columns:
            print(f"skip {d}: metric '{col}' not in both CSVs")
            continue
        for _, r in m.iterrows():
            rows.append({"dataset": d, "reducer": r["reducer"], "dim": r["dim"],
                         "precision": r["precision"],
                         "bits_fused": r.get("bits_fused"), "bits_arc": r.get("bits_arc"),
                         cf: r[cf], ca: r[ca],
                         "arc_minus_fused": round(float(r[ca] - r[cf]), 4)})
    df = pd.DataFrame(rows)
    C.ensure_dirs()
    df.to_csv(C.RESULTS_DIR / args.out, index=False)
    if len(df):
        print(df.sort_values("arc_minus_fused", ascending=False)
                .head(15).to_string(index=False))
    print("\nwrote", C.RESULTS_DIR / args.out)


if __name__ == "__main__":
    main()
