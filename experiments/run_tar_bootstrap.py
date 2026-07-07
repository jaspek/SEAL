"""Bootstrap CIs for the headline TAR@FAR=1% deltas (binary-1536, itq256, and
the weak-vs-strong model), which the paper reports as point estimates with no
uncertainty. Pure numpy on cached score vectors; reuses run_significance's
score builders.

  python experiments/run_tar_bootstrap.py --buffalo
Writes outputs/results/tar_bootstrap.csv"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_significance import dataset_scores          # noqa: E402

from facecomp import config as C                      # noqa: E402
from facecomp.data import emb as E                     # noqa: E402
from facecomp.evaluate.verification import paired_tar_bootstrap  # noqa: E402

COMPARISONS = [("binary-1536", "fp32-1536"),
               ("itq256-binary", "fp32-1536"),
               ("magface", "arcface")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*", default=None)
    ap.add_argument("--buffalo", action="store_true")
    ap.add_argument("--far", type=float, default=0.01)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="tar_bootstrap.csv")
    args = ap.parse_args()

    paths = [Path(p) for p in (args.emb or [])]
    if args.buffalo:
        paths += [C.EMBEDDINGS_DIR / "lfw_embeddings.npz"] + \
                 [C.EMBEDDINGS_DIR / f"emb_{d}.npz" for d in
                  ("sllfw", "cplfw", "xqlfw", "calfw", "cfp_fp")]
    if not paths:
        ap.error("give --emb <files> or --buffalo")

    rows = []
    for p in paths:
        b = E.load_emb(p)
        if b.pair_idx is None:
            print(f"SKIP {p.name}: no pair_idx")
            continue
        name = p.stem.replace("emb_", "").replace("_embeddings", "")
        sc = dataset_scores(b)
        labels = b.pair_idx[:, 2]
        print(f"\n=== {name} ===")
        for a, ref in COMPARISONS:
            if a in sc and ref in sc:
                r = paired_tar_bootstrap(sc[a], sc[ref], labels,
                                         far=args.far, n_boot=args.n_boot)
                rows.append({"dataset": name, "A": a, "B": ref, **r})
                print(f"  {a:>14} vs {ref:<12} dTAR {r['delta']:+.3f} "
                      f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  p_boot={r['p_boot']:.3f}")
    C.ensure_dirs()
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / args.out, index=False)
    print("\nwrote", C.RESULTS_DIR / args.out)


if __name__ == "__main__":
    main()
