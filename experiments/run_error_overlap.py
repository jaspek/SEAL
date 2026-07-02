"""Error-overlap + redundancy (effective rank) per dataset.

Tests the fusion-compression duality: where the effective rank rises (less
redundancy), the shared-error fraction should fall and fusion should start
to help — and compression should start to cost.

Examples:
  python experiments/run_error_overlap.py --emb outputs/embeddings/emb_xqlfw.npz
  python experiments/run_error_overlap.py --all
Writes outputs/results/overlap_summary.csv (one row per dataset)."""
import argparse
from pathlib import Path

import pandas as pd

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.evaluate import error_overlap as EO
from facecomp.evaluate import redundancy as R


def analyze_file(path):
    name = Path(path).stem
    for pre, post in (("emb_", ""), ("_embeddings", "")):
        name = name.replace(pre, post)
    b = E.load_emb(path)
    fused = b.fuse()

    row = {"dataset": name, "n_images": b.N}
    row.update({f"fused_{k}": v for k, v in R.spectrum_metrics(fused).items()})
    for m in E.MODELS:
        if m in b.per_model:
            row[f"erank_{m}"] = R.spectrum_metrics(b.per_model[m])["effective_rank"]

    if b.pair_idx is not None:
        res = EO.analyze(b.per_model, b.pair_idx)
        row.update(res["summary"])
        print(EO.format_report(name, res))
    else:
        print(f"\n=== {name}: no pair_idx (1:N set) — redundancy metrics only ===")
    print(f"  redundancy: fused effective-rank {row['fused_effective_rank']:.0f}/"
          f"{row['fused_dim']}  (k95={row['fused_k95']}, k99={row['fused_k99']})")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*", default=None, help="one or more emb npz files")
    ap.add_argument("--all", action="store_true",
                    help="every *.npz in the embeddings dir (except caches)")
    ap.add_argument("--out", default="overlap_summary.csv")
    args = ap.parse_args()

    if args.all:
        paths = sorted(p for p in C.EMBEDDINGS_DIR.glob("*.npz")
                       if "aligned_cache" not in p.name)
    elif args.emb:
        paths = [Path(p) for p in args.emb]
    else:
        ap.error("give --emb <files> or --all")

    rows = [analyze_file(p) for p in paths]

    C.ensure_dirs()
    out = Path(args.out)
    out = out if out.is_absolute() else C.RESULTS_DIR / out
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    cols = [c for c in ("dataset", "fused_effective_rank", "fused_k95",
                        "shared_frac", "fusion_gain", "acc_best_single") if c in df.columns]
    print("\n=== duality summary (redundancy vs fusion) ===")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
