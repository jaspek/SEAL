"""Open-set 1:N identification (FNIR@FPIR) on TinyFace under template compression.

The paper reports only closed-set rank-k/mAP, which assumes every probe is
enrolled. The watchlist deployment setting is open-set. Here we remove half the
mated identities from the gallery -> their probes become known non-mates, keep
all real distractors, and report FNIR at FPIR=1%/10% per compressor.

  python experiments/run_openset.py --emb outputs/embeddings/emb_tinyface.npz
Writes outputs/results/openset_tinyface.csv"""
import argparse

import numpy as np
import pandas as pd

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.compress import template as T
from facecomp.evaluate.identification import openset_metrics

SPECS = [(None, None, "fp32"), (None, None, "int8"), (None, None, "binary"),
         ("itq", 256, "binary"), ("pca", 128, "binary")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="openset_tinyface.csv")
    args = ap.parse_args()

    b = E.load_emb(args.emb)
    if b.split is None or b.labels is None:
        raise SystemExit("need a TinyFace-style npz with split + labels")
    fused = b.fuse()
    g = np.where(b.split == 0)[0]
    p = np.where(b.split == 1)[0]
    d = np.where(b.split == 2)[0]

    rng = np.random.default_rng(args.seed)
    ids = np.unique(b.labels[g])
    keep = set(rng.choice(ids, size=len(ids) // 2, replace=False).tolist())
    g_keep = g[np.array([b.labels[i] in keep for i in g])]     # enrolled half
    print(f"gallery ids {len(ids)} -> enrolled {len(keep)}; "
          f"probes {len(p)}, distractors {len(d)}")

    rows = []
    for kind, k, prec in SPECS:
        red = None if not kind else (T.PCAReducer(k) if kind == "pca"
                                     else T.ITQReducer(k)).fit(fused)
        comp, bits = T.compress(fused, prec, red)
        gal = np.vstack([comp[g_keep], comp[d]])
        gal_lab = np.concatenate([b.labels[g_keep], np.full(len(d), -1, np.int64)])
        m = openset_metrics(gal, gal_lab, comp[p], b.labels[p])
        tag = f"{kind or 'none'}{k or ''}-{prec}"
        rows.append({"method": tag, "bits": bits, **m})
        print(f"  {tag:16s} FNIR@1%={m['fnir@fpir0.01']}  FNIR@10%={m['fnir@fpir0.1']}  "
              f"(mate {m['n_mate']} / nonmate {m['n_nonmate']})")
    C.ensure_dirs()
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / args.out, index=False)
    print("wrote", C.RESULTS_DIR / args.out)


if __name__ == "__main__":
    main()
