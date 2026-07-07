"""LFW gallery sweep using REAL TinyFace face embeddings as distractors, each
passed through the same compressor as the gallery. Reuses emb_tinyface.npz
(split==2) -- no new extraction. Fixes two issues in the paper's synthetic sweep:
distractors are real faces (not Gaussian noise) AND they are compressed the same
way as the gallery templates.

  python experiments/run_real_distractor_sweep.py"""
import argparse

import numpy as np
import pandas as pd

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.compress import template as T
from facecomp.evaluate import identification as idn

SPECS = [(None, None, "fp32"), (None, None, "binary"),
         ("pca", 128, "binary"), ("itq", 256, "binary")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", default=str(C.EMBEDDINGS_DIR / "emb_lfw_bin.npz"))
    ap.add_argument("--tinyface", default=str(C.EMBEDDINGS_DIR / "emb_tinyface.npz"))
    ap.add_argument("--counts", nargs="*", type=int, default=[0, 10000, 100000, 153000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="real_distractor_sweep.csv")
    args = ap.parse_args()

    lfw = E.load_emb(args.lfw)
    if lfw.labels is None:
        raise SystemExit("LFW npz needs identity labels (build with paths/labels)")
    g_idx, p_idx = E.gallery_probe_split(lfw.labels, seed=args.seed)
    fused_lfw = lfw.fuse()

    tf = E.load_emb(args.tinyface)
    if tf.split is None:
        raise SystemExit("TinyFace npz needs a split array (split==2 are distractors)")
    distract_raw = tf.fuse()[np.where(tf.split == 2)[0]]      # real face distractors
    print(f"LFW: {len(g_idx)} gallery / {len(p_idx)} probe   "
          f"real distractors available: {len(distract_raw)}")

    rows = []
    for kind, k, prec in SPECS:
        red = None if not kind else (T.PCAReducer(k) if kind == "pca"
                                     else T.ITQReducer(k)).fit(fused_lfw)
        gc, bits = T.compress(fused_lfw, prec, red)
        dc, _ = T.compress(distract_raw, prec, red)          # same compressor
        tag = f"{kind or 'none'}{k or ''}-{prec}"
        for n in args.counts:
            m = min(n, len(dc))
            gemb = np.vstack([gc[g_idx], dc[:m]]) if m else gc[g_idx]
            glab = np.concatenate([lfw.labels[g_idx], np.full(m, -1, np.int64)])
            r = idn.cmc_and_map(gemb, glab, gc[p_idx], lfw.labels[p_idx])
            rows.append({"method": tag, "bits": bits, "N_distract": n,
                         "rank1": round(r["rank-1"], 6), "rank5": round(r["rank-5"], 6)})
            print(f"  {tag:16s} N={n:>7}: rank1 {r['rank-1']:.4f}")
    C.ensure_dirs()
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / args.out, index=False)
    print("wrote", C.RESULTS_DIR / args.out)


if __name__ == "__main__":
    main()
