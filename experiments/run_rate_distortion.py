"""Rate-distortion study over a real emb_<dataset>.npz (or synthetic if --emb omitted)."""
import argparse
from pathlib import Path

import numpy as np

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.evaluate import rate_distortion as RD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=None, help="emb_<dataset>.npz; omit for synthetic")
    ap.add_argument("--models", nargs="+", default=list(E.MODELS))
    ap.add_argument("--num-distractors", type=int, default=0)
    ap.add_argument("--out", default="rd.csv")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.emb:
        b = E.load_emb(args.emb, models=tuple(args.models))
    else:
        b = E.make_synthetic_bundle(seed=args.seed)
    models = tuple(m for m in args.models if m in b.per_model)
    fused = b.fuse(models=models, method="concat")
    print(f"N={b.N}, fused_dim={fused.shape[1]}, "
          f"labels={'y' if b.labels is not None else 'n'}, "
          f"pairs={0 if b.pair_idx is None else len(b.pair_idx)}")

    # TinyFace ships a predefined gallery/probe/distractor split; honor it.
    gallery_idx = probe_idx = None
    num_distractors = args.num_distractors
    if b.split is not None:
        gallery_idx = np.where(b.split != 1)[0]   # gallery_match (0) + distractor (2)
        probe_idx = np.where(b.split == 1)[0]
        num_distractors = 0                        # real distractors already in the gallery
        print(f"  predefined split: {len(gallery_idx)} gallery, {len(probe_idx)} probe")

    df = RD.rate_distortion(fused, labels=b.labels, pair_idx=b.pair_idx,
                            gallery_idx=gallery_idx, probe_idx=probe_idx,
                            num_distractors=num_distractors, seed=args.seed).sort_values("bits")
    print(df.to_string(index=False))

    C.ensure_dirs()
    out = Path(args.out)
    out = out if out.is_absolute() else C.RESULTS_DIR / out
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    if args.plot:
        from facecomp.viz.plots import plot_rate_distortion
        ycol = "rank1" if "rank1" in df.columns else "acc"
        fig = C.FIGURES_DIR / (out.stem + ".png")
        plot_rate_distortion(df, y=ycol, out=fig, title=out.stem)
        print(f"wrote {fig}")


if __name__ == "__main__":
    main()
