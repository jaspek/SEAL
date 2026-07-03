"""Build emb_<dataset>.npz. Sources:
  bin       InsightFace .bin pack (pre-aligned pairs -> 1:1)
  sllfw     SLLFW pairs.txt over your aligned LFW crops (-> 1:1)
  tinyface  TinyFace Testing_Set (-> 1:N with gallery/probe/distractor split)

Examples:
  python experiments/build_embeddings.py --dataset cplfw
  python experiments/build_embeddings.py --dataset sllfw    --source sllfw
  python experiments/build_embeddings.py --dataset tinyface --source tinyface --max-distractors 20000
"""
import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")   # silence cv2 findDecoder spam

import argparse
import numpy as np

from facecomp import config as C
from facecomp.data import bin_loader, pairs, tinyface
from facecomp.extract import models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="tag -> writes emb_<tag>.npz")
    ap.add_argument("--source", choices=["bin", "sllfw", "tinyface"], default="bin")
    ap.add_argument("--bin", default=None, help="bin source: path; default <bin_pack>/<dataset>.bin")
    ap.add_argument("--swap-rb", action="store_true", help="bin source: flip R/B if cross-check is off")
    ap.add_argument("--max-distractors", type=int, default=None,
                    help="tinyface source: cap distractor gallery (start small, e.g. 20000)")
    ap.add_argument("--arcface", choices=["buffalo", "ms1mv2", "int8"], default="buffalo",
                    help="'buffalo' = R50/WebFace600K (seminar default); "
                         "'ms1mv2' = R100/MS1MV2 (same-training-set control); "
                         "'int8' = PTQ-quantized buffalo (network compression)")
    ap.add_argument("--batch", type=int, default=64, help="inference batch size")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if emb_<dataset>.npz already exists")
    args = ap.parse_args()

    out = C.emb_path(args.dataset)
    if out.exists() and not args.force:
        print(f"SKIP: {out} already exists (use --force to rebuild)")
        return

    extra = {}
    if args.source == "bin":
        bin_path = args.bin or (C.DATA["bin_pack"] / f"{args.dataset}.bin")
        print(f"loading {bin_path}")
        images, extra["pair_idx"] = bin_loader.load_bin(bin_path, swap_rb=args.swap_rb)
        print(f"  {len(images)} images, {len(extra['pair_idx'])} pairs")
    elif args.source == "sllfw":
        from facecomp.extract import align
        raw = C.DATA.get("lfw_raw")
        align_fn = (lambda rel: align.align_from_raw(rel, raw)) if raw else None
        images, extra["pair_idx"] = pairs.load_sllfw(
            C.DATA["sllfw_pairs"], C.DATA["lfw_aligned"], align_fn)
        print(f"  {len(images)} images, {len(extra['pair_idx'])} pairs")
    else:  # tinyface
        images, extra["labels"], extra["split"] = tinyface.load_tinyface(
            C.DATA["tinyface"], args.max_distractors)

    emb = models.embed_all(images, arcface=args.arcface, batch=args.batch)

    C.ensure_dirs()
    np.savez_compressed(out, **{k: v.astype(np.float32) for k, v in emb.items()}, **extra)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
