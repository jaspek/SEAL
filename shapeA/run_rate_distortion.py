#!/usr/bin/env python3
"""
Shape A main driver: template-compression rate-distortion as 1:N retrieval.

Runs entirely on CPU. Produces a CSV (one row per compression method) with
bits/template vs Rank-1 / Rank-5 / mAP / EER / TAR@FAR, and (if matplotlib is
present) the rate-distortion plots that are the headline figure of the paper.

Example
-------
  # single model
  python run_rate_distortion.py --emb emb.npz --models arcface --out rd_arc.csv
  # fused (concatenate three models, renormalize)
  python run_rate_distortion.py --emb emb.npz --models arcface magface adaface \
        --out rd_fused.csv --num-distractors 100000

For the headline "compression stops being free at scale" curve, run the same
--emb at several --num-distractors values (0, 1e4, 1e5, 1e6) and overlay Rank-1.
"""
import argparse
import csv
import numpy as np

from facecomp import compress, evaluate, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="path to .npz of embeddings")
    ap.add_argument("--models", nargs="+", default=["arcface"],
                    help="model key(s) in the npz; multiple => fused+renorm")
    ap.add_argument("--label-key", default="labels")
    ap.add_argument("--out", default="rate_distortion.csv")
    ap.add_argument("--k", type=int, default=256,
                    help="reduced dim for PCA/ITQ methods")
    ap.add_argument("--num-distractors", type=int, default=0,
                    help="extra synthetic distractors appended to the gallery")
    ap.add_argument("--n-pairs", type=int, default=20000,
                    help="sampled pairs for 1:1 verification metrics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    X, labels = data.load_npz(args.emb, args.models, label_key=args.label_key)
    print(f"loaded {X.shape[0]} embeddings, dim={X.shape[1]}, "
          f"{len(np.unique(labels))} identities")

    split = data.build_gallery_probe(labels, seed=args.seed)
    gidx, pidx = split["gallery_idx"], split["probe_idx"]
    gX, gids = X[gidx], labels[gidx]
    pX, pids = X[pidx], labels[pidx]

    if args.num_distractors:
        dX, _ = data.make_synthetic(n_ids=args.num_distractors, imgs_per_id=1,
                                    dim=X.shape[1], seed=args.seed + 7)
        gX, gids = data.append_distractors(gX, gids, dX)
    print(f"gallery={gX.shape[0]} (incl. distractors), probes={pX.shape[0]}")

    # Fit compressors on the gallery (codebooks/rotations); evaluate on probes.
    zoo = compress.default_zoo(k=args.k)
    rows = []
    fp32_bits = 32 * X.shape[1]
    for comp in zoo:
        try:
            comp.fit(gX)
        except Exception as e:
            print(f"[skip] {comp.name}: {e}")
            continue
        res = evaluate.evaluate_compressor(
            comp, gX, gids, pX, pids,
            all_X=X, all_ids=labels, n_pairs=args.n_pairs)
        res["method"] = comp.name
        res["compression_vs_fp32"] = round(fp32_bits / comp.bits, 2)
        rows.append(res)
        print(f"{comp.name:14s} bits={res['bits_per_vec']:6d} "
              f"({res['bytes_per_vec']:7.1f} B)  "
              f"R1={res['rank1']:.4f} R5={res['rank5']:.4f} "
              f"mAP={res['map']:.4f} EER={res.get('eer', float('nan')):.4f} "
              f"TAR@1e-2={res.get('tar@far=0.01', float('nan')):.4f}")

    cols = ["method", "bits_per_vec", "bytes_per_vec", "compression_vs_fp32",
            "rank1", "rank5", "rank10", "map", "eer",
            "tar@far=0.01", "tar@far=0.001"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {args.out}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            b = np.array([r["bytes_per_vec"] for r in rows])
            for metric, ylab in [("rank1", "Rank-1"), ("tar@far=0.001", "TAR@FAR=1e-3")]:
                y = np.array([r.get(metric, np.nan) for r in rows])
                plt.figure()
                plt.semilogx(b, y, "o-")
                for r in rows:
                    plt.annotate(r["method"], (r["bytes_per_vec"], r.get(metric, np.nan)),
                                 fontsize=7)
                plt.xlabel("bytes / template (log)")
                plt.ylabel(ylab)
                plt.title(f"Rate-distortion ({'+'.join(args.models)})")
                plt.grid(True, which="both", alpha=0.3)
                fn = args.out.replace(".csv", f"_{metric.replace('@','_').replace('=','')}.png")
                plt.savefig(fn, dpi=140, bbox_inches="tight")
                print(f"wrote {fn}")
        except Exception as e:
            print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
