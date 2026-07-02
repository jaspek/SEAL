"""Overlay figure: rank-1 vs gallery size, one line per compression config.
Reads the rd_lfw_N*.csv sweep tables and writes the headline plot.

  python experiments/plot_gallery_sweep.py                # defaults
  python experiments/plot_gallery_sweep.py --metric mAP
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from facecomp import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="rd_lfw_N*.csv")
    ap.add_argument("--metric", default="rank1")
    ap.add_argument("--out", default="gallery_sweep.png")
    args = ap.parse_args()

    frames = []
    for p in sorted(C.RESULTS_DIR.glob(args.pattern)):
        m = re.search(r"N(\d+)", p.stem)
        if not m:
            continue
        df = pd.read_csv(p)
        df["N"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise SystemExit(f"no files matching {args.pattern} in {C.RESULTS_DIR}")
    df = pd.concat(frames)
    df["config"] = df.apply(
        lambda r: f"{r['reducer']}-{r['dim']}/{r['precision']} ({int(r['bits'])} bits)"
        if r["reducer"] != "none" else f"{r['precision']} ({int(r['bits'])} bits)", axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = df[df["N"] == df["N"].max()].sort_values(args.metric, ascending=False)
    for cfg in order["config"]:
        g = df[df["config"] == cfg].sort_values("N")
        n = g["N"].clip(lower=1)          # log axis; N=0 plotted at 1
        ax.plot(n, g[args.metric], marker="o", label=cfg)
    ax.set_xscale("log")
    ax.set_xlabel("distractors in gallery (log)")
    ax.set_ylabel(args.metric)
    ax.set_title("Compression cost grows with the gallery (LFW probes)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    out = C.FIGURES_DIR / args.out
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
