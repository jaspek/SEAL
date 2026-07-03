"""Overlay figure: rank-1 vs gallery size. Thin wrapper -- the styling lives in
facecomp.viz.plots; `python experiments/replot.py` regenerates this and all
rate-distortion figures together."""
import re

import pandas as pd

from facecomp import config as C
from facecomp.viz.plots import plot_gallery_sweep


def main():
    frames = []
    for p in sorted(C.RESULTS_DIR.glob("rd_lfw_N*.csv")):
        m = re.fullmatch(r"rd_lfw_N(\d+)", p.stem)
        if not m:
            continue
        df = pd.read_csv(p)
        df["N"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise SystemExit("no rd_lfw_N*.csv files found")
    out = C.FIGURES_DIR / "gallery_sweep.png"
    plot_gallery_sweep(pd.concat(frames), y="rank1", out=out,
                       title="LFW probes vs. growing gallery")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
