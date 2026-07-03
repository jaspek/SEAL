"""Regenerate every figure from the existing CSVs -- no re-evaluation needed.

  python experiments/replot.py
Rebuilds outputs/figures/rd_*.png (from rd_*.csv) and gallery_sweep.png
(from rd_lfw_N*.csv) in the publication style (y axis 0-100%, labeled axes)."""
import re

import pandas as pd

from facecomp import config as C
from facecomp.viz.plots import plot_rate_distortion, plot_gallery_sweep

NICE = {"lfw": "LFW", "sllfw": "SLLFW", "cplfw": "CPLFW", "xqlfw": "XQLFW",
        "calfw": "CALFW", "cfp_fp": "CFP-FP",
        "tinyface": "TinyFace (20k distractors)",
        "tinyface_full": "TinyFace (153k distractors)"}


def nice_name(stem):
    d = stem.replace("rd_", "")
    ctrl = d.endswith("_ctrl")
    if ctrl:
        d = d[:-5]
    name = NICE.get(d, d.upper())
    return f"{name} (MS1MV2 control)" if ctrl else name


def main():
    C.ensure_dirs()
    sweep = {}
    for p in sorted(C.RESULTS_DIR.glob("rd_*.csv")):
        m = re.fullmatch(r"rd_lfw_N(\d+)", p.stem)
        if m:
            df = pd.read_csv(p)
            df["N"] = int(m.group(1))
            sweep[int(m.group(1))] = df
            continue
        df = pd.read_csv(p)
        y = "acc" if "acc" in df.columns else "rank1"
        out = C.FIGURES_DIR / f"{p.stem}.png"
        plot_rate_distortion(df, y=y, out=out, title=nice_name(p.stem))
        print(f"wrote {out.name}")
    if sweep:
        df = pd.concat(sweep.values())
        out = C.FIGURES_DIR / "gallery_sweep.png"
        plot_gallery_sweep(df, y="rank1", out=out,
                           title="LFW probes vs. growing gallery")
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
