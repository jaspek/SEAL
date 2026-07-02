from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_rate_distortion(df, y="rank1", out=None, title="Rate-distortion"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (red, prec), g in df.groupby(["reducer", "precision"]):
        g = g.sort_values("bits")
        ax.plot(g["bits"], g[y], marker="o", label=f"{red}/{prec}")
    ax.set_xscale("log")
    ax.set_xlabel("bits / template")
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig
