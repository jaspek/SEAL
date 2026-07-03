from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Y_LABELS = {
    "acc": "Verification accuracy (%)",
    "rank1": "Rank-1 identification rate (%)",
    "rank5": "Rank-5 identification rate (%)",
    "mAP": "Mean average precision (%)",
    "tar_far0.01": "TAR @ FAR = 1% (%)",
}

LINE_LABELS = {
    ("none", "fp32"): "float32",
    ("none", "fp16"): "float16",
    ("none", "int8"): "int8",
    ("none", "binary"): "binary (sign)",
    ("itq", "binary"): "ITQ + sign",
    ("pca", "binary"): "PCA + sign",
}
LINE_ORDER = [("none", "fp32"), ("none", "fp16"), ("none", "int8"),
              ("none", "binary"), ("itq", "binary"), ("pca", "binary")]

# slanted tick drawn at the ends of a broken spine
_BREAK_BASE = dict(markersize=9, linestyle="none", color="k", mec="k",
                   mew=1, clip_on=False)
BREAK_Y = dict(marker=[(-1, -0.5), (1, 0.5)], **_BREAK_BASE)   # cut in a vertical spine
BREAK_X = dict(marker=[(-0.5, -1), (0.5, 1)], **_BREAK_BASE)   # cut in a horizontal spine


def _grid(ax):
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", axis="x", alpha=0.15)
    ax.tick_params(labelsize=10)


def _broken_bottom(vals):
    """Lower y limit for the zoomed segment of a broken y axis, or None when
    the data reaches low enough that a plain 0-100 axis shows it honestly."""
    lo = float(vals.min())
    if lo <= 45:
        return None
    b = int(5 * ((lo - 3) // 5))
    if 100 - b > 30:                       # wide span: keep ticks on tens
        b = int(10 * ((lo - 3) // 10))
    return b


def _ystep(bottom):
    return 5 if 100 - bottom <= 30 else 10


def _apply_y_break(ax, cut, bottom):
    """Turn (data axis, thin strip below) into one y axis broken above 0.
    The strip keeps the same data-per-inch scale as the zoomed segment."""
    span = 100 - bottom
    ax.set_ylim(bottom, 100)
    ax.set_yticks(range(bottom, 101, _ystep(bottom)))
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)
    cut.set_ylim(0, span / 12)
    cut.set_yticks([0])
    cut.spines[["top", "right"]].set_visible(False)
    ax.plot([0], [0], transform=ax.transAxes, **BREAK_Y)
    cut.plot([0], [1], transform=cut.transAxes, **BREAK_Y)


def plot_rate_distortion(df, y="acc", out=None, title=""):
    """Rate-distortion curve: metric (%) vs bits/template on a log x axis.
    The y axis stays anchored at 0; when every configuration sits high, the
    axis is broken (thin 0-strip below a zoomed segment) so the compression
    cost stays readable. Three families: the full-dimension precision sweep
    (binary->int8->fp16->fp32, one connected annotated curve) and ITQ/PCA."""
    bottom = _broken_bottom(100 * df[y])
    if bottom is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        cut = None
    else:
        fig, (ax, cut) = plt.subplots(
            2, 1, sharex=True, figsize=(7, 4.8),
            gridspec_kw={"height_ratios": [12, 1], "hspace": 0.08})

    full = df[df["reducer"] == "none"].sort_values("bits")
    ax.plot(full["bits"], 100 * full[y], marker="o", markersize=6, linewidth=1.8,
            color="#1f77b4", label="full 1536-d (precision sweep)")
    for _, r in full.iterrows():
        ax.annotate(r["precision"], (r["bits"], 100 * r[y]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#1f77b4")

    for red, color, label, dy in (("itq", "#2ca02c", "ITQ + sign (k dims)", 8),
                                  ("pca", "#d62728", "PCA + sign (k dims)", -15)):
        g = df[df["reducer"] == red].sort_values("bits")
        if g.empty:
            continue
        ax.plot(g["bits"], 100 * g[y], marker="s", markersize=6, linewidth=1.8,
                color=color, label=label)
        for _, r in g.iterrows():
            ax.annotate(f"k={int(r['dim'])}", (r["bits"], 100 * r[y]),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=8, color=color)

    ax.set_xscale("log")
    ax.set_ylabel(Y_LABELS.get(y, y), fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    _grid(ax)
    xlabel = "Template size (bits, log scale)"
    if cut is None:
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 10))
        ax.set_xlabel(xlabel, fontsize=11)
    else:
        _apply_y_break(ax, cut, bottom)
        _grid(cut)
        cut.set_xlabel(xlabel, fontsize=11)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_gallery_sweep(df, y="rank1", out=None, title=""):
    """Metric (%) vs gallery size. The x axis starts at N=0: the zero point
    gets its own narrow linear panel, separated by an axis break from the log
    panel (log scales cannot reach 0). The y axis is anchored at 0 and broken
    to a zoomed segment when all curves sit high."""
    bottom = _broken_bottom(100 * df[y])
    col_kw = {"width_ratios": [1, 14], "wspace": 0.06}
    if bottom is None:
        fig, (tl, tr) = plt.subplots(1, 2, sharey=True, figsize=(7.5, 4.5),
                                     gridspec_kw=col_kw)
        bl = br = None
    else:
        fig, ((tl, tr), (bl, br)) = plt.subplots(
            2, 2, sharex="col", sharey="row", figsize=(7.5, 4.8),
            gridspec_kw={"height_ratios": [12, 1], "hspace": 0.08, **col_kw})

    groups = {k: g for k, g in df.groupby(["reducer", "precision", "dim"])}

    def key2(k):
        return (k[0], k[1])
    order = sorted(groups, key=lambda k: (LINE_ORDER.index(key2(k))
                                          if key2(k) in LINE_ORDER else 99, -k[2]))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, k in enumerate(order):
        g = groups[k].sort_values("N")
        base = LINE_LABELS.get(key2(k), f"{k[0]}/{k[1]}")
        label = base if k[0] == "none" else f"{base.split(' ')[0]}-{k[2]} + sign"
        c = colors[i % len(colors)]
        tl.plot(g["N"], 100 * g[y], marker="o", markersize=5, linewidth=1.8, color=c)
        gp = g[g["N"] > 0]
        tr.plot(gp["N"], 100 * gp[y], marker="o", markersize=5, linewidth=1.8,
                color=c, label=label)

    tr.set_xscale("log")
    tr.set_xlim(4e3, 2.2e6)
    tl.set_xlim(-0.4, 0.4)
    for a in ((tl, bl) if bl is not None else (tl,)):
        a.set_xticks([0])
        a.set_xticklabels(["0"])
    for a in ((tr, br) if br is not None else (tr,)):
        a.set_xticks([1e4, 1e5, 1e6])
        a.set_xticklabels([r"$10^4$", r"$10^5$", r"$10^6$"])

    tl.set_ylabel(Y_LABELS.get(y, y), fontsize=11)
    tl.spines[["top", "right"]].set_visible(False)
    tr.spines[["top", "right", "left"]].set_visible(False)
    tr.tick_params(axis="y", which="both", length=0)
    _grid(tl)
    _grid(tr)

    xlabel = "Distractors in gallery (log scale)"
    if bl is None:
        tl.set_ylim(0, 100)
        tl.set_yticks(range(0, 101, 10))
        tr.set_xlabel(xlabel, fontsize=11)
        tl.plot([1], [0], transform=tl.transAxes, **BREAK_X)
        tr.plot([0], [0], transform=tr.transAxes, **BREAK_X)
    else:
        span = 100 - bottom
        tl.set_ylim(bottom, 100)
        tl.set_yticks(range(bottom, 101, _ystep(bottom)))
        for a in (tl, tr):
            a.spines["bottom"].set_visible(False)
            a.tick_params(axis="x", which="both", length=0)
        for a, hide in ((bl, ["top", "right"]), (br, ["top", "right", "left"])):
            a.set_ylim(0, span / 12)
            a.spines[hide].set_visible(False)
            _grid(a)
        bl.set_yticks([0])
        br.tick_params(axis="y", which="both", length=0)
        br.set_xlabel(xlabel, fontsize=11)
        # y cut on the left spine, x cut on the bottom spine
        tl.plot([0], [0], transform=tl.transAxes, **BREAK_Y)
        bl.plot([0], [1], transform=bl.transAxes, **BREAK_Y)
        bl.plot([1], [0], transform=bl.transAxes, **BREAK_X)
        br.plot([0], [0], transform=br.transAxes, **BREAK_X)

    if title:
        fig.suptitle(title, fontsize=12)
    handles, labels = tr.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, fontsize=8.5, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig
