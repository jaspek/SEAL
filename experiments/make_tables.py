"""Generate every LaTeX table for the paper from outputs/results/*.csv.

  python experiments/make_tables.py
Writes outputs/tables/tab_*.tex (+ tables_all.tex master include).
Style matches the draft: booktabs, [H] float, caption + label per table.
Significance marks: ** = McNemar and Nadeau-Bengio p<0.05, * = McNemar only."""
from pathlib import Path

import numpy as np
import pandas as pd

from facecomp import config as C

OUT = C.REPO_ROOT / "outputs" / "tables"
DATASETS = ["lfw", "sllfw", "cfp_fp", "calfw", "cplfw", "xqlfw"]     # easy -> hard
NICE = {"lfw": "LFW", "sllfw": "SLLFW", "cfp_fp": "CFP-FP", "calfw": "CALFW",
        "cplfw": "CPLFW", "xqlfw": "XQLFW"}

rd_cache = {}


def rd(name):
    if name not in rd_cache:
        rd_cache[name] = pd.read_csv(C.RESULTS_DIR / f"rd_{name}.csv")
    return rd_cache[name]


def cell(df, reducer, precision, col, dim=None):
    m = (df["reducer"] == reducer) & (df["precision"] == precision)
    if dim is not None:
        m &= df["dim"] == dim
    return float(df.loc[m, col].iloc[0])


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def dl(x, base, d=2):
    return f"{100 * (x - base):+.{d}f}"


SIG = pd.read_csv(C.RESULTS_DIR / "significance_summary.csv")


def _mark(p_mc, p_nb):
    if p_mc < 0.05 and p_nb < 0.05:
        return r"$^{**}$"
    if p_mc < 0.05:
        return r"$^{*}$"
    return ""


def _csv(name):
    p = C.RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def stars(dataset, a, b="fp32-1536"):
    m = SIG[(SIG["dataset"] == dataset) & (SIG["A"] == a) & (SIG["B"] == b)]
    if m.empty:
        return ""
    return _mark(float(m["p_mcnemar"].iloc[0]), float(m["p_nb"].iloc[0]))


def write(name, caption, label, header, rows, align):
    lines = [r"\begin{table}[H]", r"\centering", rf"\begin{{tabular}}{{{align}}}",
             r"\toprule", header + r" \\", r"\midrule"]
    lines += [r + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote outputs/tables/{name}.tex")


def t_baselines():
    ov = pd.read_csv(C.RESULTS_DIR / "overlap_summary.csv").set_index("dataset")
    rows = []
    for d in DATASETS:
        r, o = rd(d), ov.loc[d]
        fp = cell(r, "none", "fp32", "acc")
        rows.append(f"{NICE[d]} & {pct(o['acc_arcface'])} & {pct(o['acc_magface'])} & "
                    f"{pct(o['acc_adaface'])} & {pct(fp)} & "
                    f"{pct(cell(r,'none','fp32','eer'))} & "
                    f"{pct(cell(r,'none','fp32','tar_far0.01'))}")
    write("tab_baselines",
          "Verification accuracy (\\%), EER and TAR@FAR=1\\% of the three models and "
          "their fused concatenation across the six benchmarks (10-fold protocol; "
          "fused = 1536-d concat, float32). Datasets ordered by difficulty.",
          "tab:baselines",
          r"Dataset & ArcFace & MagFace & AdaFace & Fused & EER & TAR@1\%FAR",
          rows, "lcccccc")


def t_compression_acc():
    rows = []
    for d in DATASETS:
        r = rd(d)
        fp = cell(r, "none", "fp32", "acc")
        def f(red, prec, a, dim=None):
            v = cell(r, red, prec, "acc", dim)
            return f"{pct(v)} ({dl(v, fp)}){stars(d, a)}"
        rows.append(f"{NICE[d]} & {pct(fp)} & {f('none','int8','int8-1536')} & "
                    f"{f('none','binary','binary-1536')} & "
                    f"{f('itq','binary','itq256-binary',256)} & "
                    f"{f('itq','binary','itq128-binary',128)}")
    write("tab_compression_acc",
          "Verification accuracy (\\%) under template compression, with the change "
          "vs.\\ float32 in parentheses. $^{**}$: significant under exact McNemar "
          "and Nadeau--Bengio ($p<0.05$); $^{*}$: McNemar only. Compression is free "
          "on LFW and becomes progressively more expensive as the data hardens.",
          "tab:compression_acc",
          r"Dataset & fp32 & int8 & binary (1536\,b) & ITQ-256 (256\,b) & ITQ-128 (128\,b)",
          rows, "lccccc")


def t_compression_tar():
    rows = []
    for d in DATASETS:
        r = rd(d)
        fp = cell(r, "none", "fp32", "tar_far0.01")
        def f(red, prec, dim=None):
            v = cell(r, red, prec, "tar_far0.01", dim)
            return f"{pct(v)} ({dl(v, fp)})"
        rows.append(f"{NICE[d]} & {pct(fp)} & {f('none','int8')} & "
                    f"{f('none','binary')} & {f('itq','binary',256)} & "
                    f"{f('itq','binary',128)}")
    write("tab_compression_tar",
          "TAR@FAR=1\\% (\\%) under template compression (change vs.\\ float32 in "
          "parentheses). At a deployed operating point the compression cost is "
          "roughly twice the accuracy cost; even full-dimension binary is no "
          "longer free on the hard benchmarks.",
          "tab:compression_tar",
          r"Dataset & fp32 & int8 & binary (1536\,b) & ITQ-256 & ITQ-128",
          rows, "lccccc")


def t_overlap():
    ov = pd.read_csv(C.RESULTS_DIR / "overlap_summary.csv").set_index("dataset")
    rows = []
    for d in DATASETS:
        o = ov.loc[d]
        rows.append(f"{NICE[d]} & {int(o['err_arcface'])} / {int(o['err_magface'])} / "
                    f"{int(o['err_adaface'])} & {int(o['err_shared'])} & "
                    f"{100*o['shared_frac']:.1f} & {int(o['shared_false_accepts'])} & "
                    f"{pct(o['acc_best_single'])} & {pct(o['acc_oracle'])}")
    write("tab_overlap",
          "Error overlap across the three models per benchmark (global-threshold "
          "diagnostic). On LFW 99\\% of errors are shared and none are false "
          "accepts; on XQLFW the shared fraction collapses to 36\\% and shared "
          "false accepts appear -- error diversity exists, see "
          "Table~\\ref{tab:fusion_learned} for whether fusion can exploit it.",
          "tab:overlap",
          r"Dataset & Errors (arc/mag/ada) & Shared & Shared \% & Shared FA & "
          r"Best single & Oracle",
          rows, "lcccccc")


def t_fusion_learned():
    lf = pd.read_csv(C.RESULTS_DIR / "learned_fusion_summary.csv").set_index("dataset")
    rows = []
    for d in DATASETS:
        f = lf.loc[d]
        p = f["p_vs_best_single"]
        rows.append(f"{NICE[d]} & {pct(f['acc_best_single'])} & "
                    f"{pct(f['acc_score_mean'])} & {pct(f['acc_learned'])} & "
                    f"{100*f['gain_vs_best_single']:+.2f} & {p:.2g} & "
                    f"{f['w_arcface']:.2f}/{f['w_magface']:.2f}/{f['w_adaface']:.2f}")
    write("tab_fusion_learned",
          "Score-level fusion vs.\\ the fold-honest best single model. Learned "
          "logistic-regression fusion repairs naive averaging but never "
          "significantly exceeds the best single model; its weights (right column) "
          "collapse onto ArcFace as the data hardens -- the error diversity of "
          "Table~\\ref{tab:overlap} is not accessible from scores alone.",
          "tab:fusion_learned",
          r"Dataset & Best single & Score-mean & Learned (LR) & Gain & $p$ & "
          r"Weights arc/mag/ada",
          rows, "lcccccc")


def t_control():
    ov = pd.read_csv(C.RESULTS_DIR / "overlap_summary.csv").set_index("dataset")
    rows = []
    for d in ["sllfw", "cfp_fp", "calfw", "cplfw", "xqlfw"]:
        b, c = rd(d), rd(f"{d}_ctrl")
        fb, fc = cell(b, "none", "fp32", "acc"), cell(c, "none", "fp32", "acc")
        ab = ov.loc[d, "acc_arcface"]
        ac = ov.loc[f"{d}_ctrl", "acc_arcface"]
        rows.append(f"{NICE[d]} & {pct(ab)} & {pct(ac)} & {dl(ac, ab)} & "
                    f"{pct(fb)} & {pct(fc)} & {dl(fc, fb)}")
    write("tab_control",
          "Same-training-set control: replacing ArcFace (iResNet50, WebFace600K) "
          "with the iResNet100/MS1MV2 checkpoint, so all three models share "
          "training data. Both ArcFace variants are equivalent on LFW "
          "($\\sim$99.8\\% on official crops), yet the WebFace600K model is far "
          "more robust on cross-quality data -- the training-set difference, not "
          "the loss, carries the robustness.",
          "tab:control",
          r"Dataset & Arc (W600K) & Arc (MS1MV2) & $\Delta$ & Fused (W600K) & "
          r"Fused (MS1MV2) & $\Delta$",
          rows, "lcccccc")


def t_tinyface():
    r = pd.read_csv(C.RESULTS_DIR / "rd_tinyface_full.csv")
    label = {("pca", 128): "PCA-128 + sign", ("itq", 128): "ITQ-128 + sign",
             ("pca", 256): "PCA-256 + sign", ("itq", 256): "ITQ-256 + sign"}
    rows = []
    for _, x in r.iterrows():
        name = (label.get((x["reducer"], x["dim"]))
                if x["reducer"] != "none" else x["precision"])
        rows.append(f"{name} & {int(x['bits'])} & {pct(x['rank1'])} & "
                    f"{pct(x['rank5'])} & {pct(x['mAP'])}")
    write("tab_tinyface",
          "1:N identification on TinyFace (3{,}728 native low-resolution probes "
          "vs.\\ 4{,}443 mated gallery images + 153{,}428 real distractors). "
          "Binary quantization costs 2.6 rank-1 points here and the 256-bit "
          "template collapses by 10 -- compression is no longer free at "
          "identification scale on low-quality data.",
          "tab:tinyface",
          r"Template & Bits & Rank-1 & Rank-5 & mAP",
          rows, "lcccc")


def t_gallery():
    ns = [0, 10000, 100000, 1000000]
    configs = [("pca", 128, "binary", "PCA-128"), ("itq", 256, "binary", "ITQ-256"),
               ("none", 1536, "binary", "binary-1536"), ("none", 1536, "fp32", "fp32")]
    data = {n: pd.read_csv(C.RESULTS_DIR / f"rd_lfw_N{n}.csv") for n in ns}
    rows = []
    for red, dim, prec, name in configs:
        vals = [cell(data[n], red, prec, "rank1", dim) for n in ns]
        rows.append(name + " & " + " & ".join(pct(v) for v in vals))
    write("tab_gallery",
          "Rank-1 identification (\\%) of LFW probes as synthetic distractors are "
          "added to the gallery. Full-precision and full-dimension binary "
          "templates are unaffected up to $10^6$ distractors, while reduced "
          "templates decay in proportion to their compression.",
          "tab:gallery",
          r"Template & $N{=}0$ & $10^4$ & $10^5$ & $10^6$",
          rows, "lcccc")


def t_rabitq():
    r11 = pd.read_csv(C.RESULTS_DIR / "rabitq_xqlfw.csv")
    r1n = pd.read_csv(C.RESULTS_DIR / "rabitq_tinyface_full.csv")
    rows = []
    for _, x in r11.iterrows():
        rows.append(f"{x['method']} & {int(x['bits'])} & {pct(x['acc'])}")
    write("tab_rabitq_11",
          "RaBitQ-style templates on XQLFW 1:1 verification (both templates "
          "compressed, corrected symmetric estimator). The per-vector correction "
          "adds score variance; the learned ITQ rotation remains the best binary "
          "code for verification.",
          "tab:rabitq_11",
          r"Method & Bits & XQLFW acc.\ (\%)", rows, "lcc")
    rows = []
    for _, x in r1n.iterrows():
        rows.append(f"{x['method']} & {int(x['bits'])} & {pct(x['rank1'])} & "
                    f"{pct(x['rank5'])}")
    write("tab_rabitq_1n",
          "RaBitQ-style templates on TinyFace 1:N retrieval (gallery quantized, "
          "probe full precision -- the asymmetric deployment setting). "
          "PCA+RaBitQ is the strongest low-bit template, recovering about a "
          "third of the gap to float32 at equal bits.",
          "tab:rabitq_1n",
          r"Method & Bits & Rank-1 (\%) & Rank-5 (\%)", rows, "lccc")


def t_pruning_ft():
    ft, pr = _csv("pruning_finetuned.csv"), _csv("pruning_sweep.csv")
    if ft is None or pr is None:
        print("SKIP tab_pruning_ft: pruning_finetuned.csv / pruning_sweep.csv missing")
        return
    mag = ft[ft["config"] == "magface"]
    base = pr[(pr["criterion"] == "none") & (pr["config"] == "magface")]

    def bval(d, col):
        return float(base[base["dataset"] == d][col].iloc[0])

    rows = [rf"0\,\% (unpruned) & 65.2 & 12.12 & {pct(bval('lfw', 'acc'))} & "
            rf"{pct(bval('xqlfw', 'acc'))} & {pct(bval('xqlfw', 'tar_far0.01'))}"]
    for ratio in sorted(mag["ratio"].unique()):
        m = mag[np.isclose(mag["ratio"], ratio)]

        def cellv(d, col):
            r = m[m["dataset"] == d]
            if r.empty:
                return "--"
            if float(r["acc"].iloc[0]) < 0.55:
                return r"\textit{coll.}" if col == "acc" else "--"
            v = float(r[col].iloc[0])
            return f"{pct(v)} ({dl(v, bval(d, col))})"

        rows.append(rf"{int(round(ratio * 100))}\,\% & "
                    rf"{float(m['params_m'].iloc[0]):.1f} & "
                    rf"{float(m['macs_g'].iloc[0]):.2f} & {cellv('lfw', 'acc')} & "
                    rf"{cellv('xqlfw', 'acc')} & {cellv('xqlfw', 'tar_far0.01')}")
    write("tab_pruning_ft",
          "Pruning with light recovery: after one-shot L1 pruning, each network "
          "is fine-tuned for ONE epoch of label-free self-distillation on "
          "CASIA-WebFace (490k images, disjoint from all evaluation sets), "
          "matching the unpruned teacher's embeddings (changes vs.\\ unpruned in "
          "parentheses). Accuracy recovers almost fully at 10--30\\,\\% -- but "
          "the strict operating point does not: XQLFW TAR@FAR=1\\% stays "
          "8--13 points below baseline. Adaptation repairs the average, not the "
          "tail. At 50\\,\\% the distillation plateaus and the network stays "
          "collapsed.",
          "tab:pruning_ft",
          r"Ratio & Params (M) & MACs (G) & LFW acc.\ & XQLFW acc.\ & "
          r"XQLFW TAR@1\%",
          rows, "lccccc")


def t_matched():
    """Fair three-way loss comparison: the ctrl bundles hold training data
    (MS1MV2) and backbone depth (iResNet100/100/101) fixed."""
    ov = pd.read_csv(C.RESULTS_DIR / "overlap_summary.csv").set_index("dataset")
    sigc = _csv("significance_ctrl.csv")

    def mstars(d):
        if sigc is None:
            return ""
        m = sigc[(sigc["dataset"] == f"{d}_ctrl") & (sigc["A"] == "arcface")
                 & (sigc["B"] == "adaface")]
        if m.empty:
            return ""
        return _mark(float(m["p_mcnemar"].iloc[0]), float(m["p_nb"].iloc[0]))

    rows = []
    for d in DATASETS:
        key = f"{d}_ctrl"
        if key not in ov.index:
            continue
        o = ov.loc[key]
        rows.append(f"{NICE[d]} & {pct(o['acc_arcface'])} & "
                    f"{pct(o['acc_magface'])} & {pct(o['acc_adaface'])} & "
                    f"{dl(o['acc_adaface'], o['acc_arcface'])}{mstars(d)}")
    if not rows:
        print("SKIP tab_matched: no *_ctrl rows in overlap_summary.csv")
        return
    write("tab_matched",
          "Matched-training comparison of the three losses: every model trained "
          "on MS1MV2 with iResNet backbones of comparable depth "
          "(ArcFace/MagFace: iResNet100; AdaFace: iResNet101). With training "
          "data equalized, AdaFace is the strongest model on every benchmark -- "
          "the ArcFace dominance elsewhere in this paper is a training-data "
          "advantage (WebFace600K, 600k identities vs.\\ 85k), not a property "
          "of the loss. Significance marks compare AdaFace vs.\\ ArcFace where "
          "\\texttt{significance\\_ctrl.csv} is available.",
          "tab:matched",
          r"Dataset & ArcFace & MagFace & AdaFace & $\Delta$ (Ada$-$Arc)",
          rows, "lcccc")


def t_fusion_ablation():
    fa, fs = _csv("fusion_ablation.csv"), _csv("fusion_ablation_significance.csv")
    if fa is None or fs is None:
        print("SKIP tab_fusion_ablation: fusion_ablation*.csv not found")
        return
    rows = []
    for d in DATASETS:
        g = fa[fa["dataset"] == d]
        if g.empty:
            continue
        best = g[g["n_models"] == 1].sort_values("acc", ascending=False).iloc[0]
        full = g[g["combo"] == "arc+mag+ada"].iloc[0]
        row = (f"{NICE[d]} & {best['combo']} ({pct(best['acc'])}) & "
               f"{pct(full['acc'])}")
        for m in ("arcface", "magface", "adaface"):
            s = fs[(fs["dataset"] == d) & (fs["dropped"] == m)]
            if s.empty:
                row += " & --"
            else:
                mk = _mark(float(s["p_mcnemar"].iloc[0]), float(s["p_nb"].iloc[0]))
                row += f" & {100 * float(s['diff'].iloc[0]):+.2f}{mk}"
        rows.append(row)
    write("tab_fusion_ablation",
          "Leave-one-out fusion ablation (accuracy \\%; $\\Delta$ = change when the "
          "model is removed from the fused concatenation, positive = removal "
          "helps). MagFace is a significant liability on the pose/quality "
          "benchmarks and the best single model outperforms every fusion on "
          "CPLFW and XQLFW: equal-weight concatenation dilutes the strong model "
          "with the weak one. On CALFW the (marginally) harmful member is "
          "ArcFace instead -- no fixed weighting suits all domains. LFW is "
          "evaluated on the official crops.",
          "tab:fusion_ablation",
          r"Dataset & Best single & Fused (all 3) & $\Delta$ drop arc & "
          r"$\Delta$ drop mag & $\Delta$ drop ada",
          rows, "lccccc")


def t_quant():
    qs = _csv("quant_significance.csv")

    def qstars(d, config):
        if qs is None:
            return ""
        m = qs[(qs["dataset"] == d) & (qs["config"] == config)]
        if m.empty:
            return ""
        return _mark(float(m["p_mcnemar"].iloc[0]), float(m["p_nb"].iloc[0]))

    rows = []
    for d in DATASETS:
        a32 = _csv("rd_lfw_bin_arc.csv" if d == "lfw" else f"rd_{d}_arc.csv")
        a8 = _csv(f"rd_{d}_q8arc.csv")
        if a32 is None or a8 is None:
            continue
        acc32, acc8 = cell(a32, "none", "fp32", "acc"), cell(a8, "none", "fp32", "acc")
        t32 = cell(a32, "none", "fp32", "tar_far0.01")
        t8 = cell(a8, "none", "fp32", "tar_far0.01")
        fd = "--"
        if qs is not None:
            m = qs[(qs["dataset"] == d) & (qs["config"] == "fused")]
            if not m.empty:
                diff = float(m["acc_int8net"].iloc[0]) - float(m["acc_fp32net"].iloc[0])
                fd = f"{100 * diff:+.2f}{qstars(d, 'fused')}"
        rows.append(f"{NICE[d]} & {pct(acc32)} & {pct(acc8)} "
                    f"({dl(acc8, acc32)}){qstars(d, 'arcface')} & "
                    f"{pct(t32)} & {pct(t8)} ({dl(t8, t32)}) & {fd}")
    if not rows:
        print("SKIP tab_quant: rd_*_arc / rd_*_q8arc CSVs not found")
        return
    write("tab_quant",
          "Network-level compression: static INT8 post-training quantization of "
          "the ArcFace backbone (per-channel QDQ, 256 calibration images; "
          "166\\,MB $\\to$ 42\\,MB, $4.1\\times$ faster on CPU). Accuracy and "
          "TAR@FAR=1\\% of the quantized network vs.\\ float32 (change in "
          "parentheses), plus the fused-template accuracy change. Quantization "
          "is statistically free on the easy benchmarks and its cost surfaces "
          "only at the hard end of the ladder -- the template-level law repeats "
          "at the network level. LFW rows use the official crops.",
          "tab:quant",
          r"Dataset & \multicolumn{2}{c}{ArcFace acc.\ (fp32 / int8 net)} & "
          r"\multicolumn{2}{c}{TAR@1\%FAR (fp32 / int8 net)} & Fused $\Delta$",
          rows, "lccccc")


def t_pruning():
    pr = _csv("pruning_sweep.csv")
    if pr is None:
        print("SKIP tab_pruning: pruning_sweep.csv not found")
        return
    pr = pr.dropna(subset=["acc"]) if "acc" in pr else pr
    mag = pr[pr["config"] == "magface"]

    def acc_of(crit, ratio, dataset):
        m = mag[(mag["criterion"] == crit) & np.isclose(mag["ratio"], ratio)
                & (mag["dataset"] == dataset)]
        if m.empty:
            return "--"
        v = float(m["acc"].iloc[0])
        return r"\textit{coll.}" if v < 0.55 else pct(v)

    base = mag[mag["criterion"] == "none"]
    b_lfw = float(base[base["dataset"] == "lfw"]["acc"].iloc[0])
    b_xq = float(base[base["dataset"] == "xqlfw"]["acc"].iloc[0])
    rows = [rf"0\,\% & 65.2 & 12.12 & \multicolumn{{6}}{{c}}{{unpruned MagFace: "
            rf"LFW {pct(b_lfw)}, XQLFW {pct(b_xq)}}}"]
    for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
        m = mag[np.isclose(mag["ratio"], ratio)]
        if m.empty:
            continue
        row = (rf"{int(round(ratio * 100))}\,\% & {float(m['params_m'].iloc[0]):.1f} & "
               rf"{float(m['macs_g'].iloc[0]):.2f}")
        for crit in ("l1", "bnscale", "random"):
            row += f" & {acc_of(crit, ratio, 'lfw')} & {acc_of(crit, ratio, 'xqlfw')}"
        rows.append(row)
    write("tab_pruning",
          "One-shot structured channel pruning of the MagFace iResNet100 "
          "(DepGraph / Torch-Pruning, no fine-tuning), verification accuracy "
          "(\\%) of the pruned model alone. \\textit{coll.} = collapsed to "
          "chance (degenerate embeddings). Even 10\\% L1 pruning, which looks "
          "nearly free on LFW ($-0.5$ points), destroys cross-quality "
          "performance -- and BN-scale importance underperforms random, since "
          "MagFace was trained without a sparsity regularizer on the BN scales. "
          "One-shot pruning without adaptation fails: the embedding-level "
          "redundancy does not transfer to the weights.",
          "tab:pruning",
          r"Ratio & Params (M) & MACs (G) & \multicolumn{2}{c}{L1} & "
          r"\multicolumn{2}{c}{BN-scale} & \multicolumn{2}{c}{Random} \\"
          "\n" + r" & & & LFW & XQLFW & LFW & XQLFW & LFW & XQLFW",
          rows, "lcccccccc")


# narrative order: setup -> fusion story -> compression story -> compressor
# upgrade -> network compression
ORDER = ["tab_baselines", "tab_overlap", "tab_fusion_learned",
         "tab_fusion_ablation", "tab_control", "tab_matched",
         "tab_compression_acc", "tab_compression_tar", "tab_tinyface",
         "tab_gallery", "tab_rabitq_11", "tab_rabitq_1n",
         "tab_quant", "tab_pruning", "tab_pruning_ft"]


def fig_block(files, caption, label, width):
    imgs = "\n".join(
        rf"\includegraphics[width={width}\textwidth]{{{f}}}" for f in files)
    return "\n".join([r"\begin{figure}[H]", r"\centering", imgs,
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\end{figure}"])


# figures inserted after the named table in the standalone document
FIGS_AFTER = {
    "tab_compression_tar": fig_block(
        ["rd_lfw", "rd_xqlfw"],
        "Rate--distortion curves on the easiest and hardest 1:1 benchmark "
        "(left: LFW, right: XQLFW; accuracy vs.\\ bits per template, log scale). "
        "On LFW every compression level sits on the float32 ceiling; on XQLFW "
        "the reduced templates fall away.",
        "fig:rd_easy_hard", 0.49),
    "tab_tinyface": fig_block(
        ["rd_tinyface_full"],
        "Rate--distortion on TinyFace 1:N identification (rank-1 vs.\\ bits per "
        "template, 153k real distractors). The compression cost appears at every "
        "bit budget, including full-dimension binary.",
        "fig:rd_tinyface", 0.65),
    "tab_gallery": fig_block(
        ["gallery_sweep"],
        "Rank-1 identification of LFW probes vs.\\ gallery size (log scale), one "
        "line per compression level. Full-precision and full-dimension binary "
        "templates are flat across four orders of magnitude; reduced templates "
        "decay in proportion to their compression.",
        "fig:gallery_sweep", 0.7),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_baselines(); t_compression_acc(); t_compression_tar(); t_overlap()
    t_fusion_learned(); t_control(); t_tinyface(); t_gallery(); t_rabitq()
    t_fusion_ablation(); t_quant(); t_pruning(); t_pruning_ft(); t_matched()
    names = [n for n in ORDER if (OUT / f"{n}.tex").exists()]
    names += sorted(p.stem for p in OUT.glob("tab_*.tex") if p.stem not in names)
    (OUT / "tables_all.tex").write_text(
        "\n".join(rf"\input{{tables/{n}}}" for n in names) + "\n", encoding="utf-8")
    print(f"wrote outputs/tables/tables_all.tex ({len(names)} tables)")

    # standalone document with every table + relevant figures -- compiles on its own
    doc = [r"\documentclass[11pt]{article}",
           r"\usepackage[a4paper,margin=2cm]{geometry}",
           r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}",
           r"\usepackage{booktabs}", r"\usepackage{float}",
           r"\usepackage{graphicx}",
           r"\graphicspath{{../figures/}}",
           r"\title{Generated result tables and figures\\[0.3em]\large SEAL / facecomp}",
           r"\date{\today}",
           r"\begin{document}",
           r"\maketitle",
           ""]
    for n in names:
        doc.append((OUT / f"{n}.tex").read_text(encoding="utf-8"))
        if n in FIGS_AFTER:
            doc.append(FIGS_AFTER[n])
            doc.append("")
    doc.append(r"\end{document}")
    (OUT / "all_tables.tex").write_text("\n".join(doc) + "\n", encoding="utf-8")
    print("wrote outputs/tables/all_tables.tex (standalone, tables + figures)")


if __name__ == "__main__":
    main()
