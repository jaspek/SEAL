"""RaBitQ vs PCA/ITQ binary templates: can a better compressor rescue the
low-bit template where it cracked (XQLFW 1:1, TinyFace 1:N)?

  python experiments/run_rabitq.py --emb outputs/embeddings/emb_xqlfw.npz
  python experiments/run_rabitq.py --emb outputs/embeddings/emb_tinyface_full.npz

Writes outputs/results/rabitq_<dataset>.csv"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.compress import template as T
from facecomp.compress.rabitq import RaBitQ
from facecomp.evaluate import identification as idn
from facecomp.evaluate.verification import cosine_scores


def _kfold_acc(scores, labels, n_splits=10):
    """Threshold grid from train-score quantiles (corrected RaBitQ scores are
    not bounded to [-1,1], so a fixed grid would clip the optimum)."""
    accs = []
    for tr, te in KFold(n_splits=n_splits, shuffle=False).split(scores):
        grid = np.quantile(scores[tr], np.linspace(0.001, 0.999, 400))
        best_acc, best_thr = -1.0, grid[0]
        for thr in grid:
            a = ((scores[tr] > thr) == labels[tr]).mean()
            if a > best_acc:
                best_acc, best_thr = a, thr
        accs.append(float(((scores[te] > best_thr) == labels[te]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def run_11(name, fused, pair_idx, ks=(128, 256)):
    labels = pair_idx[:, 2]
    rows = []

    def add(method, bits, scores):
        acc, std = _kfold_acc(scores, labels)
        rows.append({"dataset": name, "task": "1:1", "method": method,
                     "bits": bits, "acc": round(acc, 6), "acc_std": round(std, 6)})
        print(f"  {method:>22} ({bits:5d} bits): acc {acc:.4f} ± {std:.4f}")

    for k in ks:
        pca_emb, _ = T.compress(fused, "binary", T.PCAReducer(k).fit(fused))
        add(f"pca{k}+sign", k, cosine_scores(pca_emb, pair_idx))
        itq_emb, _ = T.compress(fused, "binary", T.ITQReducer(k).fit(fused))
        add(f"itq{k}+sign", k, cosine_scores(itq_emb, pair_idx))
        rq = RaBitQ(k).fit(fused)
        codes, c = rq.encode(fused)
        add(f"rabitq{k}", rq.bits(), rq.score_sym(codes, c, pair_idx))
        rqp = RaBitQ(k, pca_front=True).fit(fused)
        codes, c = rqp.encode(fused)
        add(f"pca+rabitq{k}", rqp.bits(), rqp.score_sym(codes, c, pair_idx))

    bin_emb, bits = T.compress(fused, "binary")
    add("binary-1536", bits, cosine_scores(bin_emb, pair_idx))
    add("fp32-1536", fused.shape[1] * 32, cosine_scores(fused, pair_idx))
    return rows


def run_1n(name, fused, labels, split, ks=(128, 256)):
    g_idx = np.where(split != 1)[0]
    p_idx = np.where(split == 1)[0]
    rows = []

    def add(method, bits, gemb, pemb):
        res = idn.cmc_and_map(gemb, gemb_labels, pemb, labels[p_idx])
        rows.append({"dataset": name, "task": "1:N", "method": method, "bits": bits,
                     "rank1": round(res["rank-1"], 6), "rank5": round(res["rank-5"], 6),
                     "mAP": round(res["mAP"], 6)})
        print(f"  {method:>22} ({bits:5d} bits): rank1 {res['rank-1']:.4f}  "
              f"rank5 {res['rank-5']:.4f}")

    gemb_labels = labels[g_idx]
    for k in ks:
        pca = T.PCAReducer(k).fit(fused)
        comp, _ = T.compress(fused, "binary", pca)
        add(f"pca{k}+sign", k, comp[g_idx], comp[p_idx])
        itq = T.ITQReducer(k).fit(fused)
        comp, _ = T.compress(fused, "binary", itq)
        add(f"itq{k}+sign", k, comp[g_idx], comp[p_idx])
        # RaBitQ asymmetric: gallery quantized (+correction), probe full precision
        rq = RaBitQ(k).fit(fused)
        codes, c = rq.encode(fused[g_idx])
        add(f"rabitq{k}-asym", rq.bits(), rq.gallery_effective(codes, c),
            rq.rotate(fused[p_idx]))
        rqp = RaBitQ(k, pca_front=True).fit(fused)
        codes, c = rqp.encode(fused[g_idx])
        add(f"pca+rabitq{k}-asym", rqp.bits(), rqp.gallery_effective(codes, c),
            rqp.rotate(fused[p_idx]))

    comp, bits = T.compress(fused, "binary")
    add("binary-1536", bits, comp[g_idx], comp[p_idx])
    add("fp32-1536", fused.shape[1] * 32, fused[g_idx], fused[p_idx])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="+", required=True)
    args = ap.parse_args()

    for path in args.emb:
        name = Path(path).stem
        for pre, post in (("emb_", ""), ("_embeddings", "")):
            name = name.replace(pre, post)
        b = E.load_emb(path)
        fused = b.fuse()
        print(f"\n=== {name} ===")
        if b.pair_idx is not None:
            rows = run_11(name, fused, b.pair_idx)
        elif b.labels is not None and b.split is not None:
            rows = run_1n(name, fused, b.labels, b.split)
        else:
            print("  SKIP: neither pair_idx nor predefined 1:N split")
            continue
        C.ensure_dirs()
        out = C.RESULTS_DIR / f"rabitq_{name}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
