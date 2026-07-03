"""One-shot structured pruning of the MagFace iResNet100 (DepGraph sweep).

NETWORK-compression experiment #2 (after static-INT8 PTQ): dependency-aware
channel pruning via Torch-Pruning (DepGraph, Fang et al. CVPR 2023), one-shot,
NO fine-tuning, sweeping prune ratio x importance criterion, then re-extracting
embeddings and scoring the standard 10-fold verification. MagFace is the
target because it is the stack's PyTorch backbone (ArcFace ships as ONNX,
which DepGraph cannot trace). This directly tests the hypothesis that the
embedding-level redundancy (256 of 1536 fused dims suffice) extends to the
weights -- and 'random' is included as the control criterion.

  pip install torch-pruning        # once
  python experiments/run_pruning.py                          # ~40-60 min on GPU
  python experiments/run_pruning.py --ratios 0.1 0.3 --criteria l1 --datasets xqlfw

Per dataset it scores the pruned MagFace alone AND the fused template with the
pruned MagFace inside (needs emb_lfw_bin.npz / emb_<d>.npz for the fp32
arcface+adaface columns; fused eval is skipped if missing). Writes
outputs/results/pruning_sweep.csv with params / MACs per configuration."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_pruning as tp

from facecomp import config as C
from facecomp.data import bin_loader, emb as E
from facecomp.extract import models as M
from facecomp.evaluate import verification as ver

FUSED_NPZ = {"lfw": "emb_lfw_bin.npz"}   # lfw fused baseline = official-crop build


def importance(name):
    if name == "l1":
        cls = getattr(tp.importance, "GroupNormImportance", None) \
            or tp.importance.MagnitudeImportance
        return cls(p=1)
    if name == "bnscale":
        return tp.importance.BNScaleImportance()
    if name == "random":
        return tp.importance.RandomImportance()
    if name == "fpgm":
        cls = getattr(tp.importance, "FPGMImportance", None)
        if cls is None:
            raise SystemExit("this torch-pruning version has no FPGMImportance; drop 'fpgm'")
        return cls()
    raise SystemExit(f"unknown criterion {name!r}")


def prune_magface(ratio, crit):
    """Fresh model, pruned in place. The Linear head and BatchNorm1d stay
    ignored so the embedding remains 512-d; DepGraph adjusts their in-features
    automatically when the last conv block loses channels."""
    model = M.load_magface()
    ex = torch.randn(1, 3, 112, 112, device=next(model.parameters()).device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, ex)
    if ratio > 0:
        ignored = [m for m in model.modules()
                   if isinstance(m, (torch.nn.Linear, torch.nn.BatchNorm1d))]
        kw = dict(importance=importance(crit), ignored_layers=ignored,
                  global_pruning=False)
        try:
            pruner = tp.pruner.MetaPruner(model, ex, pruning_ratio=ratio, **kw)
        except TypeError:                                   # older torch-pruning API
            pruner = tp.pruner.MetaPruner(model, ex, ch_sparsity=ratio, **kw)
        pruner.step()
    macs, params = tp.utils.count_ops_and_params(model, ex)
    with torch.no_grad():
        out = model(ex)
        out = out[0] if isinstance(out, tuple) else out
    assert out.shape[-1] == 512, f"embedding dim changed: {tuple(out.shape)}"
    return model, base_macs, base_params, macs, params


def load_data(datasets):
    out = {}
    for d in datasets:
        bin_path = C.DATA["bin_pack"] / f"{d}.bin"
        print(f"loading {bin_path}")
        images, pair_idx = bin_loader.load_bin(bin_path)
        arc = ada = None
        npz = C.EMBEDDINGS_DIR / FUSED_NPZ.get(d, f"emb_{d}.npz")
        if npz.exists():
            b = E.load_emb(npz)
            if b.N == len(images) and "arcface" in b.per_model and "adaface" in b.per_model:
                arc, ada = b.per_model["arcface"], b.per_model["adaface"]
            else:
                print(f"  [{d}] {npz.name} does not match the bin; fused eval skipped")
        else:
            print(f"  [{d}] {npz.name} not found; fused eval skipped")
        out[d] = (images, pair_idx, arc, ada)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["lfw", "xqlfw"])
    ap.add_argument("--ratios", nargs="*", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    ap.add_argument("--criteria", nargs="*", default=["l1", "bnscale", "random"],
                    help="l1 | bnscale | random | fpgm (if the installed "
                         "torch-pruning provides it)")
    ap.add_argument("--out", default="pruning_sweep.csv")
    args = ap.parse_args()

    data = load_data(args.datasets)
    configs = [("none", 0.0)] + [(c, r) for c in args.criteria for r in args.ratios]
    rows = []
    for crit, ratio in configs:
        print(f"\n=== criterion={crit}  ratio={ratio:.0%} ===")
        try:
            model, bmacs, bparams, macs, params = prune_magface(ratio, crit)
        except Exception as e:
            print(f"  PRUNE FAILED: {e}")
            rows.append({"criterion": crit, "ratio": ratio, "error": str(e)[:200]})
            continue
        print(f"  params {bparams / 1e6:.1f}M -> {params / 1e6:.1f}M   "
              f"MACs {bmacs / 1e9:.2f}G -> {macs / 1e9:.2f}G")
        for d, (images, pair_idx, arc, ada) in data.items():
            magp = M._embed_torch(model, images, M._pre_magface, f"  magface[{d}]")
            evals = {"magface": magp}
            if arc is not None:
                evals["fused"] = E._l2(np.concatenate([arc, magp, ada], axis=1))
            for cfg, embx in evals.items():
                acc, std = ver.evaluate_lfw(embx, pair_idx)
                row = {"criterion": crit, "ratio": ratio,
                       "params_m": round(params / 1e6, 2),
                       "macs_g": round(macs / 1e9, 2),
                       "dataset": d, "config": cfg,
                       "acc": round(acc, 6), "acc_std": round(std, 6)}
                row.update(ver.roc_metrics(embx, pair_idx))
                rows.append(row)
                print(f"    {d}/{cfg}: acc {acc:.4f}  tar@1% {row['tar_far0.01']:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    C.ensure_dirs()
    out = Path(args.out)
    out = out if out.is_absolute() else C.RESULTS_DIR / out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
