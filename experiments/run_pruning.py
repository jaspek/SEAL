"""One-shot structured pruning of a PyTorch backbone (DepGraph sweep).

NETWORK-compression experiment #2 (after static-INT8 PTQ): dependency-aware
channel pruning via Torch-Pruning (DepGraph, Fang et al. CVPR 2023), one-shot,
NO fine-tuning, sweeping prune ratio x importance criterion, then re-extracting
embeddings and scoring the standard 10-fold verification. Prunable targets are
the PyTorch members of the stack -- MagFace (default) and AdaFace; ArcFace
ships as ONNX, which DepGraph cannot trace. 'random' is the control criterion.

  pip install torch-pruning        # once
  python experiments/run_pruning.py                          # magface sweep
  python experiments/run_pruning.py --model adaface          # adaface sweep
  python experiments/run_pruning.py --ratios 0.1 0.3 --criteria l1 --datasets xqlfw

Per dataset it scores the pruned model alone AND the fused template with the
pruned model spliced into its slot (needs emb_lfw_bin.npz / emb_<d>.npz for
the other two fp32 members; fused eval is skipped if missing). Writes
outputs/results/pruning_sweep.csv (magface) or pruning_sweep_<model>.csv."""
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

# torch-prunable members: name -> (loader, per-image preprocessing)
PRUNABLE = {
    "magface": (M.load_magface, M._pre_magface),
    "adaface": (M.load_adaface, M._pre_adaface),
}


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


def prune_model(model_name, ratio, crit):
    """Fresh model, pruned in place. The Linear head and BatchNorm1d stay
    ignored so the embedding remains 512-d; DepGraph adjusts their in-features
    automatically when the last conv block loses channels."""
    model = PRUNABLE[model_name][0]()
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


def fused_with(pruned_name, pruned_emb, per_model):
    """Fused template with the pruned model spliced into its slot (arc,mag,ada
    order, matching EmbBundle.fuse)."""
    arrs = [pruned_emb if m == pruned_name else per_model[m] for m in E.MODELS]
    return E._l2(np.concatenate(arrs, axis=1))


def load_data(datasets):
    out = {}
    for d in datasets:
        bin_path = C.DATA["bin_pack"] / f"{d}.bin"
        print(f"loading {bin_path}")
        images, pair_idx = bin_loader.load_bin(bin_path)
        per_model = None
        npz = C.EMBEDDINGS_DIR / FUSED_NPZ.get(d, f"emb_{d}.npz")
        if npz.exists():
            b = E.load_emb(npz)
            if b.N == len(images) and all(m in b.per_model for m in E.MODELS):
                per_model = b.per_model
            else:
                print(f"  [{d}] {npz.name} does not match the bin; fused eval skipped")
        else:
            print(f"  [{d}] {npz.name} not found; fused eval skipped")
        out[d] = (images, pair_idx, per_model)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PRUNABLE), default="magface",
                    help="torch backbone to prune (ArcFace is ONNX -> not prunable)")
    ap.add_argument("--datasets", nargs="*", default=["lfw", "xqlfw"])
    ap.add_argument("--ratios", nargs="*", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    ap.add_argument("--criteria", nargs="*", default=["l1", "bnscale", "random"],
                    help="l1 | bnscale | random | fpgm (if the installed "
                         "torch-pruning provides it)")
    ap.add_argument("--out", default=None,
                    help="default: pruning_sweep.csv (magface) / pruning_sweep_<model>.csv")
    args = ap.parse_args()
    pre_fn = PRUNABLE[args.model][1]
    out_name = args.out or ("pruning_sweep.csv" if args.model == "magface"
                            else f"pruning_sweep_{args.model}.csv")

    data = load_data(args.datasets)
    configs = [("none", 0.0)] + [(c, r) for c in args.criteria for r in args.ratios]
    rows = []
    for crit, ratio in configs:
        print(f"\n=== {args.model}  criterion={crit}  ratio={ratio:.0%} ===")
        try:
            model, bmacs, bparams, macs, params = prune_model(args.model, ratio, crit)
        except Exception as e:
            print(f"  PRUNE FAILED: {e}")
            rows.append({"model": args.model, "criterion": crit, "ratio": ratio,
                         "error": str(e)[:200]})
            continue
        print(f"  params {bparams / 1e6:.1f}M -> {params / 1e6:.1f}M   "
              f"MACs {bmacs / 1e9:.2f}G -> {macs / 1e9:.2f}G")
        for d, (images, pair_idx, per_model) in data.items():
            embp = M._embed_torch(model, images, pre_fn, f"  {args.model}[{d}]")
            evals = {args.model: embp}
            if per_model is not None:
                evals["fused"] = fused_with(args.model, embp, per_model)
            for cfg, embx in evals.items():
                acc, std = ver.evaluate_lfw(embx, pair_idx)
                row = {"model": args.model, "criterion": crit, "ratio": ratio,
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
    out = Path(out_name)
    out = out if out.is_absolute() else C.RESULTS_DIR / out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
