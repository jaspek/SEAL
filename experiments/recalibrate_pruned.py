"""BN-statistics recalibration after one-shot pruning: the cheapest possible
adaptation (forward passes only, no optimizer, no loss). Directly tests the
paper's claim that 'every survivor included a data-adaptation step'; stale BN
running statistics are a well-known cause of the collapse-to-degenerate-embeddings
the one-shot sweep observed. Produces a row that slots between the one-shot sweep
(pruning_sweep.csv) and the 1-epoch-distill table (pruning_finetuned.csv).

  python experiments/recalibrate_pruned.py --ratios 0.1 0.2 0.3 0.5 --batches 200
Writes outputs/results/pruning_bn_recalib.csv"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pruning import PRUNABLE, prune_model, load_data, fused_with   # noqa: E402
from finetune_pruned import build_dataset                              # noqa: E402

from facecomp import config as C                                       # noqa: E402
from facecomp.extract import models as M                              # noqa: E402
from facecomp.evaluate import verification as ver                     # noqa: E402


def recalibrate_bn(model, loader, device, n_batches):
    """Reset running mean/var on every BN layer and re-estimate from forward
    passes. Only BN modules go to train() (others stay eval so Dropout etc. do
    not inject noise). momentum=None -> cumulative moving average over batches."""
    bns = [m for m in model.modules()
           if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    seen = 0
    with torch.no_grad():
        for x in loader:
            model(x.to(device).contiguous())
            seen += 1
            if seen >= n_batches:
                break
    model.eval()
    return seen, len(bns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PRUNABLE), default="magface")
    ap.add_argument("--data", default=None, help="calibration crops; default data.casia")
    ap.add_argument("--ratios", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--criterion", default="l1")
    ap.add_argument("--datasets", nargs="*", default=["lfw", "xqlfw"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--batches", type=int, default=200, help="forward batches for BN")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pruning_bn_recalib.csv")
    args = ap.parse_args()
    loader_fn, pre_fn = PRUNABLE[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.data) if args.data else C.DATA.get("casia")
    if not root or not Path(root).is_dir():
        raise SystemExit("set data.casia in configs/paths.yaml or pass --data <folder>")
    ds = build_dataset(root, pre_fn, limit=args.batch * args.batches, seed=args.seed)
    if len(ds) == 0:
        raise SystemExit(f"no images found under {root}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=device == "cuda")
    data = load_data(args.datasets)

    rows = []
    for ratio in args.ratios:
        print(f"\n=== {args.model} {args.criterion} @ {ratio:.0%} + BN recalib ===")
        model, *_rest, params = prune_model(args.model, ratio, args.criterion)
        params_m = round(params / 1e6, 2)
        n, nb = recalibrate_bn(model, loader, device, args.batches)
        print(f"  recalibrated {nb} BN layers over {n} batches  ({params_m}M params)")
        for d, (images, pair_idx, per_model) in data.items():
            embp = M._embed_torch(model, images, pre_fn, f"  {args.model}[{d}]")
            evals = {args.model: embp}
            if per_model is not None:
                evals["fused"] = fused_with(args.model, embp, per_model)
            for cfg, embx in evals.items():
                acc, std = ver.evaluate_lfw(embx, pair_idx)
                row = {"model": args.model, "criterion": args.criterion, "ratio": ratio,
                       "adapt": "bn_recalib", "params_m": params_m,
                       "dataset": d, "config": cfg,
                       "acc": round(acc, 6), "acc_std": round(std, 6)}
                row.update(ver.roc_metrics(embx, pair_idx))
                rows.append(row)
                print(f"    {d}/{cfg}: acc {acc:.4f}  tar@1% {row['tar_far0.01']:.4f}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    C.ensure_dirs()
    out = Path(args.out)
    out = out if out.is_absolute() else C.RESULTS_DIR / out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}  (compare vs pruning_sweep.csv one-shot rows)")


if __name__ == "__main__":
    main()
