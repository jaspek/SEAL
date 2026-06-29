#!/usr/bin/env python3
"""
ONE-SHOT structured (channel/filter) pruning of a PyTorch iResNet recognizer,
NO retraining, swept over prune ratios.    ===> PRUNING + COUNTS RUN ON CPU <===
(The optional fine-tune to recover accuracy is faster on the home GPU.)

This tests the paper's strongest conceptual hypothesis: that the *embedding*
redundancy (256 of 1536 dims suffice) extends to the *weights*, so one-shot
pruning is nearly free. DepGraph/Torch-Pruning handles the residual-coupling that
makes hand-pruning an iResNet break.

  pip install torch torch-pruning
  python prune_oneshot.py            # edit load_model() for your checkpoint

Outputs, per ratio: params, MACs, and a pruned state_dict you then feed back
through your extractor -> run_rate_distortion.py to get accuracy-vs-compression.
"""
import argparse
import torch
import torch_pruning as tp


def load_model():
    """RETURN your recognizer in eval mode on CPU.
    Plug in your MagFace/AdaFace iResNet exactly as you load it for inference,
    e.g.:
        from your_repo import build_iresnet100
        m = build_iresnet100(); m.load_state_dict(torch.load(CKPT, map_location='cpu'))
        return m.eval()
    """
    raise NotImplementedError("wire in your checkpoint here")


def find_embedding_head(model):
    """Return the final layer(s) that produce the 512-d embedding so the pruner
    keeps the output dimension at 512. Usually the last Linear/BN. Adjust names."""
    ignored = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 512:
            ignored.append(m)
    return ignored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", nargs="+", type=float,
                    default=[0.1, 0.3, 0.5, 0.7])
    ap.add_argument("--criterion", default="l2",
                    choices=["l2", "l1", "bn"])
    ap.add_argument("--out-prefix", default="pruned")
    args = ap.parse_args()

    example = torch.randn(1, 3, 112, 112)
    imp = {
        "l2": tp.importance.MagnitudeImportance(p=2),
        "l1": tp.importance.MagnitudeImportance(p=1),
        "bn": tp.importance.BNScaleImportance(),       # Network-Slimming style
    }[args.criterion]

    base = load_model()
    base_macs, base_params = tp.utils.count_ops_and_params(base, example)
    print(f"baseline: {base_params/1e6:.2f} M params, {base_macs/1e9:.2f} GMACs")

    for r in args.ratios:
        model = load_model()                            # fresh copy each ratio
        pruner = tp.pruner.MetaPruner(
            model, example, importance=imp,
            pruning_ratio=r,
            ignored_layers=find_embedding_head(model),
        )
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, example)
        torch.save(model.state_dict(), f"{args.out_prefix}_r{r:.2f}.pt")
        print(f"ratio={r:.2f}: {params/1e6:.2f} M params "
              f"({base_params/params:.2f}x), {macs/1e9:.2f} GMACs "
              f"({base_macs/macs:.2f}x)  -> {args.out_prefix}_r{r:.2f}.pt")

    print("\nNext: extract embeddings with each pruned model and run "
          "run_rate_distortion.py to plot accuracy vs prune-ratio "
          "(one-shot, no retrain). Then optionally light-fine-tune on the GPU "
          "box and re-measure to show how little tuning recovers.")


if __name__ == "__main__":
    main()
