#!/usr/bin/env python3
"""
2:4 semi-structured sparsity (+ optional INT8).   ===> HOME RTX-3060 BOX ONLY <===

This is the ONE network experiment that does NOT belong on the CPU machine: the
~2x throughput win from 2:4 sparsity needs Ampere Sparse Tensor Cores (your
RTX 3060). It COMPOSES with INT8, fulfilling the "pruning x quantization, real
hardware speed-up" promise on the exact GPU in the seminar.

Pipeline (run on the GPU box):
  dense -> apply 2:4 mask -> (light fine-tune) -> INT8 -> benchmark on CUDA.

  # PyTorch native 2:4 (Ampere):
  from torch.sparse import to_sparse_semi_structured
  import torch.ao.pruning as sparsity

Skeleton below; wire in your model. On CPU this script will run but give NO
speed-up (the mask is applied but dense kernels are used) — measure it on CUDA.
"""
import torch


def apply_2to4(model):
    """Apply a 2:4 magnitude mask to every prunable Linear/Conv weight."""
    from torch.ao.pruning import WeightNormSparsifier
    sparsifier = WeightNormSparsifier(
        sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2)
    cfg = [{"tensor_fqn": f"{n}.weight"}
           for n, m in model.named_modules()
           if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))]
    sparsifier.prepare(model, cfg)
    sparsifier.step()
    sparsifier.squash_mask()
    return model


def main():
    raise NotImplementedError(
        "Run on the RTX-3060 box. Load your model, call apply_2to4(model), "
        "optionally fine-tune a few epochs, convert weights with "
        "torch.sparse.to_sparse_semi_structured, then INT8 + benchmark on CUDA. "
        "Compare within-device (GPU) latency vs the dense FP16 baseline.")


if __name__ == "__main__":
    main()
