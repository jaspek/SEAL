"""Unsupervised Matryoshka-Adaptor (Yu et al., EMNLP 2024, arXiv:2407.20243) for
face templates. Trains a small residual MLP that reshapes FROZEN embeddings so
truncated prefixes stay faithful, via a neighborhood/similarity-preservation loss
(no labels, no backbone access). Fit on a DISJOINT corpus, then evaluate truncated
128/256/64-d outputs against PCA-k/ITQ-k with the existing verification harness.

Directly probes the paper's 'dimension removal is progressively fatal' finding:
PCA/ITQ are linear unsupervised projections; a learned adaptor may reach the same
low dimension with far less XQLFW damage (or -- an even stronger result -- may not).

  python experiments/train_matryoshka_adaptor.py \
         --fit outputs/embeddings/emb_cplfw.npz \
         --eval outputs/embeddings/emb_xqlfw.npz --dims 256 128 64
Writes outputs/results/matryoshka_adaptor.csv"""
import argparse

import numpy as np
import pandas as pd
import torch

from facecomp import config as C
from facecomp.data import emb as E
from facecomp.evaluate.verification import evaluate_lfw


def l2n(x, eps=1e-10):
    return x / (x.norm(dim=1, keepdim=True) + eps)


class Adaptor(torch.nn.Module):
    def __init__(self, d, hidden=1024):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, d))

    def forward(self, x):
        return l2n(x + self.net(x))            # residual; keep full dim, truncate later


def matryoshka_loss(z, x, dims):
    """Preserve the frozen-embedding pairwise similarity structure at every nested
    prefix dimension (batch-local Gram-matrix matching)."""
    with torch.no_grad():
        target = x @ x.T
    loss = 0.0
    for d in dims:
        zd = l2n(z[:, :d])
        loss = loss + ((zd @ zd.T - target) ** 2).mean()
    return loss / len(dims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True, help="disjoint emb_*.npz to train on")
    ap.add_argument("--eval", required=True, help="emb_*.npz to evaluate (needs pair_idx)")
    ap.add_argument("--dims", nargs="*", type=int, default=[256, 128, 64])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="matryoshka_adaptor.csv")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    Xf = torch.tensor(E.load_emb(args.fit).fuse(), device=dev)
    d = Xf.shape[1]
    model = Adaptor(d).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    for ep in range(args.epochs):
        idx = torch.randint(0, len(Xf), (min(args.batch, len(Xf)),),
                            generator=gen, device=dev)
        xb = l2n(Xf[idx])
        loss = matryoshka_loss(model(xb), xb, args.dims)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  ep {ep + 1}: loss {float(loss):.5f}")

    beval = E.load_emb(args.eval)
    if beval.pair_idx is None:
        raise SystemExit("--eval npz needs pair_idx for verification")
    Xe = torch.tensor(beval.fuse(), device=dev)
    with torch.no_grad():
        Z = model(l2n(Xe)).cpu().numpy()

    rows = []
    for dim in args.dims:
        sub = Z[:, :dim]
        sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-10)
        acc, _ = evaluate_lfw(sub.astype(np.float32), beval.pair_idx)
        rows.append({"eval": args.eval, "dim": dim, "acc_adaptor": round(acc, 6)})
        print(f"  adaptor dim {dim}: acc {acc:.4f}  (compare to PCA{dim}/ITQ{dim} rows)")
    C.ensure_dirs()
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / args.out, index=False)
    print("wrote", C.RESULTS_DIR / args.out)


if __name__ == "__main__":
    main()
