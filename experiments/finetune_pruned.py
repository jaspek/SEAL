"""Recovery fine-tuning of pruned MagFace by SELF-DISTILLATION.

Tests the paper's open prediction (Hooker et al.: retraining repairs the
average, not the tail): prune MagFace one-shot (same DepGraph path as
run_pruning.py), then train the pruned STUDENT to reproduce the unpruned
TEACHER's embeddings -- cosine loss, no identity labels, no classification
head -- and re-evaluate on LFW + XQLFW exactly like the one-shot sweep, so
the rows in pruning_finetuned.csv line up with pruning_sweep.csv.

Training data: any folder of face crops, e.g. CASIA-WebFace (~490k images).
Images are resized to 112x112 if needed; teacher and student always see the
SAME input, so imperfect alignment only makes the target slightly noisier.
NEVER point --data at images from the evaluation benchmarks (LFW family,
TinyFace) -- that would be test contamination.

  python experiments/finetune_pruned.py                          # full run
  python experiments/finetune_pruned.py --limit 100000 --ratios 0.3   # quick

Runtime on an RTX 3060: roughly 45-70 min per (ratio, epoch) at 490k images;
the default 3 ratios x 1 epoch is an overnight-safe ~3-4 h. Checkpoints go
to outputs/models/ (gitignored)."""
import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
import torch_pruning as tp
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pruning import PRUNABLE, prune_model, load_data, fused_with  # noqa: E402

from facecomp import config as C                          # noqa: E402
from facecomp.data import emb as E                        # noqa: E402
from facecomp.extract import models as M                  # noqa: E402
from facecomp.evaluate import verification as ver         # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_REC_MAGIC = 0xCED7230A


def _prep(img, pre):
    if img is None:
        img = np.zeros((112, 112, 3), np.uint8)
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112))
    return torch.from_numpy(pre(img))


class RecFile(Dataset):
    """InsightFace MXNet RecordIO pack (train.rec + train.idx), read without
    mxnet. Record layout: magic u32, length u32 (low 29 bits), then payload =
    24-byte IRHeader (flag u32, label f32, id u64, id2 u64) + flag*4 bytes of
    float labels (when flag>0) + JPEG bytes. Record key 0 is a meta header
    whose label[0] marks the end of the image records (InsightFace layout)."""

    def __init__(self, rec_path, idx_path, pre, limit=None, seed=0):
        self.rec_path = str(rec_path)
        self.pre = pre                     # module-level fn -> picklable to workers
        offsets = {}
        with open(idx_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    offsets[int(parts[0])] = int(parts[1])
        self._f = open(self.rec_path, "rb")
        flag, label, _ = self._read(offsets[0])
        if flag > 0 and label is not None:
            keys = [k for k in range(1, int(label[0])) if k in offsets]
        else:
            keys = sorted(k for k in offsets if k != 0)
        self._f.close()
        self._f = None                      # workers reopen their own handle
        rng = np.random.default_rng(seed)
        rng.shuffle(keys)
        self.offsets = [offsets[k] for k in (keys[:limit] if limit else keys)]

    def _read(self, offset):
        if self._f is None:
            self._f = open(self.rec_path, "rb")
        self._f.seek(offset)
        magic, lrec = struct.unpack("<II", self._f.read(8))
        if magic != _REC_MAGIC:
            raise IOError(f"bad record magic at offset {offset}")
        data = self._f.read(lrec & ((1 << 29) - 1))
        flag = struct.unpack("<I", data[:4])[0]
        label = np.frombuffer(data[24:24 + 4 * flag], np.float32) if flag else None
        payload = data[24 + 4 * flag:] if flag else data[24:]
        return flag, label, payload

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, i):
        _, _, payload = self._read(self.offsets[i])
        img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        return _prep(img, self.pre)


class FaceFolder(Dataset):
    """Recursively collects face crops; returns MagFace-preprocessed CHW floats."""

    def __init__(self, root, pre, limit=None, seed=0):
        files = [p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS]
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        self.files = files[:limit] if limit else files
        self.pre = pre

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return _prep(cv2.imread(str(self.files[i])), self.pre)


def build_dataset(root, pre, limit=None, seed=0):
    """Folder of images, a .rec file, or a folder containing train.rec."""
    root = Path(root)
    if root.suffix == ".rec":
        return RecFile(root, root.with_suffix(".idx"), pre, limit, seed)
    recs = sorted(root.rglob("*.rec"))
    if recs:
        rec = next((r for r in recs if r.stem == "train"), recs[0])
        print(f"found RecordIO pack: {rec}")
        return RecFile(rec, rec.with_suffix(".idx"), pre, limit, seed)
    return FaceFolder(root, pre, limit, seed)


def _emb(model, x):
    y = model(x)
    y = y[0] if isinstance(y, tuple) else y
    return torch.nn.functional.normalize(y.float(), dim=1)


def distill(student, teacher, loader, epochs, lr, device, ckpt=None):
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    student.train()
    step, t0, running = 0, time.perf_counter(), 0.0
    for ep in range(epochs):
        for x in loader:
            x = x.to(device, non_blocking=True).contiguous()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda"):
                t = _emb(teacher, x)
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                s = _emb(student, x)
                loss = (1.0 - (t * s).sum(dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss)
            step += 1
            if step % 200 == 0:
                ips = 200 * x.shape[0] / (time.perf_counter() - t0)
                print(f"    ep {ep + 1} step {step}: loss {running / 200:.4f}"
                      f"  ({ips:.0f} img/s)")
                running, t0 = 0.0, time.perf_counter()
        if ckpt is not None:                       # survive freezes/crashes
            torch.save(student, ckpt)
            print(f"    ep {ep + 1} done -> checkpoint {ckpt}")
    student.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PRUNABLE), default="magface",
                    help="torch backbone to prune + recover")
    ap.add_argument("--data", default=None,
                    help="folder of face crops (e.g. CASIA-WebFace); "
                         "default: data.casia from configs/paths.yaml")
    ap.add_argument("--ratios", nargs="*", type=float, default=[0.1, 0.3, 0.5])
    ap.add_argument("--criterion", default="l1")
    ap.add_argument("--datasets", nargs="*", default=["lfw", "xqlfw"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on training images (for quick runs)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="continue distilling from a saved checkpoint .pt "
                         "(use with exactly one --ratios value)")
    ap.add_argument("--out", default=None,
                    help="default: pruning_finetuned.csv (magface) / "
                         "pruning_finetuned_<model>.csv")
    args = ap.parse_args()
    loader_fn, pre_fn = PRUNABLE[args.model]
    out_name = args.out or ("pruning_finetuned.csv" if args.model == "magface"
                            else f"pruning_finetuned_{args.model}.csv")

    root = Path(args.data) if args.data else C.DATA.get("casia")
    if not root or not Path(root).is_dir():
        raise SystemExit("training-data folder not found -- set data.casia in "
                         "configs/paths.yaml or pass --data <folder>")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset(root, pre_fn, args.limit, args.seed)
    if len(ds) == 0:
        raise SystemExit(f"no images found under {root}")
    print(f"training images: {len(ds)} from {root}  (device: {device})")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=device == "cuda")

    data = load_data(args.datasets)
    teacher = loader_fn().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    rows = []
    for ratio in args.ratios:
        print(f"\n=== prune {args.model} {args.criterion} @ {ratio:.0%} "
              f"+ distill {args.epochs} epoch(s) ===")
        if args.resume:
            if len(args.ratios) != 1:
                raise SystemExit("--resume needs exactly one --ratios value")
            M._add_repo_paths()
            student = torch.load(args.resume, map_location=device)
            ex = torch.randn(1, 3, 112, 112, device=device)
            bmacs, bparams = tp.utils.count_ops_and_params(teacher, ex)
            macs, params = tp.utils.count_ops_and_params(student, ex)
            print(f"  resumed from {args.resume}")
        else:
            student, bmacs, bparams, macs, params = prune_model(
                args.model, ratio, args.criterion)
        print(f"  params {bparams / 1e6:.1f}M -> {params / 1e6:.1f}M   "
              f"MACs {bmacs / 1e9:.2f}G -> {macs / 1e9:.2f}G")
        ckpt = C.REPO_ROOT / "outputs" / "models" / \
            f"{args.model}_pruned_{args.criterion}{int(round(ratio * 100))}_ft.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        distill(student, teacher, loader, args.epochs, args.lr, device, ckpt=ckpt)
        torch.save(student, ckpt)          # whole module: pruned shapes included
        print(f"  saved {ckpt}")

        for d, (images, pair_idx, per_model) in data.items():
            embp = M._embed_torch(student, images, pre_fn, f"  {args.model}[{d}]")
            evals = {args.model: embp}
            if per_model is not None:
                evals["fused"] = fused_with(args.model, embp, per_model)
            for cfg, embx in evals.items():
                acc, std = ver.evaluate_lfw(embx, pair_idx)
                row = {"model": args.model, "criterion": args.criterion,
                       "ratio": ratio, "epochs": args.epochs, "n_train": len(ds),
                       "params_m": round(params / 1e6, 2),
                       "macs_g": round(macs / 1e9, 2),
                       "dataset": d, "config": cfg,
                       "acc": round(acc, 6), "acc_std": round(std, 6)}
                row.update(ver.roc_metrics(embx, pair_idx))
                rows.append(row)
                print(f"    {d}/{cfg}: acc {acc:.4f}  tar@1% {row['tar_far0.01']:.4f}")
        del student
        if device == "cuda":
            torch.cuda.empty_cache()

    C.ensure_dirs()
    out = Path(out_name)
    out = out if out.is_absolute() else C.RESULTS_DIR / out
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}  (compare against pruning_sweep.csv one-shot rows)")


if __name__ == "__main__":
    main()
