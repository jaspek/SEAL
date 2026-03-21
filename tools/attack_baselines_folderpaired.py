"""
attack_baselines_folderpaired.py

Learning-based attacks on partially-encrypted (decoded) face images:
- baseline (no reconstruction)
- U-Net reconstruction
- DnCNN denoising
- Pix2Pix image2image
- embedding-space regressor (optional)

Key changes vs your pasted script:
1) Folder-based pairing: enc_root/pXX/<rel>  <->  clean_root/pYY/<rel>
2) Identity-disjoint train/val split (by identity folder)
3) Correct identity-based verification protocol (genuine/impostor pairs)
4) Fixed argparse + config (removed --root / pairs_csv)
5) Fixed evaluation to use roc_metrics_identity (and no broken roc_metrics_from_embeddings)
"""

from __future__ import annotations
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# (optional but often helps)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_curve, auc


# -------------------------
# Optional SSIM (piq)
# -------------------------
try:
    import piq
    HAVE_PIQ = True
except Exception:
    HAVE_PIQ = False


def _ssim_fallback(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Pure-PyTorch SSIM (per-sample) when piq is not installed.
    pred/target: B,C,H,W in [0,1]. Returns tensor of shape (B,).
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Average over channels first
    pred_gray = pred.mean(dim=1, keepdim=True)    # B,1,H,W
    target_gray = target.mean(dim=1, keepdim=True)

    # Gaussian-like uniform window
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device) / (window_size * window_size)

    mu_p = F.conv2d(pred_gray, kernel, padding=pad)
    mu_t = F.conv2d(target_gray, kernel, padding=pad)

    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_pp = F.conv2d(pred_gray * pred_gray, kernel, padding=pad) - mu_pp
    sigma_tt = F.conv2d(target_gray * target_gray, kernel, padding=pad) - mu_tt
    sigma_pt = F.conv2d(pred_gray * target_gray, kernel, padding=pad) - mu_pt

    ssim_map = ((2.0 * mu_pt + C1) * (2.0 * sigma_pt + C2)) / \
               ((mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2))

    # Mean per sample
    return ssim_map.flatten(1).mean(dim=1)


# -------------------------
# InsightFace embeddings
# -------------------------
try:
    from insightface.app import FaceAnalysis
    HAVE_INSIGHTFACE = True
except Exception:
    HAVE_INSIGHTFACE = False


# =========================
# Config
# =========================

@dataclass
class AttackConfig:
    # data
    enc_root: str                 # e.g. data/dec_enc_png_merged_0_100
    clean_root: str               # e.g. data/dec_plain_png_merged_0_100 (or same as enc_root if p00 is clean)
    percent: int                  # e.g. 30 -> uses enc_root/p30
    clean_percent: int = 0        # usually 0 -> uses clean_root/p00

    img_size: int = 112
    batch_size: int = 32
    num_workers: int = 4

    # training
    epochs: int = 5
    lr: float = 2e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # eval (identity verification)
    eval_pairs: int = 2000        # total verification pairs, split into 50/50 genuine/impostor
    outdir: str = "attack_results"

    # insightface
    insightface_name: str = "buffalo_l"
    det_size: Tuple[int, int] = (320, 320)

    # GAN
    gan_lambda_l1: float = 100.0

    # saving examples
    save_examples: bool = True
    examples_per_attack: int = 8


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Utilities: image I/O
# =========================

def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor_01(img_rgb: np.ndarray) -> torch.Tensor:
    # HWC uint8 -> CHW float32 [0,1]
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return t


def to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    # x: CHW float [0,1] -> HWC uint8
    x = x.detach().clamp(0, 1).cpu()
    x = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
    return x


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred/target in [0,1]
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1)  # per-sample
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr


def ssim_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred/target: B,C,H,W in [0,1]
    if HAVE_PIQ:
        return piq.ssim(pred, target, data_range=1.0, reduction="none")
    return _ssim_fallback(pred, target)


def save_triplet_grid(enc_bchw: torch.Tensor, rec_bchw: torch.Tensor, clean_bchw: torch.Tensor,
                      out_path: str, max_rows: int = 8):
    """
    Saves a grid image:
    Row i: [ENC | REC | CLEAN]
    enc/rec/clean: B,C,H,W in [0,1]
    """
    os.makedirs(str(Path(out_path).parent), exist_ok=True)

    b = min(enc_bchw.shape[0], max_rows)
    enc = enc_bchw[:b]
    rec = rec_bchw[:b]
    clean = clean_bchw[:b]

    rows = []
    for i in range(b):
        e = to_uint8_rgb(enc[i])     # HWC
        r = to_uint8_rgb(rec[i])
        c = to_uint8_rgb(clean[i])
        row = np.concatenate([e, r, c], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)


def save_diff_heatmap(enc_bchw: torch.Tensor, rec_bchw: torch.Tensor, clean_bchw: torch.Tensor,
                      out_path: str, max_rows: int = 8):
    """
    Saves a grid with absolute error heatmaps:
    Row i: [ENC | REC | CLEAN | heatmap(|REC-CLEAN|)]
    """
    os.makedirs(str(Path(out_path).parent), exist_ok=True)

    b = min(enc_bchw.shape[0], max_rows)
    enc = enc_bchw[:b]
    rec = rec_bchw[:b]
    clean = clean_bchw[:b]

    rows = []
    for i in range(b):
        e = to_uint8_rgb(enc[i])
        r = to_uint8_rgb(rec[i])
        c = to_uint8_rgb(clean[i])

        err = (rec[i] - clean[i]).abs().mean(dim=0).detach().cpu().numpy()  # H,W
        err = (err / (err.max() + 1e-8) * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(err, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        row = np.concatenate([e, r, c, heat], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)
    bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)


# =========================
# Pair discovery (folder-based)
# =========================

def list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def resolve_clean_base(cfg: AttackConfig) -> Path:
    """
    If clean_root has a pXX subfolder, use it.
    Otherwise, use clean_root directly (your case).
    """
    root = Path(cfg.clean_root)
    candidate = root / f"p{cfg.clean_percent:02d}"
    return candidate if candidate.exists() else root

def load_pairs_from_folders(cfg: AttackConfig) -> List[Tuple[str, str]]:
    """
    Pairs by relative path:
      enc:   enc_root/pXX/<identity>/<img>.png
      clean: clean_root/<identity>/<img>.png
         OR  clean_root/pYY/<identity>/<img>.png (if such folder exists)
    """
    enc_base = Path(cfg.enc_root) / f"p{cfg.percent:02d}"
    clean_base = resolve_clean_base(cfg)

    if not enc_base.exists():
        raise RuntimeError(f"Missing enc folder: {enc_base}")
    if not clean_base.exists():
        raise RuntimeError(f"Missing clean folder: {clean_base}")

    enc_files = list_images(enc_base)
    if len(enc_files) == 0:
        raise RuntimeError(f"No images found under: {enc_base}")

    pairs: List[Tuple[str, str]] = []
    missing = 0

    for enc_path in enc_files:
        rel = enc_path.relative_to(enc_base)
        clean_path = clean_base / rel
        if clean_path.exists():
            pairs.append((str(enc_path), str(clean_path)))
        else:
            missing += 1

    if len(pairs) == 0:
        raise RuntimeError(
            f"No paired files found.\nenc_base={enc_base}\nclean_base={clean_base}\n"
            f"Example enc file: {enc_files[0] if enc_files else 'none'}"
        )

    print(f"[INFO] enc_base   : {enc_base}")
    print(f"[INFO] clean_base : {clean_base}")
    print(f"[INFO] Pairs: {len(pairs)} (missing clean: {missing})")
    return pairs


# =========================
# Dataset
# =========================

class EncCleanPairs(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], img_size: int):
        self.pairs = pairs
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        enc_path, clean_path = self.pairs[idx]
        enc = resize_square(imread_rgb(enc_path), self.img_size)
        clean = resize_square(imread_rgb(clean_path), self.img_size)
        x = to_tensor_01(enc)
        y = to_tensor_01(clean)
        # return enc path so we can recover identity later
        return x, y, enc_path


def identity_from_path(p: str) -> str:
    # identity = parent folder name: .../pXX/<identity>/<img>.png
    return Path(p).parent.name


# =========================
# Models
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=48):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)          # 48
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base*2)         # 96
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base*2, base*4)       # 192
        self.pool3 = nn.MaxPool2d(2)

        self.mid = ConvBlock(base*4, base*4)        # 192

        # Decoder
        self.up3 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)          # 192 -> 96
        self.dec3 = ConvBlock(base*2 + base*4, base*2)                      # (96+192)=288 -> 96

        self.up2 = nn.ConvTranspose2d(base*2, base, 2, stride=2)            # 96 -> 48
        self.dec2 = ConvBlock(base + base*2, base)                          # (48+96)=144 -> 48

        self.up1 = nn.ConvTranspose2d(base, base, 2, stride=2)              # 48 -> 48
        self.dec1 = ConvBlock(base + base, base)                            # (48+48)=96 -> 48

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)                 # B,48,H,W
        e2 = self.enc2(self.pool1(e1))    # B,96,H/2,W/2
        e3 = self.enc3(self.pool2(e2))    # B,192,H/4,W/4
        m  = self.mid(self.pool3(e3))     # B,192,H/8,W/8

        d3 = self.up3(m)                  # B,96,H/4,W/4
        d3 = torch.cat([d3, e3], dim=1)   # B,288,H/4,W/4
        d3 = self.dec3(d3)                # B,96,H/4,W/4

        d2 = self.up2(d3)                 # B,48,H/2,W/2
        d2 = torch.cat([d2, e2], dim=1)   # B,144,H/2,W/2
        d2 = self.dec2(d2)                # B,48,H/2,W/2

        d1 = self.up1(d2)                 # B,48,H,W
        d1 = torch.cat([d1, e1], dim=1)   # B,96,H,W
        d1 = self.dec1(d1)                # B,48,H,W

        y = torch.sigmoid(self.out(d1))   # B,3,H,W
        return y


class DnCNN(nn.Module):
    def __init__(self, in_ch=3, depth=10, width=64):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(width, width, 3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(width, in_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        r = self.net(x)
        y = (x - r).clamp(0, 1)
        return y


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=6, base=64):
        super().__init__()
        def block(cin, cout, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, stride=2, padding=1)]
            if norm:
                layers += [nn.BatchNorm2d(cout)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.net = nn.Sequential(
            *block(in_ch, base, norm=False),
            *block(base, base*2),
            *block(base*2, base*4),
            nn.Conv2d(base*4, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class EmbeddingRegressor(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x):
        f = self.feat(x)
        z = self.head(f)
        z = F.normalize(z, dim=1)
        return z


# =========================
# InsightFace embedding helper
# =========================

class InsightFaceEmbedder:
    def __init__(self, cfg: AttackConfig):
        if not HAVE_INSIGHTFACE:
            raise RuntimeError("insightface is not installed/importable.")
        self.app = FaceAnalysis(name=cfg.insightface_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if cfg.device.startswith("cuda") else -1, det_size=cfg.det_size)

    @torch.no_grad()
    def embed_batch(self, imgs_bchw_01: torch.Tensor) -> torch.Tensor:
        embs = []
        for i in range(imgs_bchw_01.shape[0]):
            rgb = to_uint8_rgb(imgs_bchw_01[i])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = self.app.get(bgr)
            if len(faces) == 0:
                e = np.zeros((512,), dtype=np.float32)
            else:
                faces = sorted(
                    faces,
                    key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                    reverse=True
                )
                e = faces[0].normed_embedding.astype(np.float32)
            embs.append(e)
        embs = torch.from_numpy(np.stack(embs, axis=0))
        embs = F.normalize(embs, dim=1)
        return embs


# =========================
# Verification evaluation (identity-based ROC / AUC / EER)
# =========================

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=1)


def roc_metrics_identity(
    embs: torch.Tensor,
    ids: List[str],
    n_genuine: int,
    n_impostor: int,
    seed: int = 0
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    by_id: Dict[str, List[int]] = {}
    for i, ident in enumerate(ids):
        by_id.setdefault(ident, []).append(i)

    valid_ids = [k for k, v in by_id.items() if len(v) >= 2]
    if len(valid_ids) < 2:
        raise RuntimeError("Need at least 2 identities with >=2 images for verification.")

    pairs = []
    labels = []

    # genuine
    for _ in range(n_genuine):
        ident = rng.choice(valid_ids)
        a, b = rng.choice(by_id[ident], size=2, replace=False)
        pairs.append((a, b))
        labels.append(1)

    # impostor
    for _ in range(n_impostor):
        id1, id2 = rng.choice(valid_ids, size=2, replace=False)
        a = rng.choice(by_id[id1])
        b = rng.choice(by_id[id2])
        pairs.append((a, b))
        labels.append(0)

    pairs = np.array(pairs, dtype=np.int64)
    labels = np.array(labels, dtype=np.int32)

    e1 = embs[pairs[:, 0]]
    e2 = embs[pairs[:, 1]]
    scores = cosine_sim(e1, e2).detach().cpu().numpy()

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0

    return {"auc": float(roc_auc), "eer": float(eer)}


# =========================
# Train / eval loops
# =========================

def split_pairs_by_identity(pairs: List[Tuple[str, str]], val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)

    by_id: Dict[str, List[Tuple[str, str]]] = {}
    for enc, clean in pairs:
        ident = Path(enc).parent.name
        by_id.setdefault(ident, []).append((enc, clean))

    ids = np.array(sorted(by_id.keys()))
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val].tolist())

    tr, va = [], []
    for ident, lst in by_id.items():
        (va if ident in val_ids else tr).extend(lst)

    print(
        f"[INFO] Split identities: train={len(set(Path(x[0]).parent.name for x in tr))}, "
        f"val={len(set(Path(x[0]).parent.name for x in va))}"
    )
    print(f"[INFO] Split images: train={len(tr)}, val={len(va)}")
    return tr, va


def train_regression_model(
    cfg: AttackConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str,
):
    os.makedirs(cfg.outdir, exist_ok=True)
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = 1e9
    best_path = Path(cfg.outdir) / f"{name}_best.pt"

    for ep in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{name}] train ep {ep+1}/{cfg.epochs}")
        for x, y, *_ in pbar:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            pred = model(x)
            loss = F.l1_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        model.eval()
        losses = []
        with torch.no_grad():
            for x, y, *_ in val_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                pred = model(x)
                loss = F.l1_loss(pred, y)
                losses.append(float(loss.detach().cpu()))
        val_loss = float(np.mean(losses)) if losses else float("inf")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)

        print(f"[{name}] ep={ep+1} val_l1={val_loss:.6f} best={best_val:.6f}")

    return str(best_path)


def train_pix2pix(
    cfg: AttackConfig,
    G: nn.Module,
    D: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str = "pix2pix",
):
    os.makedirs(cfg.outdir, exist_ok=True)
    G = G.to(cfg.device)
    D = D.to(cfg.device)

    optG = torch.optim.AdamW(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    optD = torch.optim.AdamW(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    best_path = Path(cfg.outdir) / f"{name}_G_best.pt"
    best_val = 1e9

    for ep in range(cfg.epochs):
        G.train()
        D.train()
        pbar = tqdm(train_loader, desc=f"[{name}] train ep {ep+1}/{cfg.epochs}")
        for x, y, *_ in pbar:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            # --- Train D ---
            with torch.no_grad():
                fake = G(x)
            real_in = torch.cat([x, y], dim=1)
            fake_in = torch.cat([x, fake], dim=1)

            pred_real = D(real_in)
            pred_fake = D(fake_in)

            ones = torch.ones_like(pred_real)
            zeros = torch.zeros_like(pred_fake)

            lossD = (bce(pred_real, ones) + bce(pred_fake, zeros)) * 0.5
            optD.zero_grad(set_to_none=True)
            lossD.backward()
            optD.step()

            # --- Train G ---
            fake = G(x)
            fake_in = torch.cat([x, fake], dim=1)
            pred_fake = D(fake_in)

            loss_g_adv = bce(pred_fake, ones)
            loss_g_l1 = F.l1_loss(fake, y)
            lossG = loss_g_adv + cfg.gan_lambda_l1 * loss_g_l1

            optG.zero_grad(set_to_none=True)
            lossG.backward()
            optG.step()

            pbar.set_postfix(lossD=float(lossD.detach().cpu()), lossG=float(lossG.detach().cpu()))

        # val L1
        G.eval()
        losses = []
        with torch.no_grad():
            for x, y, *_ in val_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                fake = G(x)
                losses.append(float(F.l1_loss(fake, y).detach().cpu()))
        val_l1 = float(np.mean(losses)) if losses else float("inf")

        if val_l1 < best_val:
            best_val = val_l1
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "cfg": cfg.__dict__}, best_path)

        print(f"[{name}] ep={ep+1} val_l1={val_l1:.6f} best={best_val:.6f}")

    return str(best_path)


@torch.no_grad()
def run_full_evaluation(
    cfg: AttackConfig,
    pairs: List[Tuple[str, str]],
    reconstructor: Optional[nn.Module],
    name: str,
    embedder: Optional[InsightFaceEmbedder],
):
    os.makedirs(cfg.outdir, exist_ok=True)
    ds = EncCleanPairs(pairs, cfg.img_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    if reconstructor is not None:
        reconstructor = reconstructor.to(cfg.device)
        reconstructor.eval()

    all_psnr = []
    all_ssim = []

    # embeddings + identity labels
    embs_clean = []
    embs_enc = []
    embs_rec = []
    ids_all: List[str] = []

    saved_examples = False

    for x_enc, y_clean, enc_paths in tqdm(loader, desc=f"[{name}] eval"):
        x_enc = x_enc.to(cfg.device)
        y_clean = y_clean.to(cfg.device)

        if reconstructor is None:
            x_rec = x_enc
        else:
            x_rec = reconstructor(x_enc).clamp(0, 1)

        # identities for this batch
        ids_all.extend([identity_from_path(p) for p in enc_paths])

        # save example grids once
        if cfg.save_examples and not saved_examples:
            examples_dir = Path(cfg.outdir) / "examples" / name
            grid_path = str(examples_dir / f"{name}_grid.png")
            heat_path = str(examples_dir / f"{name}_heatmap.png")
            save_triplet_grid(
                x_enc.detach().cpu(),
                x_rec.detach().cpu(),
                y_clean.detach().cpu(),
                out_path=grid_path,
                max_rows=cfg.examples_per_attack
            )
            save_diff_heatmap(
                x_enc.detach().cpu(),
                x_rec.detach().cpu(),
                y_clean.detach().cpu(),
                out_path=heat_path,
                max_rows=cfg.examples_per_attack
            )
            # Also save individual columns for the combined figure later
            cols_dir = Path(cfg.outdir) / "examples" / "_columns"
            cols_dir.mkdir(parents=True, exist_ok=True)
            b = min(x_enc.shape[0], cfg.examples_per_attack)
            enc_col = np.concatenate([to_uint8_rgb(x_enc[i].cpu()) for i in range(b)], axis=0)
            rec_col = np.concatenate([to_uint8_rgb(x_rec[i].cpu()) for i in range(b)], axis=0)
            clean_col = np.concatenate([to_uint8_rgb(y_clean[i].cpu()) for i in range(b)], axis=0)
            cv2.imwrite(str(cols_dir / "encrypted.png"), cv2.cvtColor(enc_col, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(cols_dir / f"{name}.png"), cv2.cvtColor(rec_col, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(cols_dir / "original.png"), cv2.cvtColor(clean_col, cv2.COLOR_RGB2BGR))
            saved_examples = True

        # visual metrics (rec vs clean)
        all_psnr.append(psnr_torch(x_rec, y_clean).detach().cpu())
        all_ssim.append(ssim_torch(x_rec, y_clean).detach().cpu())

        # embeddings (on CPU for InsightFace)
        if embedder is not None:
            e_clean = embedder.embed_batch(y_clean.detach().cpu())
            e_enc = embedder.embed_batch(x_enc.detach().cpu())
            e_rec = embedder.embed_batch(x_rec.detach().cpu())
            embs_clean.append(e_clean)
            embs_enc.append(e_enc)
            embs_rec.append(e_rec)

    psnr_mean = torch.cat(all_psnr).mean().item()
    ssim_cat = torch.cat(all_ssim)
    ssim_mean = torch.nanmean(ssim_cat).item() if hasattr(torch, "nanmean") else float(np.nanmean(ssim_cat.numpy()))

    results: Dict[str, float] = {
        "name": name,
        "psnr_mean": float(psnr_mean),
        "ssim_mean": float(ssim_mean),
        "percent": float(cfg.percent),
        "clean_percent": float(cfg.clean_percent),
        "n_images_eval": float(len(ids_all)),
    }

    if embedder is not None:
        embs_clean_t = torch.cat(embs_clean, dim=0)
        embs_enc_t = torch.cat(embs_enc, dim=0)
        embs_rec_t = torch.cat(embs_rec, dim=0)

        n_g = cfg.eval_pairs // 2
        n_i = cfg.eval_pairs - n_g

        enc_metrics = roc_metrics_identity(embs_enc_t, ids_all, n_genuine=n_g, n_impostor=n_i, seed=cfg.seed)
        rec_metrics = roc_metrics_identity(embs_rec_t, ids_all, n_genuine=n_g, n_impostor=n_i, seed=cfg.seed + 1)
        clean_metrics = roc_metrics_identity(embs_clean_t, ids_all, n_genuine=n_g, n_impostor=n_i, seed=cfg.seed + 2)

        results.update({
            "AUC_enc_direct": enc_metrics["auc"],
            "EER_enc_direct": enc_metrics["eer"],
            "AUC_rec": rec_metrics["auc"],
            "EER_rec": rec_metrics["eer"],
            "AUC_clean_oracle": clean_metrics["auc"],
            "EER_clean_oracle": clean_metrics["eer"],
        })

    out_json = Path(cfg.outdir) / f"metrics_{name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_json}")
    for k in sorted(results.keys()):
        print(f"{k}: {results[k]}")
    return results


# =========================
# Embedding-space attack training
# =========================

def train_embedding_attack(
    cfg: AttackConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    embedder: InsightFaceEmbedder,
    name: str = "emb_attack",
):
    os.makedirs(cfg.outdir, exist_ok=True)
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_val = 1e9
    best_path = Path(cfg.outdir) / f"{name}_best.pt"

    for ep in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{name}] train ep {ep+1}/{cfg.epochs}")
        for x_enc, y_clean, *_ in pbar:
            x_enc = x_enc.to(cfg.device)
            with torch.no_grad():
                tgt = embedder.embed_batch(y_clean).to(cfg.device)

            pred = model(x_enc)
            loss = 1.0 - cosine_sim(pred, tgt).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        model.eval()
        losses = []
        with torch.no_grad():
            for x_enc, y_clean, *_ in val_loader:
                x_enc = x_enc.to(cfg.device)
                tgt = embedder.embed_batch(y_clean).to(cfg.device)
                pred = model(x_enc)
                loss = 1.0 - cosine_sim(pred, tgt).mean()
                losses.append(float(loss.detach().cpu()))
        val_loss = float(np.mean(losses)) if losses else float("inf")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, best_path)

        print(f"[{name}] ep={ep+1} val_loss={val_loss:.6f} best={best_val:.6f}")

    return str(best_path)


@torch.no_grad()
def eval_embedding_attack(
    cfg: AttackConfig,
    pairs: List[Tuple[str, str]],
    model: nn.Module,
    embedder: InsightFaceEmbedder,
    name: str = "emb_attack",
):
    ds = EncCleanPairs(pairs, cfg.img_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = model.to(cfg.device)
    model.eval()

    embs_pred = []
    embs_clean = []
    embs_enc = []
    ids_all: List[str] = []

    for x_enc, y_clean, enc_paths in tqdm(loader, desc=f"[{name}] eval"):
        x_enc = x_enc.to(cfg.device)
        ids_all.extend([identity_from_path(p) for p in enc_paths])

        pred = model(x_enc).detach().cpu()
        e_clean = embedder.embed_batch(y_clean)
        e_enc = embedder.embed_batch(x_enc.detach().cpu())

        embs_pred.append(pred)
        embs_clean.append(e_clean)
        embs_enc.append(e_enc)

    embs_pred_t = F.normalize(torch.cat(embs_pred, dim=0), dim=1)
    embs_clean_t = torch.cat(embs_clean, dim=0)
    embs_enc_t = torch.cat(embs_enc, dim=0)

    n_g = cfg.eval_pairs // 2
    n_i = cfg.eval_pairs - n_g

    pred_metrics = roc_metrics_identity(embs_pred_t, ids_all, n_genuine=n_g, n_impostor=n_i, seed=cfg.seed + 10)
    enc_metrics = roc_metrics_identity(embs_enc_t, ids_all, n_genuine=n_g, n_impostor=n_i, seed=cfg.seed + 11)
    clean_metrics = roc_metrics_identity(embs_clean_t, ids_all, n_genuine=n_g, n_impostor=n_i, seed=cfg.seed + 12)

    results = {
        "name": name,
        "AUC_pred_emb": pred_metrics["auc"],
        "EER_pred_emb": pred_metrics["eer"],
        "AUC_enc_direct": enc_metrics["auc"],
        "EER_enc_direct": enc_metrics["eer"],
        "AUC_clean_oracle": clean_metrics["auc"],
        "EER_clean_oracle": clean_metrics["eer"],
        "percent": float(cfg.percent),
        "clean_percent": float(cfg.clean_percent),
        "n_images_eval": float(len(ids_all)),
    }

    out_json = Path(cfg.outdir) / f"metrics_{name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_json}")
    for k in sorted(results.keys()):
        print(f"{k}: {results[k]}")
    return results


# =========================
# Main runner
# =========================

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--enc_root", required=True, help="e.g. data/dec_enc_png_merged_0_100")
    ap.add_argument("--clean_root", required=True, help="e.g. data/dec_plain_png_merged_0_100 (or same as enc_root)")
    ap.add_argument("--percent", type=int, required=True, help="encryption percent folder pXX to attack, e.g. 30")
    ap.add_argument("--clean_percent", type=int, default=0, help="clean percent folder, usually 0 (p00)")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=112)
    ap.add_argument("--eval_pairs", type=int, default=2000)
    ap.add_argument("--outdir", default="attack_results")

    ap.add_argument("--mode", required=True, choices=[
        "baseline_only",
        "unet",
        "dncnn",
        "pix2pix",
        "emb_attack",
        "run_all"
    ])

    ap.add_argument("--no_examples", action="store_true")
    ap.add_argument("--examples_per_attack", type=int, default=8)

    args = ap.parse_args()

    cfg = AttackConfig(
        enc_root=args.enc_root,
        clean_root=args.clean_root,
        percent=args.percent,
        clean_percent=args.clean_percent,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        eval_pairs=args.eval_pairs,
        outdir=args.outdir,
        save_examples=(not args.no_examples),
        examples_per_attack=args.examples_per_attack,
    )
    set_seed(cfg.seed)

    pairs = load_pairs_from_folders(cfg)
    train_pairs, val_pairs = split_pairs_by_identity(pairs, val_frac=0.1, seed=cfg.seed)

    train_ds = EncCleanPairs(train_pairs, cfg.img_size)
    val_ds = EncCleanPairs(val_pairs, cfg.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    embedder = None
    if HAVE_INSIGHTFACE:
        try:
            embedder = InsightFaceEmbedder(cfg)
        except Exception as e:
            print(f"[WARN] InsightFace init failed: {e}")
            embedder = None
    else:
        print("[WARN] insightface not available; will skip embedding/ROC evaluation.")

    # Baseline: encrypted directly (no reconstruction)
    if args.mode in ["baseline_only", "run_all"]:
        run_full_evaluation(cfg, val_pairs, reconstructor=None, name="baseline_enc_direct", embedder=embedder)

    # U-Net recon attack
    if args.mode in ["unet", "run_all"]:
        model = UNetSmall()
        best = train_regression_model(cfg, model, train_loader, val_loader, name="unet_recon")
        ckpt = torch.load(best, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        run_full_evaluation(cfg, val_pairs, reconstructor=model, name="unet_recon", embedder=embedder)

    # DnCNN denoise attack
    if args.mode in ["dncnn", "run_all"]:
        model = DnCNN(depth=10)
        best = train_regression_model(cfg, model, train_loader, val_loader, name="dncnn_denoise")
        ckpt = torch.load(best, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        run_full_evaluation(cfg, val_pairs, reconstructor=model, name="dncnn_denoise", embedder=embedder)

    # Pix2Pix attack
    if args.mode in ["pix2pix", "run_all"]:
        G = UNetSmall()
        D = PatchDiscriminator(in_ch=6)
        best = train_pix2pix(cfg, G, D, train_loader, val_loader, name="pix2pix")
        ckpt = torch.load(best, map_location="cpu")
        G.load_state_dict(ckpt["G"])
        run_full_evaluation(cfg, val_pairs, reconstructor=G, name="pix2pix", embedder=embedder)

    # Embedding-space attack
    if args.mode in ["emb_attack", "run_all"]:
        if embedder is None:
            raise RuntimeError("Embedding attack requires insightface working.")
        model = EmbeddingRegressor(emb_dim=512)
        best = train_embedding_attack(cfg, model, train_loader, val_loader, embedder=embedder, name="emb_attack")
        ckpt = torch.load(best, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        eval_embedding_attack(cfg, val_pairs, model, embedder=embedder, name="emb_attack")

    print("\nDone.")

    # ── Build combined comparison figure ──
    # Columns: [Encrypted | UNet | DnCNN | Pix2Pix | Original]
    cols_dir = Path(cfg.outdir) / "examples" / "_columns"
    if cols_dir.exists():
        col_order = [
            ("encrypted",        "Encrypted"),
            ("unet_recon",       "U-Net"),
            ("dncnn_denoise",    "DnCNN"),
            ("pix2pix",          "Pix2Pix"),
            ("original",         "Original"),
        ]
        available_cols = []
        for filename, label in col_order:
            p = cols_dir / f"{filename}.png"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    available_cols.append((img, label))

        if len(available_cols) >= 2:
            print(f"\n[Combined] Building comparison grid with {len(available_cols)} columns ...")

            # All columns should have same height (same number of rows * 112)
            col_imgs = [c[0] for c in available_cols]
            col_labels = [c[1] for c in available_cols]

            # Determine cell dimensions
            cell_h = col_imgs[0].shape[0]  # total height of stacked faces
            cell_w = col_imgs[0].shape[1]  # 112

            label_band = 30
            pad = 4
            canvas_w = len(col_imgs) * (cell_w + pad) + pad
            canvas_h = label_band + cell_h + pad

            canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thick = 2

            for i, (col_img, label) in enumerate(zip(col_imgs, col_labels)):
                x = pad + i * (cell_w + pad)
                y = label_band

                # Paste column
                h_actual = min(col_img.shape[0], canvas_h - y)
                w_actual = min(col_img.shape[1], cell_w)
                canvas[y:y + h_actual, x:x + w_actual] = col_img[:h_actual, :w_actual]

                # Column header label
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
                lx = x + (cell_w - tw) // 2
                ly = th + 6
                cv2.putText(canvas, label, (lx, ly), font, font_scale, (30, 30, 30), font_thick, cv2.LINE_AA)

            combined_path = Path(cfg.outdir) / "combined_nn_recon_comparison.png"
            cv2.imwrite(str(combined_path), canvas)
            print(f"  Saved: {combined_path}")
            print(f"  Size:  {canvas.shape[1]}x{canvas.shape[0]} px")
        else:
            print("[Combined] Not enough columns to build comparison grid.")


if __name__ == "__main__":
    main()