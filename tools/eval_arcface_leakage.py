import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# (optional but often helps)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from insightface.model_zoo import get_model


import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple


import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import onnxruntime as ort

from sklearn.metrics import roc_curve


# -------------------------
# Helpers
# -------------------------

def list_images_by_id(root: Path, identity_level: int = 1) -> Dict[str, List[Path]]:
    """
    Groups images by identity using folder name at a chosen depth.
    identity_level=1 means parent folder name (root/.../<id>/<img>)
    identity_level=2 means grandparent folder name (root/.../<id>/<something>/<img>)
    """
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    id_to_imgs: Dict[str, List[Path]] = {}

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    print(f"[DEBUG] Found {len(files)} image files under: {root}")

    for p in files:
        parts = p.relative_to(root).parts
        # identity folder should exist
        if len(parts) <= identity_level:
            # fallback: use parent name
            identity = p.parent.name
        else:
            identity = parts[-(identity_level+1)]
        id_to_imgs.setdefault(identity, []).append(p)

    # sort + print stats
    for k in id_to_imgs:
        id_to_imgs[k] = sorted(id_to_imgs[k])

    counts = sorted([len(v) for v in id_to_imgs.values()], reverse=True)
    print(f"[DEBUG] Unique identities: {len(id_to_imgs)}")
    if counts:
        print(f"[DEBUG] Max imgs/identity: {counts[0]}, median: {counts[len(counts)//2]}, min: {counts[-1]}")
    else:
        print("[DEBUG] No identities built (no images found).")

    return id_to_imgs


def split_identities(id_to_imgs: Dict[str, List[Path]], test_frac: float = 0.3, seed: int = 42):
    rng = random.Random(seed)
    ids = [i for i, imgs in id_to_imgs.items() if len(imgs) >= 2]
    print(f"[DEBUG] Identities with >=2 images: {len(ids)}")
    if len(ids) < 2:
        raise RuntimeError(
            "Not enough identities with >=2 images. "
            "Check that p00 contains multiple images per identity."
        )
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_frac))
    test_ids = sorted(ids[:n_test])
    train_ids = sorted(ids[n_test:])
    if len(test_ids) == 0:
        test_ids = ids[:1]
    if len(train_ids) == 0:
        train_ids = ids[1:]
    return train_ids, test_ids


def make_verification_pairs(
    id_to_imgs: Dict[str, List[Path]],
    ids: List[str],
    n_genuine: int = 3000,
    n_impostor: int = 3000,
    seed: int = 123
) -> List[Tuple[Path, Path, int]]:
    """
    Create pairs (img1, img2, label) where label=1 genuine, 0 impostor.
    """
    rng = random.Random(seed)

    # Genuine pairs
    genuine: List[Tuple[Path, Path, int]] = []
    for _ in range(n_genuine):
        person = rng.choice(ids)
        imgs = id_to_imgs[person]
        a, b = rng.sample(imgs, 2)
        genuine.append((a, b, 1))

    # Impostor pairs
    impostor: List[Tuple[Path, Path, int]] = []
    for _ in range(n_impostor):
        p1, p2 = rng.sample(ids, 2)
        a = rng.choice(id_to_imgs[p1])
        b = rng.choice(id_to_imgs[p2])
        impostor.append((a, b, 0))

    pairs = genuine + impostor
    rng.shuffle(pairs)
    return pairs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    EER = point where FAR ~= FRR.
    Using ROC curve: fpr vs tpr. FRR = 1 - tpr.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


def tar_at_far(y_true: np.ndarray, y_score: np.ndarray, far_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # take best TPR under FAR constraint
    ok = np.where(fpr <= far_target)[0]
    if len(ok) == 0:
        return 0.0
    return float(np.max(tpr[ok]))


# -------------------------
# ArcFace embedding
# -------------------------

@dataclass
class EmbedderConfig:
    det_size: Tuple[int, int] = (640, 640)   # unused now, keep for compatibility
    ctx_id: int = 0                          # GPU id; -1 for CPU
    enforce_single_face: bool = True         # unused now


class ArcFaceEmbedder:
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg

        avail = ort.get_available_providers()
        print(f"[DEBUG] onnxruntime available providers: {avail}")

        if "CUDAExecutionProvider" in avail and cfg.ctx_id >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Directly load the ArcFace recognition model (NO DETECTOR)
        model_path = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ArcFace model not found at: {model_path}")

        self.model = get_model(str(model_path), providers=providers)
        self.model.prepare(ctx_id=cfg.ctx_id if "CUDAExecutionProvider" in providers else -1)

    def embed_112(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None:
            raise RuntimeError("Empty image")

        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        if img_bgr.shape[0] != 112 or img_bgr.shape[1] != 112:
            img_bgr = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        emb = self.model.get_feat(img_rgb).astype(np.float32).reshape(-1)
        return emb


def evaluate_folder(root: Path, pairs_rel: List[Tuple[Path, Path, int]], embedder: ArcFaceEmbedder) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    root: folder like data/dec_enc_png/p12
    pairs_rel: pairs using relative paths (identity/img.png) based on p00 root structure
    Returns y_true, y_score, n_failed_pairs
    """
    cache: Dict[str, np.ndarray] = {}
    y_true = []
    y_score = []
    failed = 0

    for a_rel, b_rel, lab in tqdm(pairs_rel, desc=f"Eval {root.name}"):
        try:
            if a_rel not in cache:
                a = cv2.imread(str(root / a_rel))
                if a is None:
                    raise RuntimeError("imread failed")
                cache[a_rel] = embedder.embed_112(a)
            if b_rel not in cache:
                b = cv2.imread(str(root / b_rel))
                if b is None:
                    raise RuntimeError("imread failed")
                cache[b_rel] = embedder.embed_112(b)

            s = cosine(cache[a_rel], cache[b_rel])
            y_true.append(lab)
            y_score.append(s)
        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"[FAIL] {root.name} a={a_rel} b={b_rel} err={repr(e)}")

    return np.array(y_true, dtype=np.int32), np.array(y_score, dtype=np.float32), failed


def main():
    base = Path(__file__).resolve().parents[1]

    # merged folders produced by your merge script
    MERGED_ROOT = base / "data" / "dec_enc_png_merged_0_100"
    p00 = MERGED_ROOT / "p00"

    percents = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100]
    roots = {p: MERGED_ROOT / f"p{p:02d}" for p in percents}

    # Build identity list from p00
    id_to_imgs = list_images_by_id(p00, identity_level=1)
    train_ids, test_ids = split_identities(id_to_imgs, test_frac=0.3, seed=42)

    # Create evaluation pairs on TEST identities only
    pairs = make_verification_pairs(id_to_imgs, test_ids, n_genuine=3000, n_impostor=3000, seed=123)

    # Convert to relative paths so we can reuse across pXX folders
    pairs_rel: List[Tuple[Path, Path, int]] = []
    for a, b, lab in pairs:
        a_rel = a.relative_to(p00)
        b_rel = b.relative_to(p00)
        pairs_rel.append((a_rel, b_rel, lab))

    # Embedder
    embedder = ArcFaceEmbedder(EmbedderConfig(det_size=(640, 640), ctx_id=0, enforce_single_face=True))

    # quick sanity: find first readable image in p00
    sample_path = None
    for _, imgs in id_to_imgs.items():
        for img_path in imgs:
            sample_path = img_path
            break
        if sample_path:
            break

    test_img = cv2.imread(str(sample_path))
    print("[DEBUG] sample file:", sample_path)
    print("[DEBUG] sample shape:", None if test_img is None else test_img.shape)

    if test_img is None:
        raise RuntimeError("cv2.imread failed on the first sample. Your merged PNGs may be corrupted.")

    emb = embedder.embed_112(test_img)
    print("[DEBUG] emb shape:", emb.shape, "norm:", float(np.linalg.norm(emb)))

    rows = []
    for p, root in roots.items():
        if not root.exists():
            print(f"Missing folder: {root}")
            continue

        y_true, y_score, failed = evaluate_folder(root, pairs_rel, embedder)

        if len(y_true) < 1000:
            print(f"Too few successful pairs for p={p}: {len(y_true)} (failed {failed})")
            continue

        eer = compute_eer(y_true, y_score)
        tar_1 = tar_at_far(y_true, y_score, far_target=0.01)
        tar_01 = tar_at_far(y_true, y_score, far_target=0.001)

        rows.append({
            "percent": p,
            "n_pairs": int(len(y_true)),
            "failed_pairs": int(failed),
            "eer": eer,
            "tar@far=1%": tar_1,
            "tar@far=0.1%": tar_01,
        })

    df = pd.DataFrame(rows).sort_values("percent")
    out_csv = base / "results" / "arcface_leakage_0_100.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()