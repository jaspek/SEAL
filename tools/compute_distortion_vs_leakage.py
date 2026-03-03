from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim


DEFAULT_PERCENTS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100]


def list_rel_pngs(p00_dir: Path) -> List[Path]:
    rels: List[Path] = []
    for p in p00_dir.rglob("*.png"):
        rels.append(p.relative_to(p00_dir))
    rels.sort()
    return rels


def safe_read_bgr(p: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return img


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    # expects uint8 grayscale (H,W)
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    mse = float(np.mean((a_f - b_f) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def compute_metrics_for_pair(ref_bgr: np.ndarray, test_bgr: np.ndarray) -> Tuple[float, float]:
    # Both images are 112x112 in your pipeline; still be safe:
    if ref_bgr.shape != test_bgr.shape:
        test_bgr = cv2.resize(test_bgr, (ref_bgr.shape[1], ref_bgr.shape[0]), interpolation=cv2.INTER_AREA)

    ref_g = to_gray(ref_bgr)
    test_g = to_gray(test_bgr)

    p = psnr_u8(ref_g, test_g)
    s = float(ssim(ref_g, test_g, data_range=255))
    return p, s


def main():
    base = Path(__file__).resolve().parents[1]

    # Adjust these if your merged folders have different names:
    merged_root = base / "data" / "dec_enc_png_merged_0_100"
    p00_dir = merged_root / "p00"

    arcface_csv = base / "results" / "arcface_leakage_0_100.csv"
    out_csv = base / "results" / "distortion_vs_leakage_0_100.csv"

    if not p00_dir.exists():
        raise RuntimeError(f"Missing p00 folder: {p00_dir}")

    if not arcface_csv.exists():
        raise RuntimeError(f"Missing ArcFace results CSV: {arcface_csv}")

    rels = list_rel_pngs(p00_dir)
    print(f"[INFO] Reference images in p00: {len(rels)}")

    rows = []
    for pct in DEFAULT_PERCENTS:
        p_dir = merged_root / f"p{pct:02d}"
        if not p_dir.exists():
            print(f"[WARN] Missing: {p_dir} (skipping)")
            continue

        psnrs: List[float] = []
        ssims: List[float] = []
        missing = 0
        bad = 0

        for rel in tqdm(rels, desc=f"metrics p{pct:02d}"):
            ref_path = p00_dir / rel
            test_path = p_dir / rel

            if not test_path.exists():
                missing += 1
                continue

            ref = safe_read_bgr(ref_path)
            test = safe_read_bgr(test_path)
            if ref is None or test is None:
                bad += 1
                continue

            p_val, s_val = compute_metrics_for_pair(ref, test)
            psnrs.append(p_val)
            ssims.append(s_val)

        if len(psnrs) == 0:
            print(f"[WARN] No valid pairs for p{pct:02d}")
            continue

        rows.append({
            "percent": pct,
            "n_ref": len(rels),
            "n_used": len(psnrs),
            "n_missing": missing,
            "n_bad": bad,
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_median": float(np.median(psnrs)),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_median": float(np.median(ssims)),
        })

    df_dist = pd.DataFrame(rows).sort_values("percent")

    df_arc = pd.read_csv(arcface_csv).sort_values("percent")

    # Merge into one “thesis table”
    df = pd.merge(df_arc, df_dist, on="percent", how="inner").sort_values("percent")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\n[INFO] Distortion table:")
    print(df_dist)

    print("\n[INFO] Combined leakage + distortion table:")
    print(df)

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()