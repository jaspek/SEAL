from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# =========================
# EDIT THESE TWO PATHS
# =========================
SRC_ROOT = Path(r"C:\Users\jasko\Programming\Python\Program_3\data\dec_plain_png_0_100")         # contains Name/.../*_0.pgm, *_1.pgm, *_2.pgm
DST_ROOT = Path(r"C:\Users\jasko\Programming\Python\Program_3\data\dec_plain_png_merged_0_100")  # will create Name/.../*.png


# =========================
# SETTINGS
# =========================
OVERWRITE = False
DELETE_COMPONENTS = False
MAX_FAIL_PRINT = 20


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    mn = float(img_f.min())
    mx = float(img_f.max())
    if mx <= mn:
        return np.zeros_like(img_f, dtype=np.uint8)
    img_f = (img_f - mn) * (255.0 / (mx - mn))
    return img_f.clip(0, 255).astype(np.uint8)


def safe_imread(p: Path) -> Optional[np.ndarray]:
    try:
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def find_pgm_triplet(base_no_suffix: Path) -> Optional[Tuple[Path, Path, Path]]:
    # Variant A: base_0.pgm base_1.pgm base_2.pgm
    c0 = Path(str(base_no_suffix) + "_0.pgm")
    c1 = Path(str(base_no_suffix) + "_1.pgm")
    c2 = Path(str(base_no_suffix) + "_2.pgm")
    if c0.exists() and c1.exists() and c2.exists():
        return (c0, c1, c2)

    # Variant B: base-1.pgm base-2.pgm base-3.pgm
    d1 = Path(str(base_no_suffix) + "-1.pgm")
    d2 = Path(str(base_no_suffix) + "-2.pgm")
    d3 = Path(str(base_no_suffix) + "-3.pgm")
    if d1.exists() and d2.exists() and d3.exists():
        return (d1, d2, d3)

    return None


def merge_triplet_to_png(pA: Path, pB: Path, pC: Path, out_png: Path) -> bool:
    a = safe_imread(pA)
    b = safe_imread(pB)
    c = safe_imread(pC)
    if a is None or b is None or c is None:
        return False

    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        return False

    a = normalize_to_uint8(a)
    b = normalize_to_uint8(b)
    c = normalize_to_uint8(c)

    if a.shape != b.shape or a.shape != c.shape:
        return False

    # Interpret components as R,G,B -> OpenCV BGR
    bgr = cv2.merge([c, b, a])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_png), bgr))


def discover_bases_from_components(src_root: Path) -> List[Path]:
    bases = set()
    for p in src_root.rglob("*_0.pgm"):
        bases.add(Path(str(p)[:-6]))  # strip "_0.pgm"
    for p in src_root.rglob("*-1.pgm"):
        bases.add(Path(str(p)[:-6]))  # strip "-1.pgm"
    return sorted(bases)


def merge_tree(src_root: Path, dst_root: Path) -> Tuple[int, int, int]:
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    bases = discover_bases_from_components(src_root)

    ok = skipped = failed = 0
    printed = 0

    for base in bases:
        rel = base.relative_to(src_root)                 # e.g. Name/imgbase
        out_png = (dst_root / rel).with_suffix(".png")   # e.g. dst/Name/imgbase.png

        if out_png.exists() and not OVERWRITE:
            skipped += 1
            continue

        trip = find_pgm_triplet(base)
        if trip is None:
            failed += 1
            continue

        pA, pB, pC = trip
        if merge_triplet_to_png(pA, pB, pC, out_png):
            ok += 1
            if DELETE_COMPONENTS:
                for p in (pA, pB, pC):
                    try:
                        p.unlink()
                    except Exception:
                        pass
        else:
            failed += 1
            if printed < MAX_FAIL_PRINT:
                print(f"[FAIL MERGE] {base} (components: {pA.name}, {pB.name}, {pC.name})")
                printed += 1

    return ok, skipped, failed


def main() -> None:
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT does not exist: {SRC_ROOT.resolve()}")

    ok, skipped, failed = merge_tree(SRC_ROOT, DST_ROOT)

    print("Done.")
    print(f"  src: {SRC_ROOT.resolve()}")
    print(f"  dst: {DST_ROOT.resolve()}")
    print(f"  ok={ok} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()