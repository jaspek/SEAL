from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


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


def jj2000_outputs_for_base(base_no_suffix: Path) -> Tuple[Optional[Path], Optional[Tuple[Path, Path, Path]]]:
    """
    base_no_suffix is path without suffix, e.g. .../p12/A/B (no .pgm/.png).
    Returns either single.pgm or (c0,c1,c2) triplet.
    """
    single = base_no_suffix.with_suffix(".pgm")
    c0 = Path(str(base_no_suffix) + "_0.pgm")
    c1 = Path(str(base_no_suffix) + "_1.pgm")
    c2 = Path(str(base_no_suffix) + "_2.pgm")

    if c0.exists() and c1.exists() and c2.exists():
        return None, (c0, c1, c2)
    if single.exists():
        return single, None
    return None, None


def merge_triplet_to_png(p0: Path, p1: Path, p2: Path, out_png: Path) -> bool:
    c0 = cv2.imread(str(p0), cv2.IMREAD_UNCHANGED)
    c1 = cv2.imread(str(p1), cv2.IMREAD_UNCHANGED)
    c2 = cv2.imread(str(p2), cv2.IMREAD_UNCHANGED)
    if c0 is None or c1 is None or c2 is None:
        return False

    c0 = normalize_to_uint8(c0)
    c1 = normalize_to_uint8(c1)
    c2 = normalize_to_uint8(c2)

    # components usually R,G,B -> convert to BGR for OpenCV
    bgr = cv2.merge([c2, c1, c0])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_png), bgr))


def single_pgm_to_png(pgm: Path, out_png: Path) -> bool:
    img = cv2.imread(str(pgm), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    img = normalize_to_uint8(img)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_png), img))


def convert_tree(src_root: Path, dst_root: Path, delete_pgms: bool, overwrite: bool) -> None:
    """
    Mirror src_root into dst_root, but write .png outputs there.
    This assumes JJ2000 created .pgm / _0/_1/_2.pgm under src_root.
    """
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    # We detect bases by scanning for *_0.pgm (triplets) and *.pgm (singles).
    # Triplets first:
    bases = set()

    for p0 in src_root.rglob("*_0.pgm"):
        base = Path(str(p0)[:-6])  # removes "_0.pgm"
        p1 = Path(str(base) + "_1.pgm")
        p2 = Path(str(base) + "_2.pgm")
        if p1.exists() and p2.exists():
            bases.add(base)

    # Singles:
    for pgm in src_root.rglob("*.pgm"):
        if str(pgm).endswith("_0.pgm") or str(pgm).endswith("_1.pgm") or str(pgm).endswith("_2.pgm"):
            continue
        bases.add(pgm.with_suffix(""))  # base without suffix

    ok = skipped = failed = 0

    for base in sorted(bases):
        rel = base.relative_to(src_root)  # relative path inside src_root
        out_png = (dst_root / rel).with_suffix(".png")

        if out_png.exists() and not overwrite:
            skipped += 1
            continue

        single, trip = jj2000_outputs_for_base(base)
        success = False

        if trip is not None:
            p0, p1, p2 = trip
            success = merge_triplet_to_png(p0, p1, p2, out_png)
            if success and delete_pgms:
                for p in (p0, p1, p2):
                    try: p.unlink()
                    except Exception: pass

        elif single is not None:
            success = single_pgm_to_png(single, out_png)
            if success and delete_pgms:
                try: single.unlink()
                except Exception: pass

        if success:
            ok += 1
        else:
            failed += 1

    print(f"[DONE] {src_root} -> {dst_root} | ok={ok} skipped={skipped} failed={failed}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_plain", type=str, required=True, help="e.g. data/dec_plain_png")
    ap.add_argument("--src_enc", type=str, required=True, help="e.g. data/dec_enc_png")
    ap.add_argument("--dst_plain", type=str, required=True, help="e.g. data/dec_plain_png_fixed")
    ap.add_argument("--dst_enc", type=str, required=True, help="e.g. data/dec_enc_png_fixed")
    ap.add_argument("--delete_pgms", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    convert_tree(Path(args.src_plain), Path(args.dst_plain), delete_pgms=args.delete_pgms, overwrite=args.overwrite)
    convert_tree(Path(args.src_enc), Path(args.dst_enc), delete_pgms=args.delete_pgms, overwrite=args.overwrite)


if __name__ == "__main__":
    main()