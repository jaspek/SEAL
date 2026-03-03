from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_PERCENTS = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100]


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """JJ2000 PGMs can be 16-bit; normalize to uint8 for ArcFace pipeline."""
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
    """Safe read; returns None on any failure."""
    try:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return img
    except Exception:
        return None


def find_pgm_triplet(base_no_suffix: Path) -> Optional[Tuple[Path, Path, Path]]:
    """
    Component naming variants (PGM):
      A) base_0.pgm base_1.pgm base_2.pgm
      B) base-1.pgm base-2.pgm base-3.pgm
    """
    # Variant A
    c0 = Path(str(base_no_suffix) + "_0.pgm")
    c1 = Path(str(base_no_suffix) + "_1.pgm")
    c2 = Path(str(base_no_suffix) + "_2.pgm")
    if c0.exists() and c1.exists() and c2.exists():
        return (c0, c1, c2)

    # Variant B
    d1 = Path(str(base_no_suffix) + "-1.pgm")
    d2 = Path(str(base_no_suffix) + "-2.pgm")
    d3 = Path(str(base_no_suffix) + "-3.pgm")
    if d1.exists() and d2.exists() and d3.exists():
        return (d1, d2, d3)

    return None


def merge_triplet_to_png(pA: Path, pB: Path, pC: Path, out_png: Path) -> bool:
    """
    Reads 3 grayscale component images and merges them into one 3-channel PNG.
    We interpret them as R,G,B (common JJ2000 component order for color),
    then write BGR for OpenCV.
    """
    a = safe_imread(pA)
    b = safe_imread(pB)
    c = safe_imread(pC)
    if a is None or b is None or c is None:
        return False

    # Must be 2D (grayscale); if not, something weird happened
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        return False

    a = normalize_to_uint8(a)
    b = normalize_to_uint8(b)
    c = normalize_to_uint8(c)

    # Ensure equal sizes (JJ2000 should output identical shapes, but be defensive)
    if a.shape != b.shape or a.shape != c.shape:
        return False

    # Treat as R,G,B -> OpenCV BGR
    bgr = cv2.merge([c, b, a])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_png), bgr))


def discover_bases_from_components(percent_root: Path) -> List[Path]:
    """
    percent_root is a single percent folder like .../p02
    Return base paths WITHOUT suffix that have component pgms present.
    base = .../George_W_Bush_0026  (no -1/-2/-3, no .pgm)
    """
    bases = set()
    for p in percent_root.rglob("*-1.pgm"):
        bases.add(Path(str(p)[:-6]))  # strip "-1.pgm"
    for p in percent_root.rglob("*_0.pgm"):
        bases.add(Path(str(p)[:-6]))  # strip "_0.pgm"
    return sorted(bases)


def merge_tree(src_percent_root: Path, dst_percent_root: Path, overwrite: bool, delete_components: bool,
               max_fail_print: int = 10) -> Tuple[int, int, int]:
    """
    Merge component PGMs from src_percent_root into single PNGs in dst_percent_root.
    Returns (ok, skipped, failed).
    """
    src_percent_root = src_percent_root.resolve()
    dst_percent_root = dst_percent_root.resolve()
    dst_percent_root.mkdir(parents=True, exist_ok=True)

    bases = discover_bases_from_components(src_percent_root)
    ok = skipped = failed = 0
    printed = 0

    for base in bases:
        rel = base.relative_to(src_percent_root)  # identity/filename (within this percent)
        out_png = (dst_percent_root / rel).with_suffix(".png")

        if out_png.exists() and not overwrite:
            skipped += 1
            continue

        trip = find_pgm_triplet(base)
        if trip is None:
            failed += 1
            continue

        pA, pB, pC = trip
        if merge_triplet_to_png(pA, pB, pC, out_png):
            ok += 1
            if delete_components:
                for p in (pA, pB, pC):
                    try:
                        p.unlink()
                    except Exception:
                        pass
        else:
            failed += 1
            if printed < max_fail_print:
                print(f"[FAIL MERGE] {base} (components: {pA.name}, {pB.name}, {pC.name})")
                printed += 1

    return ok, skipped, failed


def merge_all_percents(
    src_root: Path,
    dst_root: Path,
    percents: List[int],
    overwrite: bool,
    delete_components: bool,
) -> Dict[int, Tuple[int, int, int]]:
    """
    Merge src_root/pXX -> dst_root/pXX for all percents that exist.
    Returns dict pct -> (ok, skipped, failed)
    """
    results: Dict[int, Tuple[int, int, int]] = {}
    for pct in percents:
        src_p = src_root / f"p{pct:02d}"
        if not src_p.exists():
            continue
        dst_p = dst_root / f"p{pct:02d}"
        results[pct] = merge_tree(src_p, dst_p, overwrite=overwrite, delete_components=delete_components)
    return results


def list_merged_pngs(percent_root: Path) -> List[Path]:
    """
    List merged PNGs in a percent folder root (e.g. .../p00).
    Returns paths relative to that percent folder: identity/img.png
    """
    out: List[Path] = []
    for p in percent_root.rglob("*.png"):
        out.append(p.relative_to(percent_root))
    out.sort()
    return out


def write_decode_report(
    out_csv: Path,
    merged_enc_root: Path,
    merged_plain_root: Path,
    percent_list: List[int],
) -> None:
    """
    Report based on existence of merged PNGs.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    enc_p00 = merged_enc_root / "p00"
    plain_p00 = merged_plain_root / "p00"

    if not enc_p00.exists():
        raise RuntimeError(f"Missing merged enc p00 folder at: {enc_p00}")
    if not plain_p00.exists():
        print(f"[WARN] Missing merged plain p00 folder at: {plain_p00} (plain_decode_ok will be 0)")

    files_rel = list_merged_pngs(enc_p00)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "percent", "plain_decode_ok", "enc_decode_ok"])

        for rel in files_rel:
            plain_png = plain_p00 / rel
            plain_ok = int(plain_png.exists() and plain_png.stat().st_size > 0)

            for pct in percent_list:
                enc_png = merged_enc_root / f"p{pct:02d}" / rel
                enc_ok = int(enc_png.exists() and enc_png.stat().st_size > 0)
                w.writerow([str(rel).replace("/", "\\"), pct, plain_ok, enc_ok])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_plain_fixed", type=str, required=True, help=r"e.g. data\dec_plain_png_0_100")
    ap.add_argument("--src_enc_fixed", type=str, required=True, help=r"e.g. data\dec_enc_png_0_100")
    ap.add_argument("--dst_plain_merged", type=str, required=True, help=r"e.g. data\dec_plain_png_merged_0_100")
    ap.add_argument("--dst_enc_merged", type=str, required=True, help=r"e.g. data\dec_enc_png_merged_0_100")
    ap.add_argument("--report_out", type=str, required=True, help=r"e.g. results\decode_report_0_100.csv")
    ap.add_argument("--percents", type=str, default=",".join(map(str, DEFAULT_PERCENTS)))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--delete_components", action="store_true")
    args = ap.parse_args()

    percent_list = [int(x.strip()) for x in args.percents.split(",") if x.strip()]

    src_plain = Path(args.src_plain_fixed)
    src_enc = Path(args.src_enc_fixed)
    dst_plain = Path(args.dst_plain_merged)
    dst_enc = Path(args.dst_enc_merged)

    dst_plain.mkdir(parents=True, exist_ok=True)
    dst_enc.mkdir(parents=True, exist_ok=True)

    plain_res = merge_all_percents(src_plain, dst_plain, percent_list, args.overwrite, args.delete_components)
    enc_res = merge_all_percents(src_enc, dst_enc, percent_list, args.overwrite, args.delete_components)

    if plain_res:
        ok = sum(v[0] for v in plain_res.values())
        skipped = sum(v[1] for v in plain_res.values())
        failed = sum(v[2] for v in plain_res.values())
        print(f"[MERGE plain ALL] ok={ok} skipped={skipped} failed={failed} (percents merged: {sorted(plain_res.keys())})")
    else:
        print("[MERGE plain ALL] nothing merged (no src plain pXX folders found)")

    if enc_res:
        ok = sum(v[0] for v in enc_res.values())
        skipped = sum(v[1] for v in enc_res.values())
        failed = sum(v[2] for v in enc_res.values())
        print(f"[MERGE enc   ALL] ok={ok} skipped={skipped} failed={failed} (percents merged: {sorted(enc_res.keys())})")
    else:
        raise RuntimeError("No enc percent folders found to merge. Check --src_enc_fixed path.")

    write_decode_report(
        out_csv=Path(args.report_out),
        merged_enc_root=dst_enc,
        merged_plain_root=dst_plain,
        percent_list=percent_list,
    )
    print(f"[REPORT] wrote: {args.report_out}")


if __name__ == "__main__":
    main()