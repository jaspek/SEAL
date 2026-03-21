"""
j2k_selective_encrypt_demo.py

Encrypts a single image at multiple encryption levels and produces:
  1) Individual decoded PNGs for each level
  2) A combined grid image (like the thesis figure) with labels

Handles JJ2000's multi-component PGM output (e.g. lena_0.pgm, lena_1.pgm, lena_2.pgm)
by merging them into a single RGB PNG.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURATION — EDIT HERE                     ║
# ╚══════════════════════════════════════════════════════════════════╝

INPUT_PNG    = r"C:\Users\jasko\Programming\Python\Program_3\tools\lena_std.png"
OUTPUT_DIR   = r""   # leave empty = "encrypted_output" next to input

# All encryption levels to produce
PERCENT_LIST = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100]

# Grid layout for the combined image (ncols x nrows should >= len(PERCENT_LIST))
GRID_COLS = 7

# Tool paths
BASE_DIR         = Path(__file__).resolve().parents[1]
JJ2000_CLASSPATH = BASE_DIR / "tools" / "build"
JAVA_EXE         = "java"
OPJ_COMPRESS     = "opj_compress"

# ╔══════════════════════════════════════════════════════════════════╗
# ║                 NOTHING TO EDIT BELOW THIS LINE                  ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── JPEG2000 markers ──
SOP = b"\xff\x91"
EPH = b"\xff\x92"


# ═══════════════════════════════════════════════════════════════════
# PGM merging (handles JJ2000 multi-component output)
# ═══════════════════════════════════════════════════════════════════

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """JJ2000 PGMs can be 16-bit; normalize to uint8."""
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    mn, mx = float(img_f.min()), float(img_f.max())
    if mx <= mn:
        return np.zeros_like(img_f, dtype=np.uint8)
    img_f = (img_f - mn) * (255.0 / (mx - mn))
    return img_f.clip(0, 255).astype(np.uint8)


def find_jj2000_outputs(base_no_suffix: Path):
    """
    JJ2000 can produce:
      A) Component files: base_0.pgm, base_1.pgm, base_2.pgm  (color)
      B) Component files: base-1.pgm, base-2.pgm, base-3.pgm  (color)
      C) A single file:   base.pgm                              (grayscale)

    Returns: ("triplet", (p0, p1, p2)) or ("single", path) or ("none", None)
    """
    # Variant A: _0, _1, _2
    c0 = Path(str(base_no_suffix) + "_0.pgm")
    c1 = Path(str(base_no_suffix) + "_1.pgm")
    c2 = Path(str(base_no_suffix) + "_2.pgm")
    if c0.exists() and c1.exists() and c2.exists():
        return "triplet", (c0, c1, c2)

    # Variant B: -1, -2, -3
    d1 = Path(str(base_no_suffix) + "-1.pgm")
    d2 = Path(str(base_no_suffix) + "-2.pgm")
    d3 = Path(str(base_no_suffix) + "-3.pgm")
    if d1.exists() and d2.exists() and d3.exists():
        return "triplet", (d1, d2, d3)

    # Single PGM
    single = base_no_suffix.with_suffix(".pgm")
    if single.exists() and single.stat().st_size > 0:
        return "single", single

    return "none", None


def merge_pgm_to_png(base_no_suffix: Path, out_png: Path, delete_pgms: bool = True) -> bool:
    """
    Find JJ2000 output PGMs for a given base path (without .pgm suffix),
    merge components into a single PNG, and optionally clean up PGMs.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    kind, data = find_jj2000_outputs(base_no_suffix)

    if kind == "triplet":
        p0, p1, p2 = data
        c0 = cv2.imread(str(p0), cv2.IMREAD_UNCHANGED)
        c1 = cv2.imread(str(p1), cv2.IMREAD_UNCHANGED)
        c2 = cv2.imread(str(p2), cv2.IMREAD_UNCHANGED)

        if c0 is None or c1 is None or c2 is None:
            print(f"  [WARN] Failed to read component PGMs for {base_no_suffix}")
            return False

        c0 = normalize_to_uint8(c0)
        c1 = normalize_to_uint8(c1)
        c2 = normalize_to_uint8(c2)

        if c0.shape != c1.shape or c0.shape != c2.shape:
            print(f"  [WARN] Component shapes don't match for {base_no_suffix}")
            return False

        # JJ2000 components are R, G, B -> OpenCV needs BGR
        bgr = cv2.merge([c2, c1, c0])
        ok = cv2.imwrite(str(out_png), bgr)

        if delete_pgms:
            for p in (p0, p1, p2):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        return bool(ok)

    elif kind == "single":
        pgm_path = data
        img = cv2.imread(str(pgm_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
        img = normalize_to_uint8(img)
        ok = cv2.imwrite(str(out_png), img)

        if delete_pgms:
            try:
                pgm_path.unlink(missing_ok=True)
            except Exception:
                pass
        return bool(ok)

    else:
        # List what files DO exist for debugging
        parent = base_no_suffix.parent
        stem = base_no_suffix.name
        found = list(parent.glob(f"{stem}*"))
        print(f"  [WARN] No PGM output found for {base_no_suffix}")
        if found:
            print(f"         Found files: {[f.name for f in found]}")
        return False


# ═══════════════════════════════════════════════════════════════════
# JPEG2000 encode / encrypt / decode
# ═══════════════════════════════════════════════════════════════════

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def find_packet_body_ranges(data: bytes) -> list[tuple[int, int]]:
    ranges = []
    i = 0
    n = len(data)
    while True:
        sop = data.find(SOP, i)
        if sop == -1:
            break
        eph = data.find(EPH, sop + 2)
        if eph == -1:
            break
        body_start = eph + 2
        next_sop = data.find(SOP, body_start)
        if next_sop == -1:
            ranges.append((body_start, n))
            break
        else:
            if next_sop > body_start:
                ranges.append((body_start, next_sop))
            i = next_sop
    return ranges


def aes_ctr_encrypt_ranges(
    buf: bytes,
    ranges: list[tuple[int, int]],
    enc_percent: float,
    key: bytes,
    nonce: bytes,
) -> tuple[bytes, int]:
    total_body = sum((b - a) for a, b in ranges)
    target = int(total_body * (enc_percent / 100.0))
    if enc_percent > 0 and target < 16:
        target = min(16, total_body)
    if target <= 0:
        return buf, 0

    out = bytearray(buf)
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    done = 0
    for a, b in ranges:
        if done >= target:
            break
        take = min(b - a, target - done)
        out[a : a + take] = cipher.encrypt(bytes(out[a : a + take]))
        done += take
    return bytes(out), done


@dataclass
class J2KSelectiveEncryptConfig:
    enc_percent: float
    key: bytes
    nonce: bytes


def encode_lossless_j2k(opj_compress: str, in_png: Path, out_j2k: Path) -> None:
    out_j2k.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        opj_compress,
        "-i", str(in_png),
        "-o", str(out_j2k),
        "-r", "1",
        "-n", "1",
        "-SOP",
        "-EPH",
    ]
    run(cmd)


def decode_j2k_jj2000(
    java_exe: str,
    jj2000_classpath: Path,
    in_j2k: Path,
    out_png: Path,
    verbose: bool = True,
) -> bool:
    """
    Decode J2K -> PGM(s) via JJ2000, then merge components into one PNG.
    Handles both single-PGM and multi-component-PGM output.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # JJ2000 writes to the -o path but may produce component files
    # e.g. for "out.pgm" it may create out_0.pgm, out_1.pgm, out_2.pgm
    tmp_pgm = out_png.with_suffix(".pgm")

    cmd = [
        java_exe,
        "-Xmx2g",
        "-cp", str(jj2000_classpath),
        "JJ2KDecoder",
        "-i", str(in_j2k),
        "-o", str(tmp_pgm),
    ]

    # Run JJ2000 — capture output for diagnostics instead of suppressing it
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if verbose:
            print(f"\n  [JJ2000 ERROR] exit code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            if result.stdout:
                print(f"  stdout: {result.stdout[:500]}")
        return False

    # Check what files JJ2000 actually created
    parent_dir = tmp_pgm.parent
    stem = tmp_pgm.stem  # e.g. "lena_std_dec_plain"
    created_pgms = sorted(parent_dir.glob(f"{stem}*.pgm"))

    if verbose and not created_pgms:
        # Nothing was created — look for ANY new pgm files as diagnostic
        all_pgms = sorted(parent_dir.glob("*.pgm"))
        print(f"\n  [DIAG] No PGMs matching '{stem}*.pgm' in {parent_dir}")
        if all_pgms:
            print(f"  [DIAG] But found these PGMs: {[p.name for p in all_pgms[:10]]}")
        else:
            print(f"  [DIAG] No PGM files at all in {parent_dir}")
            print(f"  [DIAG] JJ2000 stdout: {result.stdout[:300]}")

    # Merge PGM components into a single PNG
    base_no_suffix = tmp_pgm.with_suffix("")  # strip .pgm
    ok = merge_pgm_to_png(base_no_suffix, out_png, delete_pgms=True)

    if not ok and verbose:
        # Last resort: maybe JJ2000 used a different naming scheme
        # Try to find component files by globbing
        all_pgms = sorted(parent_dir.glob(f"{stem}*"))
        if all_pgms:
            print(f"  [DIAG] Files matching '{stem}*': {[p.name for p in all_pgms[:10]]}")

    return ok


def selective_encrypt_j2k(
    in_j2k: Path, out_j2k: Path, cfg: J2KSelectiveEncryptConfig
) -> Tuple[int, int]:
    out_j2k.parent.mkdir(parents=True, exist_ok=True)
    data = in_j2k.read_bytes()

    ranges = find_packet_body_ranges(data)
    if not ranges:
        raise RuntimeError("No SOP/EPH packet body ranges found. Did you encode with -SOP -EPH?")

    enc_data, enc_len = aes_ctr_encrypt_ranges(
        data, ranges, cfg.enc_percent, cfg.key, cfg.nonce
    )
    out_j2k.write_bytes(enc_data)
    return ranges[0][0], enc_len


# ═══════════════════════════════════════════════════════════════════
# Grid image creation
# ═══════════════════════════════════════════════════════════════════

def make_grid_image(
    image_paths: List[Path],
    labels: List[str],
    ncols: int = 7,
    label_font_scale: float = 0.65,
    label_thickness: int = 2,
    padding: int = 8,
    label_height: int = 30,
    title: str = "",
    caption: str = "",
    title_font_scale: float = 0.75,
    caption_font_scale: float = 0.55,
) -> np.ndarray:
    """
    Create a grid image from a list of image paths with labels underneath.
    Suitable for direct inclusion in a LaTeX document.

    Parameters
    ----------
    label_font_scale : float
        Font size for the "p = X%" labels under each image.
        Default 0.65. Use 0.8-1.0 for larger text, 0.4-0.5 for smaller.
    label_thickness : int
        Thickness of label text. 1 = thin, 2 = bold.
    title : str
        Optional title drawn above the grid.
    caption : str
        Optional caption/description drawn below the grid.
    """
    assert len(image_paths) == len(labels)

    # Load all images
    imgs = []
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read: {p}")
        imgs.append(img)

    h, w = imgs[0].shape[:2]
    n = len(imgs)
    nrows = (n + ncols - 1) // ncols

    font = cv2.FONT_HERSHEY_SIMPLEX

    cell_w = w + padding
    cell_h = h + label_height + padding

    canvas_w = ncols * cell_w + padding

    # Calculate space for title and caption
    title_h = 0
    if title:
        (tw, th), _ = cv2.getTextSize(title, font, title_font_scale, 2)
        title_h = th + 20  # text height + top/bottom margin

    caption_h = 0
    caption_lines = []
    if caption:
        # Word-wrap caption to fit canvas width
        max_chars = int(canvas_w / (caption_font_scale * 12))  # rough estimate
        words = caption.split()
        line = ""
        for word in words:
            test = f"{line} {word}".strip()
            if len(test) > max_chars and line:
                caption_lines.append(line)
                line = word
            else:
                line = test
        if line:
            caption_lines.append(line)
        (_, line_th), _ = cv2.getTextSize("Ag", font, caption_font_scale, 1)
        caption_h = len(caption_lines) * (line_th + 8) + 16

    canvas_h = title_h + nrows * cell_h + padding + caption_h

    # White canvas
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    # Draw title
    if title:
        (tw, th), _ = cv2.getTextSize(title, font, title_font_scale, 2)
        tx = (canvas_w - tw) // 2
        ty = th + 8
        cv2.putText(canvas, title, (tx, ty), font, title_font_scale, (30, 30, 30), 2, cv2.LINE_AA)

    # Draw image grid
    grid_y_offset = title_h

    for idx, (img, label) in enumerate(zip(imgs, labels)):
        row = idx // ncols
        col = idx % ncols

        x = padding + col * cell_w
        y = grid_y_offset + padding + row * cell_h

        # Resize if needed
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # Convert grayscale to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Paste image
        canvas[y : y + h, x : x + w] = img

        # Thin border
        cv2.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), (180, 180, 180), 1)

        # Label centered below image
        (tw, th_text), _ = cv2.getTextSize(label, font, label_font_scale, label_thickness)
        lx = x + (w - tw) // 2
        ly = y + h + th_text + 5
        cv2.putText(
            canvas, label, (lx, ly), font,
            label_font_scale, (40, 40, 40), label_thickness, cv2.LINE_AA,
        )

    # Draw caption below grid
    if caption_lines:
        (_, line_th), _ = cv2.getTextSize("Ag", font, caption_font_scale, 1)
        cy = grid_y_offset + nrows * cell_h + padding + line_th + 8
        for line in caption_lines:
            (lw, _), _ = cv2.getTextSize(line, font, caption_font_scale, 1)
            cx = (canvas_w - lw) // 2
            cv2.putText(canvas, line, (cx, cy), font, caption_font_scale, (80, 80, 80), 1, cv2.LINE_AA)
            cy += line_th + 8

    return canvas


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    input_png = Path(INPUT_PNG).resolve()
    output_dir = (
        Path(OUTPUT_DIR).resolve()
        if OUTPUT_DIR
        else input_png.parent / "encrypted_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_png.exists():
        raise FileNotFoundError(f"Input image not found: {input_png}")

    stem = input_png.stem

    # Temp directory for intermediate J2K files
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # AES key/nonce (same for all levels so results are comparable)
    key   = get_random_bytes(16)
    nonce = get_random_bytes(8)

    # ── Step 1: Encode input PNG → lossless J2K (once) ──
    # First, ensure the input is RGB (not RGBA). JJ2000 cannot decode
    # 4-component J2K files produced by OpenJPEG (wavelet/component
    # transform mismatch). Strip the alpha channel if present.
    img_check = cv2.imread(str(input_png), cv2.IMREAD_UNCHANGED)
    if img_check is None:
        raise RuntimeError(f"Could not read input image: {input_png}")

    actual_input = input_png
    if img_check.ndim == 3 and img_check.shape[2] == 4:
        print(f"[!] Input has 4 channels (RGBA). Stripping alpha for J2K compatibility ...")
        rgb_only = img_check[:, :, :3]  # drop alpha channel
        rgb_path = tmp_dir / f"{stem}_rgb.png"
        cv2.imwrite(str(rgb_path), rgb_only)
        actual_input = rgb_path
        print(f"    Saved 3-channel version: {rgb_path.name}")
    elif img_check.ndim == 2:
        print(f"[!] Input is grayscale ({img_check.shape})")
    else:
        print(f"    Input: {img_check.shape[1]}x{img_check.shape[0]}, {img_check.shape[2]} channels")

    plain_j2k = tmp_dir / f"{stem}_plain.j2k"
    print(f"[1] Encoding {actual_input.name} -> lossless J2K ...")
    encode_lossless_j2k(OPJ_COMPRESS, actual_input, plain_j2k)

    # Verify plain decode works
    dec_plain_png = tmp_dir / f"{stem}_dec_plain.png"
    plain_ok = decode_j2k_jj2000(JAVA_EXE, JJ2000_CLASSPATH, plain_j2k, dec_plain_png)
    print(f"    Plain decode OK = {plain_ok}")
    if not plain_ok:
        # List what's actually in the tmp dir for debugging
        tmp_files = sorted(tmp_dir.glob("*"))
        print(f"\n    [DEBUG] Files in {tmp_dir}:")
        for f in tmp_files:
            print(f"      {f.name}  ({f.stat().st_size:,} bytes)")
        print(f"    [DEBUG] JJ2000_CLASSPATH = {JJ2000_CLASSPATH}")
        print(f"    [DEBUG] JJ2000_CLASSPATH exists = {JJ2000_CLASSPATH.exists()}")
        if JJ2000_CLASSPATH.exists():
            class_files = list(JJ2000_CLASSPATH.glob("*.class"))
            print(f"    [DEBUG] .class files in classpath: {len(class_files)}")
            jj2k = list(JJ2000_CLASSPATH.glob("JJ2K*"))
            print(f"    [DEBUG] JJ2K* files: {[f.name for f in jj2k[:5]]}")
        raise RuntimeError(
            f"Plain J2K decode failed. Check JJ2000 / Java setup.\n"
            f"    JAVA_EXE = {JAVA_EXE}\n"
            f"    JJ2000_CLASSPATH = {JJ2000_CLASSPATH}\n"
            f"    Plain decode OK = {plain_ok}\n"
            f"See [DIAG] messages above for details."
        )

    # ── Step 2: Encrypt at each level and decode ──
    decoded_paths: List[Path] = []
    labels: List[str] = []

    for pct in PERCENT_LIST:
        print(f"\n[p={pct:3d}%] ", end="")

        out_png = output_dir / f"{stem}_p{pct:03d}.png"

        if pct == 0:
            # p=0% is just the plain decoded image (lossless round-trip)
            shutil.copy2(dec_plain_png, out_png)
            decoded_paths.append(out_png)
            labels.append(f"p = {pct}%")
            print("copied plain decode")
            continue

        # Encrypt
        cfg = J2KSelectiveEncryptConfig(enc_percent=pct, key=key, nonce=nonce)
        enc_j2k = tmp_dir / f"{stem}_enc_p{pct:03d}.j2k"

        try:
            _, enc_len = selective_encrypt_j2k(plain_j2k, enc_j2k, cfg)
            print(f"encrypted {enc_len} bytes -> ", end="")
        except Exception as e:
            print(f"encrypt failed: {e}")
            continue

        # Decode encrypted J2K -> PNG (with automatic PGM merging)
        ok = decode_j2k_jj2000(JAVA_EXE, JJ2000_CLASSPATH, enc_j2k, out_png)
        print(f"decode {'OK' if ok else 'FAILED'}")

        if ok:
            decoded_paths.append(out_png)
            labels.append(f"p = {pct}%")
        else:
            print(f"  [WARN] Skipping p={pct}% (decode failed)")

    # ── Step 3: Create combined grid image ──
    if len(decoded_paths) < 2:
        print("\n[WARN] Too few successful decodes to create a grid.")
        return

    print(f"\n[Grid] Creating combined grid with {len(decoded_paths)} images ...")
    grid = make_grid_image(
        decoded_paths,
        labels,
        ncols=GRID_COLS,
        label_font_scale=1.05,       # ← adjust label size here (0.5=small, 0.8=large, 1.0=very large)
        label_thickness=1,            # ← 1=thin, 2=bold
        title="",
        caption=" "
                " "
                "",
    )

    grid_path = output_dir / f"{stem}_encryption_progression.png"
    cv2.imwrite(str(grid_path), grid)

    print(f"\n{'='*60}")
    print(f"  Done! {len(decoded_paths)} encryption levels processed.")
    print(f"  Individual PNGs : {output_dir}")
    print(f"  Grid image      : {grid_path}")
    print(f"  Grid size       : {grid.shape[1]}x{grid.shape[0]} px")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()