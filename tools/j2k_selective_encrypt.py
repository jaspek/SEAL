import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import csv

import cv2


from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


SOT = b"\xff\x90"  # Start of Tile-part marker in JPEG2000 codestream
SOD = b"\xff\x93"  # Start of Data

SOP = b"\xff\x91"  # Start of Packet
EPH = b"\xff\x92"  # End of Packet Header

def find_packet_body_ranges(data: bytes) -> list[tuple[int, int]]:
    """
    With SOP/EPH enabled:
      SOP .... packet header .... EPH | packet body | SOP .....
    We encrypt only the packet body parts: (EPH+2) .. next SOP
    """
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
            body_end = n
            ranges.append((body_start, body_end))
            break
        else:
            body_end = next_sop
            if body_end > body_start:
                ranges.append((body_start, body_end))
            i = next_sop

    return ranges


def aes_ctr_encrypt_ranges(buf: bytes, ranges: list[tuple[int, int]], enc_percent: float, key: bytes, nonce: bytes) -> tuple[bytes, int]:
    """
    Encrypt enc_percent (%) of TOTAL packet-body bytes, walking ranges in order.
    Returns (new_buf, total_encrypted_bytes).
    """
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
        chunk = bytes(out[a:a+take])
        out[a:a+take] = cipher.encrypt(chunk)
        done += take

    return bytes(out), done


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def find_payload_start_offset(data: bytes) -> int:
    """
    Return a safe start offset for encryption: right AFTER the first SOD marker.
    This keeps the JPEG2000 headers + tile-part headers intact, so decoding still works.
    """
    sot = data.find(SOT)
    if sot == -1:
        raise ValueError("SOT marker not found. Not a valid J2K codestream?")

    # Find the first SOD after the first SOT
    sod = data.find(SOD, sot + 2)
    if sod == -1:
        raise ValueError("SOD marker not found after SOT. Cannot locate payload start.")

    start = sod + 2  # skip the SOD marker itself
    if start >= len(data):
        raise ValueError("SOD found but file too short.")
    return start


def aes_ctr_encrypt_region(buf: bytes, start: int, length: int, key: bytes, nonce: bytes) -> bytes:
    """
    AES-CTR encrypt exactly buf[start:start+length], keeping file length constant.
    """
    if start < 0 or length < 0 or start + length > len(buf):
        raise ValueError("Invalid encryption region")
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    region = buf[start:start+length]
    enc = cipher.encrypt(region)
    return buf[:start] + enc + buf[start+length:]


@dataclass
class J2KSelectiveEncryptConfig:
    enc_percent: float
    key: bytes
    nonce: bytes
    skip_payload_bytes: int = 0   # NEW

def encode_lossless_j2k(opj_compress: str, in_png: Path, out_j2k: Path) -> None:
    out_j2k.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        opj_compress,
        "-i", str(in_png),
        "-o", str(out_j2k),

        # lossless
        "-r", "1",
        "-n", "1",

        # IMPORTANT: add packet markers so we can encrypt safely
        "-SOP",
        "-EPH",
    ]
    run(cmd)


# def decode_j2k(opj_decompress: str, in_j2k: Path, out_png: Path) -> bool:
#     out_png.parent.mkdir(parents=True, exist_ok=True)
#     cmd = [opj_decompress, "-i", str(in_j2k), "-o", str(out_png)]
#     try:
#         run(cmd)
#         return out_png.exists() and out_png.stat().st_size > 0
#     except subprocess.CalledProcessError:
#         return False
def decode_j2k_jj2000(
    java_exe: str,
    jj2000_classpath: Path,
    in_j2k: Path,
    out_png: Path,
) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    tmp_pgm = out_png.with_suffix(".pgm")

    cmd = [
        java_exe,
        "-Xmx2g",
        "-cp", str(jj2000_classpath),
        "JJ2KDecoder",
        "-i", str(in_j2k),
        "-o", str(tmp_pgm),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return False

    if not tmp_pgm.exists() or tmp_pgm.stat().st_size == 0:
        return False

    img = cv2.imread(str(tmp_pgm), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    # Normalize to 8-bit for ArcFace/InsightFace pipeline
    if img.dtype != "uint8":
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    ok = cv2.imwrite(str(out_png), img)
    try:
        tmp_pgm.unlink(missing_ok=True)
    except Exception:
        pass
    return bool(ok)

def selective_encrypt_j2k(in_j2k: Path, out_j2k: Path, cfg: J2KSelectiveEncryptConfig) -> Tuple[int, int]:
    out_j2k.parent.mkdir(parents=True, exist_ok=True)
    data = in_j2k.read_bytes()

    ranges = find_packet_body_ranges(data)
    if not ranges:
        raise RuntimeError("No SOP/EPH packet body ranges found. Did you encode with -SOP -EPH ?")

    enc_data, enc_len = aes_ctr_encrypt_ranges(data, ranges, cfg.enc_percent, cfg.key, cfg.nonce)

    out_j2k.write_bytes(enc_data)

    # return a "start" just for reporting (first body start)
    return ranges[0][0], enc_len


def batch_process(
    input_png_dir: Path,
    plain_j2k_dir: Path,
    enc_j2k_dir: Path,
    dec_plain_dir: Path,
    dec_enc_dir: Path,
    percent_list: list[float],
    report_csv: Path,
    jj2000_classpath: Path,
    java_exe: str = "java",
    opj_compress: str = "opj_compress",
) -> None:
    key = get_random_bytes(16)    # AES-128
    nonce = get_random_bytes(8)   # CTR nonce

    png_files = sorted([p for p in input_png_dir.rglob("*.png")])
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    with report_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "percent", "header_end", "enc_len", "plain_decode_ok", "enc_decode_ok"])

        for png in png_files:
            rel = png.relative_to(input_png_dir)
            plain_j2k = plain_j2k_dir / rel.with_suffix(".j2k")
            plain_png_dec = dec_plain_dir / rel  # decoded plaintext back to png (sanity)
            encode_lossless_j2k(opj_compress, png, plain_j2k)
            plain_ok = decode_j2k_jj2000(java_exe, jj2000_classpath, plain_j2k, plain_png_dec)

            for pct in percent_list:
                cfg = J2KSelectiveEncryptConfig(enc_percent=pct, key=key, nonce=nonce, skip_payload_bytes=8192)

                out_j2k = enc_j2k_dir / f"p{int(pct):02d}" / rel.with_suffix(".j2k")
                header_end, enc_len = selective_encrypt_j2k(plain_j2k, out_j2k, cfg)

                out_png = dec_enc_dir / f"p{int(pct):02d}" / rel
                enc_ok = decode_j2k_jj2000(java_exe, jj2000_classpath, out_j2k, out_png)

                w.writerow([str(rel), pct, header_end, enc_len, int(plain_ok), int(enc_ok)])


if __name__ == "__main__":
    # Example run configuration (adjust paths)
    base = Path(__file__).resolve().parents[1]

    jj2000_classpath = base / "tools" / "build"
    java_exe = "java"

    input_png_dir = base / "data" / "lfw_aligned_png"
    plain_j2k_dir = base / "data" / "j2k_plain_0_100"
    enc_j2k_dir = base / "data" / "j2k_enc_0_100"
    dec_plain_dir = base / "data" / "dec_plain_png_0_100"
    dec_enc_dir = base / "data" / "dec_enc_png_0_100"
    report_csv = base / "results" / "decode_report_0_100.csv"

    ##percent_list = [0, 2, 4, 6, 8, 10, 12, 15]
    # percent_list = [ 0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100]
    percent_list = [40]

    batch_process(
        input_png_dir=input_png_dir,
        plain_j2k_dir=plain_j2k_dir,
        enc_j2k_dir=enc_j2k_dir,
        dec_plain_dir=dec_plain_dir,
        dec_enc_dir=dec_enc_dir,
        percent_list=percent_list,
        report_csv=report_csv,
        jj2000_classpath=jj2000_classpath,
        java_exe=java_exe,
    )