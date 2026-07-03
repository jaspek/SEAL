"""Static post-training INT8 quantization (QDQ) of the buffalo_l ArcFace
w600k_r50.onnx -- the NETWORK-compression counterpart to the template
compression study.

Calibration images come from an InsightFace .bin pack (default: lfw.bin) and
are preprocessed EXACTLY like extraction does (RGB, (x-127.5)/127.5, NCHW);
a range mismatch here would silently wreck the activation scales.

  python experiments/quantize_arcface.py
  python experiments/quantize_arcface.py --reduce-range   # if cosine check is poor

Prints file sizes, fp32-vs-int8 embedding cosine on held-out images, and a
CPU latency comparison -- the numbers for the paper's network-compression table.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import onnx
from onnx import version_converter
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

from facecomp import config as C
from facecomp.data import bin_loader


def preprocess(bgr):
    """Must match models.embed_arcface / insightface ArcFaceONNX.get_feat."""
    x = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
    return x.transpose(2, 0, 1)


class BinReader(CalibrationDataReader):
    def __init__(self, images, idxs, input_name, batch=8):
        self.batches = [
            np.stack([preprocess(images[i]) for i in idxs[s:s + batch]])
            for s in range(0, len(idxs), batch)]
        self.input_name = input_name
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.batches):
            return None
        b = self.batches[self.pos]
        self.pos += 1
        return {self.input_name: b}


def cpu_session(path):
    so = SessionOptions()
    so.log_severity_level = 3
    return InferenceSession(str(path), sess_options=so,
                            providers=["CPUExecutionProvider"])


def embed(sess, images, idxs, batch=16):
    inp = sess.get_inputs()[0].name
    out = []
    for s in range(0, len(idxs), batch):
        x = np.stack([preprocess(images[i]) for i in idxs[s:s + batch]])
        out.append(sess.run(None, {inp: x})[0])
    y = np.concatenate(out).astype(np.float32)
    return y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-10)


def bench(sess, x, warmup=10, iters=100):
    inp = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {inp: x})
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {inp: x})
    return (time.perf_counter() - t0) / iters * 1000  # ms per single image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() /
                    ".insightface/models/buffalo_l/w600k_r50.onnx"))
    ap.add_argument("--dst", default=None,
                    help="default: models.arcface_int8 from paths.yaml, "
                         "else data/models/w600k_r50_int8.onnx")
    ap.add_argument("--calib-bin", default=None,
                    help="default: <data.bin_pack>/lfw.bin")
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reduce-range", action="store_true",
                    help="7-bit weights; try this if the cosine check is < 0.97 "
                         "(helps on CPUs without VNNI)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else (
        C.MODELS.get("arcface_int8") or C.REPO_ROOT / "data/models/w600k_r50_int8.onnx")
    dst.parent.mkdir(parents=True, exist_ok=True)
    calib_bin = Path(args.calib_bin) if args.calib_bin else C.DATA["bin_pack"] / "lfw.bin"

    print(f"src   : {src}")
    print(f"dst   : {dst}")
    print(f"calib : {calib_bin} ({args.calib_n} images, seed {args.seed})")

    images, _ = bin_loader.load_bin(calib_bin)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(images))
    calib_idx = perm[:args.calib_n]
    check_idx = perm[args.calib_n:args.calib_n + 64]   # held out from calibration

    sess32 = cpu_session(src)
    input_name = sess32.get_inputs()[0].name

    pre = dst.with_name(dst.stem + "_pre.onnx")
    print("preprocessing graph (shape inference + optimization)...")
    quant_pre_process(str(src), str(pre))

    # Per-channel QDQ puts an `axis` attribute on DequantizeLinear, which only
    # exists from opset 13 -- w600k_r50 ships with an older opset, so upgrade.
    m = onnx.load(str(pre))
    opset = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), 0)
    per_channel = True
    if opset < 13:
        print(f"model opset {opset}: per-channel QDQ needs >= 13, upgrading to 13...")
        try:
            onnx.save(version_converter.convert_version(m, 13), str(pre))
        except Exception as e:
            print(f"  opset upgrade failed ({e}); falling back to per-tensor weights")
            per_channel = False

    print(f"calibrating + quantizing (QDQ, per_channel={per_channel})...")
    quantize_static(str(pre), str(dst),
                    BinReader(images, calib_idx, input_name),
                    quant_format=QuantFormat.QDQ,
                    per_channel=per_channel,
                    reduce_range=args.reduce_range,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8)
    pre.unlink(missing_ok=True)

    mb32 = src.stat().st_size / 2**20
    mb8 = dst.stat().st_size / 2**20
    print(f"size  : fp32 {mb32:.1f} MB -> int8 {mb8:.1f} MB ({mb32 / mb8:.2f}x smaller)")

    sess8 = cpu_session(dst)
    e32 = embed(sess32, images, check_idx)
    e8 = embed(sess8, images, check_idx)
    cos = (e32 * e8).sum(axis=1)
    print(f"cosine(fp32, int8) on {len(check_idx)} held-out images: "
          f"mean {cos.mean():.4f}  min {cos.min():.4f}")
    if cos.mean() < 0.97:
        print("WARNING: low agreement -- retry with --reduce-range")

    x1 = np.stack([preprocess(images[check_idx[0]])])
    ms32, ms8 = bench(sess32, x1), bench(sess8, x1)
    print(f"CPU latency (batch 1): fp32 {ms32:.1f} ms -> int8 {ms8:.1f} ms "
          f"({ms32 / ms8:.2f}x speedup)")


if __name__ == "__main__":
    main()
