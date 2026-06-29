#!/usr/bin/env python3
"""
STATIC INT8 PTQ of an ONNX face recognizer (calibrated).  ===> RUNS ON CPU <===

This is the experiment that closes the seminar's own gap: dynamic INT8 only
quantized the final Linear layer (1.2x). Static INT8 with calibration quantizes
the CONVOLUTIONS too -> the real ~4x on MagFace/AdaFace, not just ArcFace.

You only need onnxruntime (CPU) and a folder of a few hundred aligned 112x112
crops for calibration. No GPU, no retraining.

  pip install onnx onnxruntime
  python quantize_onnx_static.py --model arcface.onnx --calib crops/ --out arcface_int8.onnx

Wiring you must check:
  * INPUT_NAME / input layout (NCHW vs NHWC) and the normalization must match how
    the model was exported. Defaults below assume NCHW, BGR, (x-127.5)/127.5
    (ArcFace/buffalo_l convention). Adjust per model.
"""
import argparse
import os
import glob
import numpy as np

import onnx
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat)
from onnxruntime.quantization.shape_inference import quant_pre_process

try:
    import cv2
except Exception:
    cv2 = None


def load_crop(path, layout="NCHW", mean=127.5, std=127.5, to_rgb=False):
    img = cv2.imread(path)                              # BGR, HxWx3
    img = cv2.resize(img, (112, 112))
    if to_rgb:
        img = img[:, :, ::-1]
    x = (img.astype(np.float32) - mean) / std
    if layout == "NCHW":
        x = np.transpose(x, (2, 0, 1))                 # 3x112x112
    return x[None].astype(np.float32)                  # 1x3x112x112


class FolderCalibReader(CalibrationDataReader):
    def __init__(self, folder, input_name, limit=300, **kw):
        self.input_name = input_name
        self.files = sorted(glob.glob(os.path.join(folder, "**", "*.*"),
                                      recursive=True))[:limit]
        self.kw = kw
        self._it = iter(self.files)

    def get_next(self):
        try:
            f = next(self._it)
        except StopIteration:
            return None
        return {self.input_name: load_crop(f, **self.kw)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True, help="folder of aligned crops")
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-name", default=None)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--to-rgb", action="store_true")
    args = ap.parse_args()
    assert cv2 is not None, "pip install opencv-python-headless"

    pre = args.model.replace(".onnx", ".preproc.onnx")
    quant_pre_process(args.model, pre)                 # shape inference + cleanup

    if args.input_name is None:
        args.input_name = onnx.load(pre).graph.input[0].name
        print("input name:", args.input_name)

    reader = FolderCalibReader(args.calib, args.input_name,
                               limit=args.limit, to_rgb=args.to_rgb)
    quantize_static(
        pre, args.out, reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,                              # per-channel weights = the win
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    a = os.path.getsize(args.model) / 1e6
    b = os.path.getsize(args.out) / 1e6
    print(f"{a:.1f} MB -> {b:.1f} MB   ({a/b:.2f}x)")
    print("now re-extract embeddings with the INT8 model and feed them to "
          "run_rate_distortion.py to confirm accuracy is unchanged.")


if __name__ == "__main__":
    main()
