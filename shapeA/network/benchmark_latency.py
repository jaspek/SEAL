#!/usr/bin/env python3
"""
Device-honest latency benchmark.  ===> RUNS ANYWHERE (CPU here / GPU at home) <===

The seminar's Table 6 is device-confounded (FP32/FP16 on GPU, INT8 on CPU), so
its latency column can't support any claim. Run THIS same script on each device
and quote within-device numbers only.

  python benchmark_latency.py --onnx arcface_int8.onnx           # onnxruntime
  python benchmark_latency.py --onnx arcface_int8.onnx --gpu     # CUDA EP (home box)
"""
import argparse
import time
import numpy as np


def bench_onnx(path, use_gpu, runs=100, warmup=10):
    import onnxruntime as ort
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu
                 else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(path, providers=providers)
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.random.randn(*shape).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        sess.run(None, {inp.name: x})
        t.append((time.perf_counter() - s) * 1000)
    return np.mean(t), np.std(t), sess.get_providers()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()
    m, s, prov = bench_onnx(args.onnx, args.gpu, runs=args.runs)
    print(f"{args.onnx}: {m:.2f} +/- {s:.2f} ms/img  on {prov}")


if __name__ == "__main__":
    main()
