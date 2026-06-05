"""
Network-level quantization: produce FP32 / FP16 / INT8 variants of all three models,
save them to disk, measure file size and inference latency.
"""
import sys, os, time, copy
import numpy as np
import cv2
import torch
from onnxruntime.quantization import quantize_dynamic as onnx_quantize_dynamic, QuantType

import config as C

for d in (C.MAGFACE_DIR, C.ADAFACE_DIR, C.ARCFACE_DIR):
    if d not in sys.path:
        sys.path.append(d)

QUANT_DIR = C.EMBEDDINGS_DIR.parent / "quantized_models"
QUANT_DIR.mkdir(exist_ok=True)


# ---------------- Model loaders ----------------
def load_magface_cpu():
    from inference.network_inf import builder_inf
    class Args:
        arch = "iresnet100"
        embedding_size = 512
        cpu_mode = True
        resume = C.MAGFACE_PTH
    return builder_inf(Args()).eval()

def load_adaface_cpu():
    import net
    sd = torch.load(C.ADAFACE_CKPT, map_location="cpu")["state_dict"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
    m = net.build_model("ir_101")
    m.load_state_dict(sd)
    return m.eval()


# ---------------- Sample images for benchmarking ----------------
def load_sample_images(n=120):
    from extract_embeddings import align_from_raw
    samples = []
    for d in sorted(os.listdir(C.LFW_RAW_DIR)):
        if len(samples) >= n: break
        try:
            files = sorted(os.listdir(C.LFW_RAW_DIR / d))
            for f in files:
                if not f.endswith(".jpg"): continue
                idx = int(f.rsplit("_", 1)[-1].split(".")[0])
                img = align_from_raw(d, idx)
                if img is not None:
                    samples.append(img); break
        except Exception:
            continue
    return samples


# ---------------- Helpers ----------------
def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

class TupleUnwrap(torch.nn.Module):
    """AdaFace returns (emb, norm); unwrap for clean benchmarking."""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, tuple) else out

def preprocess_for_bench(img_bgr, dtype=np.float32):
    """Generic preprocessing — only used to feed the model with realistic input."""
    x = (img_bgr.astype(np.float32) - 127.5) / 127.5
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x.astype(dtype)


# ---------------- Bench functions ----------------
def bench_pytorch(model, samples, device, half=False, n_warmup=10, n_iter=100):
    model = model.to(device).eval()
    tensors = []
    for img in samples[:n_warmup + n_iter]:
        t = torch.from_numpy(preprocess_for_bench(img)).to(device)
        if half: t = t.half()
        tensors.append(t)

    with torch.no_grad():
        for t in tensors[:n_warmup]: _ = model(t)
        if device == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        for t in tensors[n_warmup:]: _ = model(t)
        if device == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / n_iter) * 1000  # ms


def bench_onnx(session, samples, n_warmup=10, n_iter=100):
    inp = session.get_inputs()[0].name
    arrays = [preprocess_for_bench(img) for img in samples[:n_warmup + n_iter]]
    for x in arrays[:n_warmup]: _ = session.run(None, {inp: x})
    start = time.perf_counter()
    for x in arrays[n_warmup:]: _ = session.run(None, {inp: x})
    elapsed = time.perf_counter() - start
    return (elapsed / n_iter) * 1000


# ---------------- Main ----------------
def main():
    print("Loading sample images for benchmarking...")
    samples = load_sample_images(120)
    print(f"  {len(samples)} samples loaded\n")

    results = []

    # ------- MagFace -------
    print("=== MagFace ===")
    mag = load_magface_cpu()

    if torch.cuda.is_available():
        m = copy.deepcopy(mag).cuda()
        path = QUANT_DIR / "magface_fp32.pt"
        torch.save(m.state_dict(), path)
        sz, lat = file_size_mb(path), bench_pytorch(m, samples, "cuda", half=False)
        print(f"  FP32 (GPU):  {sz:6.1f} MB | {lat:6.2f} ms/img")
        results.append(("MagFace", "FP32 GPU", sz, lat)); del m

        m = copy.deepcopy(mag).cuda().half()
        path = QUANT_DIR / "magface_fp16.pt"
        torch.save(m.state_dict(), path)
        sz, lat = file_size_mb(path), bench_pytorch(m, samples, "cuda", half=True)
        print(f"  FP16 (GPU):  {sz:6.1f} MB | {lat:6.2f} ms/img")
        results.append(("MagFace", "FP16 GPU", sz, lat)); del m

    m = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(mag), {torch.nn.Linear}, dtype=torch.qint8)
    path = QUANT_DIR / "magface_int8_dyn.pt"
    torch.save(m.state_dict(), path)
    sz, lat = file_size_mb(path), bench_pytorch(m, samples, "cpu", half=False)
    print(f"  INT8 dyn (CPU): {sz:5.1f} MB | {lat:6.2f} ms/img")
    results.append(("MagFace", "INT8 CPU", sz, lat))

    # ------- AdaFace -------
    print("\n=== AdaFace ===")
    ada = load_adaface_cpu()

    if torch.cuda.is_available():
        a = copy.deepcopy(ada).cuda()
        path = QUANT_DIR / "adaface_fp32.pt"
        torch.save(a.state_dict(), path)
        sz, lat = file_size_mb(path), bench_pytorch(TupleUnwrap(a), samples, "cuda", half=False)
        print(f"  FP32 (GPU):  {sz:6.1f} MB | {lat:6.2f} ms/img")
        results.append(("AdaFace", "FP32 GPU", sz, lat)); del a

        a = copy.deepcopy(ada).cuda().half()
        path = QUANT_DIR / "adaface_fp16.pt"
        torch.save(a.state_dict(), path)
        sz, lat = file_size_mb(path), bench_pytorch(TupleUnwrap(a), samples, "cuda", half=True)
        print(f"  FP16 (GPU):  {sz:6.1f} MB | {lat:6.2f} ms/img")
        results.append(("AdaFace", "FP16 GPU", sz, lat)); del a

    a = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(ada), {torch.nn.Linear}, dtype=torch.qint8)
    path = QUANT_DIR / "adaface_int8_dyn.pt"
    torch.save(a.state_dict(), path)
    sz, lat = file_size_mb(path), bench_pytorch(TupleUnwrap(a), samples, "cpu", half=False)
    print(f"  INT8 dyn (CPU): {sz:5.1f} MB | {lat:6.2f} ms/img")
    results.append(("AdaFace", "INT8 CPU", sz, lat))

    # ------- ArcFace ONNX -------
    print("\n=== ArcFace (ONNX) ===")
    import onnxruntime as ort

    sess = ort.InferenceSession(C.ARCFACE_ONNX,
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    sz, lat = file_size_mb(C.ARCFACE_ONNX), bench_onnx(sess, samples)
    print(f"  FP32:        {sz:6.1f} MB | {lat:6.2f} ms/img")
    results.append(("ArcFace", "FP32 ONNX", sz, lat))

    int8_path = str(QUANT_DIR / "arcface_int8.onnx")
    print("  Quantizing ArcFace ONNX to INT8 (this takes a moment)...")
    onnx_quantize_dynamic(C.ARCFACE_ONNX, int8_path, weight_type=QuantType.QUInt8)
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    sz, lat = file_size_mb(int8_path), bench_onnx(sess_int8, samples)
    print(f"  INT8 ONNX:   {sz:6.1f} MB | {lat:6.2f} ms/img")
    results.append(("ArcFace", "INT8 ONNX", sz, lat))

    # ------- Summary -------
    print("\n\n=== Summary Table ===")
    print(f"  {'Model':10s} | {'Precision':12s} | {'Size (MB)':>10s} | {'Latency (ms/img)':>17s}")
    print("-" * 62)
    for model, prec, sz, lat in results:
        print(f"  {model:10s} | {prec:12s} | {sz:10.1f} | {lat:17.2f}")

if __name__ == "__main__":
    main()