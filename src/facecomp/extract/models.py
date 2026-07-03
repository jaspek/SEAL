"""Load ArcFace / MagFace / AdaFace and embed pre-aligned 112x112 BGR crops.

Preprocessing matches the seminar pipeline that produced the validated ~98% LFW
numbers. Inference is BATCHED; on the seminar .venv (torch-CUDA + onnxruntime-gpu)
all three models run on the GPU. Paths come from configs/paths.yaml."""
from __future__ import annotations
import os
import sys
import numpy as np
import cv2
import torch
from tqdm import tqdm

from .. import config as C

# Expose torch's bundled CUDA libs (cublasLt64_12.dll, cudnn*, ...) to onnxruntime-gpu
# so its CUDAExecutionProvider loads instead of falling back to CPU with
# "cublasLt64_12.dll is missing". Harmless if the dir/libs aren't there.
if hasattr(os, "add_dll_directory"):
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        try:
            os.add_dll_directory(_torch_lib)
        except OSError:
            pass

# NB: do NOT enable cudnn.benchmark — it can pick a channels-last algo whose
# non-contiguous output breaks MagFace/AdaFace iResNet's `x.view(...)` flatten.
BATCH = 64
_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _add_repo_paths():
    for key in ("magface_repo", "adaface_repo"):
        d = C.MODELS.get(key)
        if d and str(d) not in sys.path:
            sys.path.append(str(d))


def _l2_rows(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _valid_idxs(images):
    return [i for i in range(len(images)) if images[i].any()]


# ---------------- loaders ----------------
def load_arcface():
    """buffalo_l recognition head (w600k_r50) — reproduces the seminar's ArcFace."""
    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
    return app.models["recognition"]


def load_arcface_onnx():
    """Same-backbone control: ArcFace iResNet100 / MS1MV2 (ms1mv2_arcface_r100.onnx).
    log_severity_level=3: the model declares a static {1,512} output shape while
    accepting dynamic batches, so ORT warns once per batch — verified harmless
    (batch vs single outputs agree to float32 noise)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if torch.cuda.is_available() else ["CPUExecutionProvider"])
    return ort.InferenceSession(str(C.MODELS["arcface_onnx"]), sess_options=so,
                                providers=providers)


def load_arcface_int8():
    """Static-PTQ INT8 (QDQ) w600k_r50 from experiments/quantize_arcface.py.
    Runs on the CPU EP deliberately: CPU is the deployment target of int8,
    and the CUDA EP would just dequantize the QDQ nodes anyway."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(C.MODELS["arcface_int8"]), sess_options=so,
                                providers=["CPUExecutionProvider"])


def load_magface():
    _add_repo_paths()
    from inference.network_inf import builder_inf

    class Args:
        arch = "iresnet100"; embedding_size = 512
        cpu_mode = not torch.cuda.is_available(); resume = str(C.MODELS["magface_pth"])
    return builder_inf(Args()).eval().to(_DEV)


def load_adaface():
    _add_repo_paths()
    import net
    sd = torch.load(str(C.MODELS["adaface_ckpt"]), map_location="cpu")["state_dict"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
    m = net.build_model("ir_101"); m.load_state_dict(sd); m.eval()
    return m.to(_DEV)


# ---------------- preprocessing -> CHW float array (verbatim from the seminar) ----------------
def _pre_magface(bgr):
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))


def _pre_adaface(bgr):
    x = ((bgr / 255.0) - 0.5) / 0.5
    return np.transpose(x.astype(np.float32), (2, 0, 1))


# ---------------- batched embedding ----------------
def embed_arcface(rec, images, batch=BATCH):
    """buffalo_l rec.get_feat() batches a list of BGR crops internally."""
    out = np.zeros((len(images), 512), np.float32)
    valid = _valid_idxs(images)
    for s in tqdm(range(0, len(valid), batch), desc="  arcface"):
        idxs = valid[s:s + batch]
        feats = _l2_rows(rec.get_feat([images[i] for i in idxs]).astype(np.float32))
        for j, i in enumerate(idxs):
            out[i] = feats[j]
    return out


def embed_arcface_onnx(sess, images, batch=BATCH):
    """MXNet-era model-zoo convention: RGB, RAW 0-255 pixels, NO normalization.
    Feeding (x-127.5)/127.5 collapses this model to a constant embedding
    (genuine/imposter gap 0.001 vs 0.731 with raw inputs — measured)."""
    inp = sess.get_inputs()[0].name
    out = np.zeros((len(images), 512), np.float32)
    valid = _valid_idxs(images)
    for s in tqdm(range(0, len(valid), batch), desc="  arcface(ms1mv2)"):
        idxs = valid[s:s + batch]
        x = np.stack([cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB).astype(np.float32)
                      .transpose(2, 0, 1) for i in idxs])
        y = _l2_rows(sess.run(None, {inp: x})[0].astype(np.float32))
        for j, i in enumerate(idxs):
            out[i] = y[j]
    return out


def embed_arcface_int8(sess, images, batch=BATCH):
    """Quantized w600k_r50: SAME preprocessing as buffalo (RGB, (x-127.5)/127.5)
    -- NOT the raw 0-255 convention of the ms1mv2 control."""
    inp = sess.get_inputs()[0].name
    out = np.zeros((len(images), 512), np.float32)
    valid = _valid_idxs(images)
    for s in tqdm(range(0, len(valid), batch), desc="  arcface(int8)"):
        idxs = valid[s:s + batch]
        x = np.stack([((cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB).astype(np.float32)
                        - 127.5) / 127.5).transpose(2, 0, 1) for i in idxs])
        y = _l2_rows(sess.run(None, {inp: x})[0].astype(np.float32))
        for j, i in enumerate(idxs):
            out[i] = y[j]
    return out


@torch.no_grad()
def _embed_torch(model, images, pre_fn, desc, batch=BATCH):
    out = np.zeros((len(images), 512), np.float32)
    valid = _valid_idxs(images)
    for s in tqdm(range(0, len(valid), batch), desc=desc):
        idxs = valid[s:s + batch]
        # .contiguous(): np.transpose->np.stack yields a non-contiguous tensor that
        # MagFace/AdaFace iResNet's `x.view(...)` flatten rejects.
        x = torch.from_numpy(np.stack([pre_fn(images[i]) for i in idxs])).to(_DEV).contiguous()
        y = model(x)
        if isinstance(y, tuple):
            y = y[0]
        rows = _l2_rows(y.float().cpu().numpy())
        for j, i in enumerate(idxs):
            out[i] = rows[j]
    return out


def embed_all(images, arcface="buffalo", batch=BATCH):
    """dict name -> (N,512). Loads each model, frees it before the next (VRAM-friendly)."""
    print(f"  device: {_DEV}  (batch={batch})")
    res = {}
    if arcface == "ms1mv2":
        sess = load_arcface_onnx(); res["arcface"] = embed_arcface_onnx(sess, images, batch); del sess
    elif arcface == "int8":
        sess = load_arcface_int8(); res["arcface"] = embed_arcface_int8(sess, images, batch); del sess
    else:
        rec = load_arcface(); res["arcface"] = embed_arcface(rec, images, batch); del rec
    mag = load_magface(); res["magface"] = _embed_torch(mag, images, _pre_magface, "  magface", batch); del mag
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ada = load_adaface(); res["adaface"] = _embed_torch(ada, images, _pre_adaface, "  adaface", batch); del ada
    return res
