"""
Sanity check: verify all four components load correctly:
- Face detector (InsightFace, used only for detection + alignment)
- ArcFace recognition (ONNX, MS1MV2 R100)
- MagFace recognition (PyTorch, MS1MV2 R100)
- AdaFace recognition (PyTorch, MS1MV2 R101)
Run this first before touching extract_embeddings.py.
"""
import sys
import os
import numpy as np
import cv2

# ---------------------------------------------------------------
# Add MagFace, AdaFace, and ArcFace folders to Python path
# Self-contained — works regardless of PyCharm interpreter env vars
# ---------------------------------------------------------------
MAGFACE_DIR = r"C:\Users\jasko\Programming\Python\Libraries\MagFace"
ADAFACE_DIR = r"C:\Users\jasko\Programming\Python\Libraries\AdaFace"
ARCFACE_DIR = r"C:\Users\jasko\Programming\Python\Libraries\ArcFace"

for d in (MAGFACE_DIR, ADAFACE_DIR, ARCFACE_DIR):
    if d not in sys.path:
        sys.path.append(d)


def check_face_detector():
    """InsightFace's FaceAnalysis — used only for detection + alignment.
       NOT used as the recognition model anymore (we use ArcFace ONNX for that)."""
    print("\n=== Face Detector (InsightFace) ===")
    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Face detector loaded OK.")
    return app


def check_arcface():
    """ArcFace MS1MV2 R100 (ONNX) — parity with MagFace/AdaFace."""
    print("\n=== ArcFace (ONNX, MS1MV2 R100) ===")
    import onnxruntime as ort

    ckpt_path = os.path.join(ARCFACE_DIR, "pretrained", "ms1mv2_arcface_r100.onnx")
    if not os.path.exists(ckpt_path):
        print(f"  ! Checkpoint not found: {ckpt_path}")
        print("  ! Download from https://github.com/deepinsight/insightface/tree/master/model_zoo")
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(ckpt_path, providers=providers)

    # Sanity check: print input/output info
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"  input:  {inp.name} {inp.shape}")
    print(f"  output: {out.name} {out.shape}")
    print("ArcFace (ONNX) loaded OK.")
    return session


def check_magface():
    print("\n=== MagFace ===")
    import torch
    from inference.network_inf import builder_inf

    class Args:
        arch = "iresnet100"
        embedding_size = 512
        cpu_mode = not torch.cuda.is_available()
        resume = os.path.join(MAGFACE_DIR, "checkpoints", "magface_epoch_00025.pth")
        # Some MagFace versions also need:
        # last_fc_size = 512
        # arc_scale = 64

    if not os.path.exists(Args.resume):
        print(f"  ! Checkpoint not found: {Args.resume}")
        print("  ! Download it from the MagFace repo README (Google Drive link).")
        return None

    model = builder_inf(Args())
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("MagFace loaded OK.")
    return model


def check_adaface():
    print("\n=== AdaFace ===")
    import torch
    import net  # AdaFace's net.py at the repo root

    ckpt_path = os.path.join(ADAFACE_DIR, "pretrained", "adaface_ir101_ms1mv2.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"  ! Checkpoint not found: {ckpt_path}")
        print("  ! Download it from the AdaFace repo README (Google Drive link).")
        return None

    model = net.build_model("ir_101")
    statedict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    # Strip the Lightning "model." prefix
    model_statedict = {k[6:]: v for k, v in statedict.items() if k.startswith("model.")}
    model.load_state_dict(model_statedict)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("AdaFace loaded OK.")
    return model


if __name__ == "__main__":
    print("Hey, doing model sanity checks...")

    import torch
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA avail: {torch.cuda.is_available()}")

    detector = check_face_detector()
    arc = check_arcface()
    mag = check_magface()
    ada = check_adaface()

    print("\n=== Summary ===")
    print(f"Face detector: {'OK' if detector is not None else 'FAILED'}")
    print(f"ArcFace:       {'OK' if arc is not None else 'FAILED'}")
    print(f"MagFace:       {'OK' if mag is not None else 'FAILED'}")
    print(f"AdaFace:       {'OK' if ada is not None else 'FAILED'}")