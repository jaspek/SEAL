"""
Extract ArcFace, MagFace, AdaFace embeddings for every LFW image
referenced in pairs.csv. Saves to a single .npz for fast re-use.

Run this ONCE; afterwards run_evaluation.py loads the cached embeddings.
"""
import sys, os, csv
import numpy as np
import cv2
import torch
from tqdm import tqdm
import insightface
from insightface.utils import face_align

import config as C

for d in (C.MAGFACE_DIR, C.ADAFACE_DIR, C.ARCFACE_DIR):
    if d not in sys.path:
        sys.path.append(d)

# ---------------- Detector + alignment fallback ----------------
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = insightface.app.FaceAnalysis(name="buffalo_l")
        _detector.prepare(ctx_id=0, det_size=(640, 640))
    return _detector

def align_from_raw(name, idx):
    """Detect+align a face from lfw_raw. Returns 112x112 BGR or None."""
    raw_path = C.LFW_RAW_DIR / name / f"{name}_{int(idx):04d}.jpg"
    if not raw_path.exists():
        return None
    img = cv2.imread(str(raw_path))
    if img is None:
        return None
    faces = get_detector().get(img)
    if not faces:
        return None
    largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return face_align.norm_crop(img, landmark=largest.kps, image_size=112)


# ---------------- Model loaders ----------------

def load_arcface():
    """Use the recognition model bundled with InsightFace's buffalo_l pack
       (w600k_r50 ONNX, R50, Glint360K). Known-good with built-in preprocessing
       via .get_feat()."""
    detector = get_detector()
    rec = detector.models['recognition']
    return rec

def load_magface():
    from inference.network_inf import builder_inf
    class Args:
        arch = "iresnet100"
        embedding_size = 512
        cpu_mode = not torch.cuda.is_available()
        resume = C.MAGFACE_PTH
    m = builder_inf(Args()).eval()
    return m.cuda() if torch.cuda.is_available() else m

def load_adaface():
    import net
    sd = torch.load(C.ADAFACE_CKPT, map_location="cpu")["state_dict"]
    sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
    m = net.build_model("ir_101")
    m.load_state_dict(sd)
    m.eval()
    return m.cuda() if torch.cuda.is_available() else m


# ---------------- Preprocessing ----------------

def preprocess_magface(img_bgr):
    """MagFace official inference: RGB, [0, 1] range
       (matches transforms.ToTensor() in MagFace/inference/gen_feat.py)."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)

def preprocess_adaface(img_bgr):
    """AdaFace: BGR, ((x/255)-0.5)/0.5, NCHW tensor."""
    img = ((img_bgr / 255.0) - 0.5) / 0.5
    img = np.transpose(img.astype(np.float32), (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


# ---------------- Embedding ----------------

def l2norm(v):
    return v / (np.linalg.norm(v) + 1e-10)

def embed_arcface(rec, img_bgr):
    """rec.get_feat() handles preprocessing internally; expects 112x112 BGR aligned."""
    feat = rec.get_feat(img_bgr)
    feat = feat.squeeze().astype(np.float32)
    return l2norm(feat)

@torch.no_grad()
def embed_torch(model, x_tensor):
    if torch.cuda.is_available():
        x_tensor = x_tensor.cuda()
    out = model(x_tensor)
    if isinstance(out, tuple):     # AdaFace returns (emb, norm)
        out = out[0]
    return l2norm(out.cpu().numpy().squeeze())


# ---------------- LFW pair parsing ----------------

def parse_pairs(csv_path):
    def img_info(name, idx):
        path = C.LFW_ALIGNED_DIR / name / f"{name}_{int(idx):04d}.png"
        return (path, name, int(idx))

    pairs, all_imgs = [], {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            row = [c.strip() for c in row if c.strip() != ""]
            if len(row) == 3:
                name, i1, i2 = row
                a, b, label = img_info(name, i1), img_info(name, i2), 1
            elif len(row) == 4:
                n1, i1, n2, i2 = row
                a, b, label = img_info(n1, i1), img_info(n2, i2), 0
            else:
                continue
            pairs.append((a[0], b[0], label))
            all_imgs[a[0]] = a
            all_imgs[b[0]] = b
    return pairs, sorted(all_imgs.values(), key=lambda t: str(t[0]))


# ---------------- Main ----------------

def main():
    print("Loading models...")
    arc = load_arcface()
    mag = load_magface()
    ada = load_adaface()

    print(f"Parsing {C.PAIRS_CSV}...")
    pairs, all_imgs = parse_pairs(C.PAIRS_CSV)
    all_paths = [info[0] for info in all_imgs]
    print(f"  {len(pairs)} pairs, {len(all_paths)} unique images")

    path_to_idx = {p: i for i, p in enumerate(all_paths)}
    N = len(all_paths)
    arc_emb = np.zeros((N, 512), dtype=np.float32)
    mag_emb = np.zeros((N, 512), dtype=np.float32)
    ada_emb = np.zeros((N, 512), dtype=np.float32)

    # n_aligned, n_raw_aligned, n_failed = 0, 0, 0
    # for i, (path, name, idx) in enumerate(tqdm(all_imgs, desc="Extracting")):
    #     img = None
    #     if path.exists():
    #         img = cv2.imread(str(path))
    #         if img is not None and img.shape[:2] == (112, 112):
    #             n_aligned += 1
    #         else:
    #             img = None
    #
    #     if img is None:
    #         img = align_from_raw(name, idx)
    #         if img is not None:
    #             n_raw_aligned += 1
    #         else:
    #             n_failed += 1
    #             continue
    #
    #     arc_emb[i] = embed_arcface(arc, img)
    #     mag_emb[i] = embed_torch(mag, preprocess_magface(img))
    #     ada_emb[i] = embed_torch(ada, preprocess_adaface(img))

    n_aligned, n_failed = 0, 0
    for i, (path, name, idx) in enumerate(tqdm(all_imgs, desc="Aligning + Extracting")):
        img = align_from_raw(name, idx)
        if img is None:
            n_failed += 1
            continue
        n_aligned += 1

        arc_emb[i] = embed_arcface(arc, img)
        mag_emb[i] = embed_torch(mag, preprocess_magface(img))
        ada_emb[i] = embed_torch(ada, preprocess_adaface(img))

    print(f"\n  aligned (InsightFace, all from raw): {n_aligned}")
    print(f"  failed: {n_failed}")

    # print(f"\n  pre-aligned: {n_aligned}")
    # print(f"  aligned from raw: {n_raw_aligned}")
    # print(f"  failed: {n_failed}")

    # ----- Filter out pairs containing failed (zero-embedding) images -----
    def is_valid(idx_):
        return (np.linalg.norm(arc_emb[idx_]) > 1e-6 and
                np.linalg.norm(mag_emb[idx_]) > 1e-6 and
                np.linalg.norm(ada_emb[idx_]) > 1e-6)

    valid_pairs = [
        (path_to_idx[a], path_to_idx[b], lbl)
        for a, b, lbl in pairs
        if a in path_to_idx and b in path_to_idx
        and is_valid(path_to_idx[a]) and is_valid(path_to_idx[b])
    ]
    pair_idx = np.array(valid_pairs, dtype=np.int32)
    print(f"  valid pairs after filtering zero embeddings: {len(pair_idx)} / {len(pairs)}")

    np.savez_compressed(
        C.EMBEDDINGS_FILE,
        paths=np.array([str(p) for p in all_paths]),
        arcface=arc_emb,
        magface=mag_emb,
        adaface=ada_emb,
        pair_idx=pair_idx,
    )
    print(f"\nSaved -> {C.EMBEDDINGS_FILE}")

if __name__ == "__main__":
    main()