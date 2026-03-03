from pathlib import Path
import shutil
import cv2
import numpy as np
from tqdm import tqdm

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop


def iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    base = Path(__file__).resolve().parents[1]

    # ---- SET THESE PATHS ----
    RAW_LFW = base / "data" / "lfw_raw"          # put original LFW here
    OUT = base / "data" / "lfw_aligned_png"      # will be created

    # -------------------------
    if not RAW_LFW.exists():
        raise RuntimeError(f"Raw LFW folder not found: {RAW_LFW}\n"
                           f"Put LFW images into {RAW_LFW} as <person>/<img>.jpg")

    OUT.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 for CPU

    # First pass: align everything we can
    kept = []  # (identity, out_path)
    for img_path in tqdm(list(iter_images(RAW_LFW)), desc="Aligning"):
        rel = img_path.relative_to(RAW_LFW)
        if len(rel.parts) < 2:
            continue
        identity = rel.parts[0]
        out_dir = OUT / identity
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (img_path.stem + ".png")

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        # choose largest detected face
        areas = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            areas.append(max(0, x2 - x1) * max(0, y2 - y1))
        face = faces[int(np.argmax(areas))]

        # landmark-based alignment to 112x112
        kps = face.kps.astype(np.float32)
        aligned = norm_crop(img, kps, image_size=112)  # returns BGR 112x112

        cv2.imwrite(str(out_path), aligned)
        kept.append((identity, out_path))

    # Second pass: remove identities with <2 images (needed for verification pairs)
    # Count per identity
    counts = {}
    for identity, _ in kept:
        counts[identity] = counts.get(identity, 0) + 1

    removed = 0
    for identity, cnt in counts.items():
        if cnt < 2:
            # delete folder
            shutil.rmtree(OUT / identity, ignore_errors=True)
            removed += 1

    # Report
    final_ids = [p for p in OUT.iterdir() if p.is_dir()]
    total_imgs = sum(len(list(p.glob("*.png"))) for p in final_ids)

    print("\nDone.")
    print(f"Aligned identities kept: {len(final_ids)}")
    print(f"Total aligned images:   {total_imgs}")
    print(f"Removed identities (<2 imgs): {removed}")
    print(f"Output folder: {OUT}")


if __name__ == "__main__":
    main()