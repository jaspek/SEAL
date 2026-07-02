import numpy as np
import cv2
from pathlib import Path
from extract_embeddings import align_from_raw
import config as C

data = np.load(C.EMBEDDINGS_FILE, allow_pickle=True)
arc, mag, ada = data["arcface"], data["magface"], data["adaface"]
pair_idx = data["pair_idx"]
paths = data["paths"]
labels = pair_idx[:, 2]

# Recompute the same 'all_wrong' mask from inspect_errors.py
def wrong_mask(emb):
    s = np.sum(emb[pair_idx[:, 0]] * emb[pair_idx[:, 1]], axis=1)
    best_acc, best_thr = 0, 0
    for thr in np.arange(-1, 1, 0.005):
        acc = ((s > thr) == labels).mean()
        if acc > best_acc: best_acc, best_thr = acc, thr
    return (s > best_thr) != labels

all_wrong = wrong_mask(arc) & wrong_mask(mag) & wrong_mask(ada)

out_dir = Path(r"C:\Users\jasko\Programming\Python\Program_6\error_crops")
out_dir.mkdir(exist_ok=True)

# Dump first 10 universally-wrong pairs as side-by-side images
count = 0
for i in np.where(all_wrong)[0]:
    if count >= 10: break
    a_idx, b_idx, lbl = pair_idx[i]
    p1 = Path(str(paths[a_idx]))
    p2 = Path(str(paths[b_idx]))
    name1, fname1 = p1.parent.name, p1.stem
    name2, fname2 = p2.parent.name, p2.stem
    idx1 = int(fname1.rsplit("_", 1)[-1])
    idx2 = int(fname2.rsplit("_", 1)[-1])

    img1 = align_from_raw(name1, idx1)
    img2 = align_from_raw(name2, idx2)
    if img1 is None or img2 is None: continue

    side = np.hstack([img1, img2])
    out = out_dir / f"pair_{count:02d}_{name1}_{idx1}_vs_{name2}_{idx2}.png"
    cv2.imwrite(str(out), side)
    count += 1

print(f"Wrote {count} side-by-side error crops to {out_dir}")