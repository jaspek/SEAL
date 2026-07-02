"""SLLFW (Similar-Looking LFW) pair-list loader.

pairs.txt has two lines per pair (image A, image B) as 'Name/Name_XXXX.jpg'.
The images ARE the original LFW images, so we pull the pre-aligned 112x112 PNG
crops from your existing lfw_aligned_png dir. Label = 1 if same identity else 0
(SLLFW keeps LFW's genuine pairs and swaps in similar-looking imposters)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2


def _aligned_png(aligned_dir, rel_jpg):
    # 'Name/Name_0001.jpg' -> <aligned_dir>/Name/Name_0001.png
    return Path(aligned_dir) / (rel_jpg.rsplit(".", 1)[0] + ".png")


def load_sllfw(pairs_txt, aligned_dir, align_fn=None):
    """align_fn(rel_jpg)->112x112 BGR is an optional fallback used when the
    pre-aligned crop is missing (e.g. align from lfw_raw). Keeps the full 6k pairs."""
    lines = [l.strip() for l in Path(pairs_txt).read_text().splitlines() if l.strip()]
    assert len(lines) % 2 == 0, "expected an even number of lines (2 per pair)"

    images, index_of, pair_idx, missing = [], {}, [], 0

    def get_idx(rel):
        if rel in index_of:
            return index_of[rel]
        p = _aligned_png(aligned_dir, rel)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR) if p.exists() else None
        if img is None and align_fn is not None:
            img = align_fn(rel)                         # fallback: align from raw
        if img is None:
            return None
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        index_of[rel] = len(images)
        images.append(img)
        return index_of[rel]

    for k in range(0, len(lines), 2):
        a_rel, b_rel = lines[k], lines[k + 1]
        ia, ib = get_idx(a_rel), get_idx(b_rel)
        if ia is None or ib is None:
            missing += 1
            continue
        same = int(a_rel.split("/")[0] == b_rel.split("/")[0])
        pair_idx.append((ia, ib, same))

    if missing:
        print(f"  [sllfw] skipped {missing} pairs with a missing aligned crop")
    return np.stack(images).astype(np.uint8), np.array(pair_idx, dtype=np.int64)
