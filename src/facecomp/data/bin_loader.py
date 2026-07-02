"""Decode InsightFace validation .bin packs (lfw/cplfw/calfw/cfp_fp/agedb_30).

Format: pickle -> (encoded_images, issame_list). Images are pre-aligned 112x112,
ordered as consecutive pairs (2k, 2k+1) with issame[k]. There are NO identity
labels, so these datasets support 1:1 verification only (not 1:N)."""
from __future__ import annotations
import pickle
import numpy as np
import cv2


def load_bin(path, swap_rb=False):
    with open(path, "rb") as f:
        try:
            bins, issame = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            bins, issame = pickle.load(f, encoding="bytes")

    images = np.empty((len(bins), 112, 112, 3), dtype=np.uint8)
    for i, b in enumerate(bins):
        raw = b if isinstance(b, (bytes, bytearray)) else bytes(b)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)  # BGR
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        images[i] = img[:, :, ::-1] if swap_rb else img

    issame = np.asarray(issame, dtype=bool)
    pair_idx = np.array([[2 * k, 2 * k + 1, int(issame[k])] for k in range(len(issame))],
                        dtype=np.int64)
    return images, pair_idx
