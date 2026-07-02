"""Detect + align a face from a raw LFW image (RetinaFace via buffalo_l), matching
the seminar's alignment so raw-aligned crops are consistent with the cached ones.
Used as a fallback when a dataset references an image that isn't pre-aligned."""
from __future__ import annotations
from pathlib import Path
import cv2

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        import insightface
        import torch
        app = insightface.app.FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        _detector = app
    return _detector


def align_from_raw(rel_jpg, raw_dir):
    """rel_jpg like 'Name/Name_0001.jpg' -> 112x112 BGR crop, or None."""
    from insightface.utils import face_align
    p = Path(raw_dir) / rel_jpg
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    if img is None:
        return None
    faces = _get_detector().get(img)
    if not faces:
        return None
    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face_align.norm_crop(img, landmark=f.kps, image_size=112)
