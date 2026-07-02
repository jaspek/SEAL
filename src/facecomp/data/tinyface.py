"""TinyFace 1:N loader.  Testing_Set/{Gallery_Match, Probe, Gallery_Distractor}.

Identity = filename prefix before '_' (e.g. 1000_9.jpg -> 1000). Distractors carry
no identity -> label -1. `split` code: 0=gallery_match, 1=probe, 2=distractor.
Faces are native low-res, so we resize to 112x112 and DO NOT re-detect."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

GALLERY, PROBE, DISTRACTOR = 0, 1, 2


def _list(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def load_tinyface(root, max_distractors=None, size=112):
    ts = Path(root) / "Testing_Set"
    gm, pr, di = _list(ts / "Gallery_Match"), _list(ts / "Probe"), _list(ts / "Gallery_Distractor")
    if max_distractors is not None:
        di = di[:max_distractors]

    entries = ([(p, p.name.split("_")[0], GALLERY) for p in gm] +
               [(p, p.name.split("_")[0], PROBE) for p in pr] +
               [(p, None, DISTRACTOR) for p in di])

    id_strs = [e[1] for e in entries if e[1] is not None]
    name2id = {n: i for i, n in enumerate(dict.fromkeys(sorted(id_strs)))}

    images = np.zeros((len(entries), size, size, 3), dtype=np.uint8)
    labels = np.empty(len(entries), dtype=np.int64)
    split = np.empty(len(entries), dtype=np.int64)
    for i, (p, ids, sp) in enumerate(tqdm(entries, desc="  tinyface read")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            images[i] = cv2.resize(img, (size, size))
        labels[i] = name2id[ids] if ids is not None else -1
        split[i] = sp
    print(f"  tinyface: {len(gm)} gallery, {len(pr)} probe, {len(di)} distractor, {len(name2id)} ids")
    return images, labels, split
