"""Embedding bundle: load emb_<dataset>.npz, fuse, split for 1:N, or synthesize.

npz contract (all optional except >=1 model array):
  arcface/magface/adaface : (N, D) float32
  labels                  : (N,)  int   identity per image  (for 1:N)
  pair_idx                : (M, 3) int   [img_a, img_b, same?]  (for 1:1 LFW protocol)
  paths                   : (N,)  str   (labels derived from parent folder if absent)
"""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np

MODELS = ("arcface", "magface", "adaface")


def _l2(x, eps=1e-10):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def labels_from_paths(paths):
    names = [os.path.basename(os.path.dirname(str(p))) for p in paths]
    uniq = {n: i for i, n in enumerate(dict.fromkeys(names))}
    return np.array([uniq[n] for n in names], dtype=np.int64)


class EmbBundle:
    def __init__(self, per_model, labels=None, pair_idx=None, paths=None, split=None):
        self.per_model = per_model
        self.labels = labels
        self.pair_idx = pair_idx
        self.paths = paths
        self.split = split          # None, or per-image {0=gallery,1=probe,2=distractor} (TinyFace)
        self.N = next(iter(per_model.values())).shape[0]

    def fuse(self, models=MODELS, method="concat", renorm=True):
        arrs = [self.per_model[m] for m in models if m in self.per_model]
        if not arrs:
            raise ValueError("no requested models present in bundle")
        if method == "concat":
            f = np.concatenate(arrs, axis=1)
        elif method == "mean":
            f = np.mean(arrs, axis=0)
        else:
            raise ValueError(f"unknown fusion method {method!r}")
        return _l2(f) if renorm else f


def load_emb(path, models=MODELS) -> EmbBundle:
    d = np.load(Path(path), allow_pickle=True)
    per_model = {m: _l2(d[m].astype(np.float32)) for m in models if m in d.files}
    if not per_model:
        raise ValueError(f"{path}: none of {models} present; keys={list(d.files)}")
    labels = d["labels"].astype(np.int64) if "labels" in d.files else None
    paths = d["paths"] if "paths" in d.files else None
    if labels is None and paths is not None:
        labels = labels_from_paths(paths)
    pair_idx = d["pair_idx"].astype(np.int64) if "pair_idx" in d.files else None
    split = d["split"].astype(np.int64) if "split" in d.files else None
    return EmbBundle(per_model, labels, pair_idx, paths, split)


def make_synthetic(n_ids=100, per_id=6, dim=512, spread=0.8, seed=0):
    rng = np.random.default_rng(seed)
    centers = _l2(rng.standard_normal((n_ids, dim)))
    labels = np.repeat(np.arange(n_ids), per_id)
    # scale noise so its L2 norm ~= `spread` (comparable to the unit-norm center),
    # otherwise per-component noise swamps the identity signal in high dim.
    noise = rng.standard_normal((n_ids * per_id, dim)) * (spread / np.sqrt(dim))
    emb = _l2(centers[labels] + noise)
    return emb.astype(np.float32), labels.astype(np.int64)


def make_synthetic_bundle(n_ids=100, per_id=6, dim=512, seed=0):
    per_model, labels = {}, None
    for i, m in enumerate(MODELS):
        emb, labels = make_synthetic(n_ids, per_id, dim, spread=0.8 + 0.05 * i, seed=seed + i)
        per_model[m] = emb
    return EmbBundle(per_model, labels=labels)


def gallery_probe_split(labels, n_enroll=1, seed=0):
    """n_enroll image(s) per identity -> gallery, the rest -> probes. Singletons skipped."""
    rng = np.random.default_rng(seed)
    gallery, probe = [], []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        if len(idx) < 2:
            continue
        gallery.extend(idx[:n_enroll])
        probe.extend(idx[n_enroll:])
    return np.array(gallery, dtype=np.int64), np.array(probe, dtype=np.int64)


def add_distractors(gallery_emb, gallery_labels, num, seed=0):
    """Append `num` random L2-normalized vectors labelled -1 (no true match)."""
    if num <= 0:
        return gallery_emb, gallery_labels
    rng = np.random.default_rng(seed + 777)
    d = _l2(rng.standard_normal((num, gallery_emb.shape[1])).astype(np.float32))
    dl = np.full(num, -1, dtype=np.int64)
    return np.vstack([gallery_emb, d]), np.concatenate([gallery_labels, dl])
