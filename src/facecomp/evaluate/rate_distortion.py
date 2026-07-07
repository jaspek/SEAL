"""Rate-distortion: sweep compression configs, evaluate 1:N (+ 1:1 if pairs given)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..compress import template as T
from ..data import emb as E
from . import identification as idn
from . import verification as ver


def _fit_reducer(kind, k, X):
    if kind == "pca":
        return T.PCAReducer(k).fit(X)
    if kind == "itq":
        return T.ITQReducer(k).fit(X)
    return None


def default_specs(full_dim):
    # precision axis now sweeps the full 1-8 bit range (int6/int4/int3/int2 fill
    # the gap the paper leaves between binary and int8):
    specs = [(None, None, p) for p in
             ("fp32", "fp16", "int8", "int6", "int4", "int3", "int2", "binary")]
    for k in (256, 128):
        if k < full_dim:
            specs += [("pca", k, "binary"), ("itq", k, "binary")]
    return specs


def _make_gallery(comp, fused, labels, g_idx, prec, reducer, num_distractors, seed):
    """Enrolled gallery + distractors passed through the SAME compressor as the
    gallery. (The legacy path appended raw fp32 Gaussians AFTER compression — a
    code-space mismatch that can make binary templates look artificially flat in
    the gallery-scaling sweep.)"""
    gemb, glab = comp[g_idx], labels[g_idx]
    if num_distractors > 0:
        rng = np.random.default_rng(seed + 777)
        raw = rng.standard_normal((num_distractors, fused.shape[1])).astype(np.float32)
        raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-10)
        dcomp, _ = T.compress(raw, method=prec, dim_reducer=reducer)
        gemb = np.vstack([gemb, dcomp])
        glab = np.concatenate([glab, np.full(num_distractors, -1, np.int64)])
    return gemb, glab


def rate_distortion(fused, labels=None, pair_idx=None, gallery_idx=None, probe_idx=None,
                    specs=None, num_distractors=0, seed=0, fit_emb=None):
    """fit_emb: if given, PCA/ITQ reducers are fit on THIS matrix (a disjoint
    corpus) instead of `fused` itself — the deployment-honest protocol. Default
    (None) reproduces the paper's transductive fit."""
    specs = specs or default_specs(fused.shape[1])
    fit_src = fit_emb if fit_emb is not None else fused
    reducers = {}
    do_1n = labels is not None
    if do_1n and gallery_idx is None:          # random split unless caller gave a predefined one
        gallery_idx, probe_idx = E.gallery_probe_split(labels, seed=seed)
    g_idx, p_idx = gallery_idx, probe_idx
    rows = []
    for kind, k, prec in specs:
        reducer = None
        if kind:
            reducer = reducers.setdefault((kind, k), _fit_reducer(kind, k, fit_src))
        comp, bits = T.compress(fused, method=prec, dim_reducer=reducer)
        row = {"reducer": kind or "none", "dim": k or fused.shape[1],
               "precision": prec, "bits": bits}
        if do_1n:
            gemb, glab = _make_gallery(comp, fused, labels, g_idx, prec, reducer,
                                       num_distractors, seed)
            res = idn.cmc_and_map(gemb, glab, comp[p_idx], labels[p_idx])
            row.update(rank1=res["rank-1"], rank5=res["rank-5"], mAP=res["mAP"])
        if pair_idx is not None:
            acc, std = ver.evaluate_lfw(comp, pair_idx)
            row.update(acc=acc, acc_std=std)
            row.update(ver.roc_metrics(comp, pair_idx))   # eer, tar@far columns
        rows.append(row)
    return pd.DataFrame(rows)
