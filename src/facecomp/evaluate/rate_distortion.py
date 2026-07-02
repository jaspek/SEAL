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
    specs = [(None, None, p) for p in ("fp32", "fp16", "int8", "binary")]
    for k in (256, 128):
        if k < full_dim:
            specs += [("pca", k, "binary"), ("itq", k, "binary")]
    return specs


def rate_distortion(fused, labels=None, pair_idx=None, gallery_idx=None, probe_idx=None,
                    specs=None, num_distractors=0, seed=0):
    specs = specs or default_specs(fused.shape[1])
    reducers = {}
    do_1n = labels is not None
    if do_1n and gallery_idx is None:          # random split unless caller gave a predefined one
        gallery_idx, probe_idx = E.gallery_probe_split(labels, seed=seed)
    g_idx, p_idx = gallery_idx, probe_idx
    rows = []
    for kind, k, prec in specs:
        reducer = None
        if kind:
            reducer = reducers.setdefault((kind, k), _fit_reducer(kind, k, fused))
        comp, bits = T.compress(fused, method=prec, dim_reducer=reducer)
        row = {"reducer": kind or "none", "dim": k or fused.shape[1],
               "precision": prec, "bits": bits}
        if do_1n:
            gemb, glab = E.add_distractors(comp[g_idx], labels[g_idx], num_distractors, seed=seed)
            res = idn.cmc_and_map(gemb, glab, comp[p_idx], labels[p_idx])
            row.update(rank1=res["rank-1"], rank5=res["rank-5"], mAP=res["mAP"])
        if pair_idx is not None:
            acc, std = ver.evaluate_lfw(comp, pair_idx)
            row.update(acc=acc, acc_std=std)
            row.update(ver.roc_metrics(comp, pair_idx))   # eer, tar@far columns
        rows.append(row)
    return pd.DataFrame(rows)
