"""1:N identification. Gallery entries with label -1 act as distractors."""
from __future__ import annotations
import numpy as np


def cmc_and_map(gallery_emb, gallery_labels, probe_emb, probe_labels,
                ranks=(1, 5, 10), chunk=512):
    cmc = {k: 0 for k in ranks}
    aps = []
    for s in range(0, probe_emb.shape[0], chunk):
        sims = probe_emb[s:s + chunk] @ gallery_emb.T          # cosine (L2-normed)
        order = np.argsort(-sims, axis=1)
        for r in range(order.shape[0]):
            gl = gallery_labels[order[r]]
            true = probe_labels[s + r]
            hit = (gl == true)
            pos = np.where(hit)[0]
            if pos.size == 0:
                continue                                        # no enrolled match
            first = pos[0]
            for k in ranks:
                if first < k:
                    cmc[k] += 1
            cum = np.cumsum(hit)
            prec = cum / (np.arange(len(hit)) + 1)
            aps.append(float((prec * hit).sum() / hit.sum()))
    n = len(aps)
    out = {f"rank-{k}": (cmc[k] / n if n else float("nan")) for k in ranks}
    out["mAP"] = float(np.mean(aps)) if aps else float("nan")
    out["n_probes"] = n
    return out
