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


def openset_metrics(gallery_emb, gallery_labels, probe_emb, probe_labels,
                    fpir_targets=(0.01, 0.1), chunk=512):
    """Open-set 1:N (the watchlist metric the closed-set cmc_and_map cannot
    express, since it silently skips probes with no gallery mate).

    Probes whose identity is NOT enrolled in the gallery are known non-mates.
    For a decision threshold t on the top-1 similarity:
      FPIR = fraction of non-mate probes with top-1 >= t          (false alarm)
      FNIR = fraction of mate probes that are wrong-at-rank-1 OR top-1 < t (miss)
    Returns FNIR at each requested FPIR, plus counts."""
    enrolled = set(int(x) for x in np.unique(gallery_labels) if x >= 0)
    top1_score = np.empty(probe_emb.shape[0])
    top1_correct = np.zeros(probe_emb.shape[0], dtype=bool)
    for s in range(0, probe_emb.shape[0], chunk):
        sims = probe_emb[s:s + chunk] @ gallery_emb.T
        j = np.argmax(sims, axis=1)
        for r in range(len(j)):
            top1_score[s + r] = sims[r, j[r]]
            top1_correct[s + r] = (gallery_labels[j[r]] == probe_labels[s + r])
    is_mate = np.array([int(l) in enrolled for l in probe_labels])
    nonmate_scores = top1_score[~is_mate]
    out = {"n_mate": int(is_mate.sum()), "n_nonmate": int((~is_mate).sum())}
    for fpir in fpir_targets:
        if nonmate_scores.size == 0 or is_mate.sum() == 0:
            out[f"fnir@fpir{fpir:g}"] = float("nan")
            continue
        t = float(np.quantile(nonmate_scores, 1.0 - fpir))     # threshold giving this FPIR
        miss = (~top1_correct[is_mate]) | (top1_score[is_mate] < t)
        out[f"fnir@fpir{fpir:g}"] = round(float(miss.mean()), 5)
    return out
