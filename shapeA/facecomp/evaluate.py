"""
Evaluation: 1:N identification (CMC / Rank-k / mAP) and 1:1 verification
(EER / TAR@FAR). All numpy; faiss used only to make large-gallery search fast.

Conventions
-----------
- gallery_ids / probe_ids: integer identity labels. A gallery item with id == -1
  is a pure distractor (no probe will ever match it).
- "decode -> L2-normalize -> inner product" is the single ranking metric for
  every compressor, so rows of the rate-distortion table differ only in bits.
"""
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:                                    # pragma: no cover
    _HAS_FAISS = False


def _normalize(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def topk_search(gallery_vecs, probe_vecs, k):
    """Return (idx, score) of the top-k gallery items per probe by cosine."""
    g = _normalize(gallery_vecs)
    p = _normalize(probe_vecs)
    k = min(k, g.shape[0])
    if _HAS_FAISS:
        index = faiss.IndexFlatIP(g.shape[1])
        index.add(np.ascontiguousarray(g))
        scores, idx = index.search(np.ascontiguousarray(p), k)
        return idx, scores
    # numpy fallback, probe-chunked to bound memory
    out_idx = np.empty((p.shape[0], k), dtype=np.int64)
    out_scr = np.empty((p.shape[0], k), dtype=np.float32)
    chunk = max(1, 4_000_000 // max(1, g.shape[0]))
    for s in range(0, p.shape[0], chunk):
        sims = p[s:s + chunk] @ g.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        row = np.arange(part.shape[0])[:, None]
        order = np.argsort(-sims[row, part], axis=1)
        out_idx[s:s + chunk] = part[row, order]
        out_scr[s:s + chunk] = sims[row, order]
    return out_idx, out_scr


# --------------------------------------------------------------------------- #
# 1:N metrics
# --------------------------------------------------------------------------- #
def identification_metrics(gallery_vecs, gallery_ids, probe_vecs, probe_ids,
                           ranks=(1, 5, 10), topk=None):
    gallery_ids = np.asarray(gallery_ids)
    probe_ids = np.asarray(probe_ids)
    topk = topk or max(max(ranks), 50)
    idx, _ = topk_search(gallery_vecs, probe_vecs, topk)
    retrieved = gallery_ids[idx]                      # (P, topk)
    match = retrieved == probe_ids[:, None]           # (P, topk) bool

    # CMC / Rank-k: a probe is "hit at r" if a genuine mate is in the top r
    cmc = {r: float(match[:, :r].any(axis=1).mean()) for r in ranks}

    # mAP (supports >1 genuine mate per probe)
    n_genuine = np.array([(gallery_ids == pid).sum() for pid in probe_ids])
    aps = []
    for i in range(len(probe_ids)):
        rel = match[i]
        if n_genuine[i] == 0:
            continue
        hits = np.cumsum(rel)
        prec = hits / (np.arange(len(rel)) + 1)
        ap = (prec * rel).sum() / n_genuine[i]
        aps.append(ap)
    mAP = float(np.mean(aps)) if aps else float("nan")
    return {"rank1": cmc.get(1), "rank5": cmc.get(5),
            "rank10": cmc.get(10), "cmc": cmc, "map": mAP}


# --------------------------------------------------------------------------- #
# 1:1 metrics
# --------------------------------------------------------------------------- #
def _sample_pairs(labels, n_pairs, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="stable")
    sl = labels[order]
    bounds = np.searchsorted(sl, np.unique(sl), side="left")
    groups = np.split(order, bounds[1:])
    groups = [g for g in groups if len(g) >= 2]

    gen = []
    for _ in range(n_pairs):
        g = groups[rng.integers(len(groups))]
        a, b = rng.choice(len(g), 2, replace=False)
        gen.append((g[a], g[b]))
    imp = []
    n = len(labels)
    while len(imp) < n_pairs:
        a, b = rng.integers(n), rng.integers(n)
        if labels[a] != labels[b]:
            imp.append((a, b))
    return np.array(gen), np.array(imp)


def verification_metrics(approx_vecs, labels, n_pairs=20_000,
                         fars=(1e-2, 1e-3), seed=0):
    v = _normalize(approx_vecs)
    gen, imp = _sample_pairs(labels, n_pairs, seed)
    sg = (v[gen[:, 0]] * v[gen[:, 1]]).sum(1)
    si = (v[imp[:, 0]] * v[imp[:, 1]]).sum(1)

    # EER
    thr = np.unique(np.concatenate([sg, si]))
    far = np.array([(si >= t).mean() for t in thr])
    frr = np.array([(sg < t).mean() for t in thr])
    eer = float((far[np.argmin(np.abs(far - frr))] +
                 frr[np.argmin(np.abs(far - frr))]) / 2)

    out = {"eer": eer}
    si_sorted = np.sort(si)
    for f in fars:
        # threshold whose impostor tail mass == target FAR
        t = si_sorted[min(len(si_sorted) - 1,
                          int(np.ceil((1 - f) * len(si_sorted))) - 1)]
        out[f"tar@far={f:g}"] = float((sg >= t).mean())
    return out


# --------------------------------------------------------------------------- #
# Convenience: full evaluation of one compressor
# --------------------------------------------------------------------------- #
def evaluate_compressor(comp, gallery_X, gallery_ids, probe_X, probe_ids,
                        all_X=None, all_ids=None, ranks=(1, 5, 10),
                        n_pairs=20_000):
    g = comp.approx_vectors(gallery_X)
    p = comp.approx_vectors(probe_X)
    res = identification_metrics(g, gallery_ids, p, probe_ids, ranks=ranks)
    if all_X is not None:
        va = comp.approx_vectors(all_X)
        res.update(verification_metrics(va, all_ids, n_pairs=n_pairs))
    res["bits_per_vec"] = int(comp.bits)
    res["bytes_per_vec"] = comp.bits / 8.0
    return res
