"""
Data loading and gallery/probe construction.

Canonical inputs (decoupled from how you extract embeddings, so your existing
extract_embeddings.py can feed this directly):

  1) An .npz with one (N, D) float array per model key plus an int label array:
       np.savez("emb.npz",
                arcface=A, magface=M, adaface=Ad,   # (N, 512) each
                labels=ids,                          # (N,) int identity id
                image_paths=paths)                   # optional (N,) str
     Build the fused representation by concatenating chosen models + renormalize.

  2) Or per-model .npy files + a meta.csv with columns image_id,identity_id.

Gallery/probe split (closed-set identification):
  - each identity with >=2 samples contributes 1 gallery template + the rest as
    probes; identities with exactly 1 sample become pure gallery distractors.
  - extra distractors can be appended to scale the gallery to 1e5..1e6.
"""
import numpy as np


def load_npz(path, model_keys, label_key="labels", normalize=True):
    """Return (X_fused, labels). model_keys: str or list[str] (concatenated)."""
    d = np.load(path, allow_pickle=True)
    if isinstance(model_keys, str):
        model_keys = [model_keys]
    mats = []
    for k in model_keys:
        m = np.asarray(d[k], dtype=np.float32)
        if normalize:
            m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
        mats.append(m)
    X = np.concatenate(mats, axis=1) if len(mats) > 1 else mats[0]
    if normalize:                                    # renormalize the fused vector
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    labels = np.asarray(d[label_key]).astype(np.int64)
    return X.astype(np.float32), labels


def build_gallery_probe(labels, seed=0, max_probes_per_id=None):
    """Closed-set split. Returns dict with gallery/probe/train index arrays."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    gallery_idx, probe_idx, distractor_idx = [], [], []
    for uid in np.unique(labels):
        members = np.where(labels == uid)[0]
        rng.shuffle(members)
        if len(members) == 1:
            distractor_idx.append(members[0])         # gallery-only distractor
        else:
            gallery_idx.append(members[0])
            rest = members[1:]
            if max_probes_per_id:
                rest = rest[:max_probes_per_id]
            probe_idx.extend(rest.tolist())
    gallery_idx = np.array(gallery_idx + distractor_idx, dtype=np.int64)
    probe_idx = np.array(probe_idx, dtype=np.int64)
    return {"gallery_idx": gallery_idx, "probe_idx": probe_idx}


def make_synthetic(n_ids=400, imgs_per_id=4, dim=512, noise=0.7,
                   n_distractor=0, seed=0):
    """Clustered unit-sphere embeddings for the smoke test (no data needed).

    `noise` is the intra-class spread RELATIVE to the unit-norm identity center
    (per-dim std is noise/sqrt(dim)); noise~0.7 gives realistic intra-class
    cosine ~0.7 and near-orthogonal impostors.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_ids, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    sd = noise / np.sqrt(dim)
    X, y = [], []
    for i in range(n_ids):
        v = centers[i] + sd * rng.standard_normal((imgs_per_id, dim))
        X.append(v)
        y += [i] * imgs_per_id
    if n_distractor:
        dv = rng.standard_normal((n_distractor, dim))
        X.append(dv)
        y += [-(j + 1) for j in range(n_distractor)]  # unique negative ids
    X = np.concatenate(X).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X, np.array(y, dtype=np.int64)


def append_distractors(gallery_X, gallery_ids, distractor_X):
    """Append external distractor embeddings (id = -1) to scale the gallery."""
    d = np.asarray(distractor_X, dtype=np.float32)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    gX = np.concatenate([gallery_X, d], axis=0)
    gids = np.concatenate([gallery_ids, -np.ones(len(d), dtype=np.int64)])
    return gX, gids
