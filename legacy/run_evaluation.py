"""
Load cached LFW embeddings and evaluate single models + fusion strategies.
Run extract_embeddings.py first.
"""
import numpy as np
from sklearn.decomposition import PCA

import config as C
from evaluation.verification import evaluate_lfw

def main():
    print(f"Loading {C.EMBEDDINGS_FILE}...")
    data = np.load(C.EMBEDDINGS_FILE, allow_pickle=True)
    arc, mag, ada = data["arcface"], data["magface"], data["adaface"]
    pair_idx = data["pair_idx"]
    print(f"  embeddings: {arc.shape[0]} images, pairs: {len(pair_idx)}")

    print("\n=== Single-model baselines ===")
    for name, emb in [("ArcFace", arc), ("MagFace", mag), ("AdaFace", ada)]:
        acc, std = evaluate_lfw(emb, pair_idx)
        print(f"  {name:8s} ({emb.shape[1]}d):  {acc*100:.2f}% ± {std*100:.2f}%")

    print("\n=== Fusion: concatenation ===")
    concat = np.concatenate([arc, mag, ada], axis=1)
    # re-normalize after concat
    concat = concat / (np.linalg.norm(concat, axis=1, keepdims=True) + 1e-10)
    acc, std = evaluate_lfw(concat, pair_idx)
    print(f"  concat   ({concat.shape[1]}d):  {acc*100:.2f}% ± {std*100:.2f}%")

    print("\n=== Fusion: concat + PCA ===")
    for d_target in (512, 256, 128):
        pca = PCA(n_components=d_target).fit(concat)
        emb = pca.transform(concat)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        acc, std = evaluate_lfw(emb, pair_idx)
        print(f"  PCA->{d_target:4d}d:  {acc*100:.2f}% ± {std*100:.2f}%")

    print("\n=== Fusion: score-level mean ===")
    s_arc = np.sum(arc[pair_idx[:, 0]] * arc[pair_idx[:, 1]], axis=1)
    s_mag = np.sum(mag[pair_idx[:, 0]] * mag[pair_idx[:, 1]], axis=1)
    s_ada = np.sum(ada[pair_idx[:, 0]] * ada[pair_idx[:, 1]], axis=1)
    avg = (s_arc + s_mag + s_ada) / 3.0
    # use evaluate_lfw-style threshold tuning on avg directly
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, shuffle=False)
    accs = []
    for tr, te in kf.split(avg):
        best_acc, best_thr = 0, 0
        for thr in np.arange(-1, 1, 0.005):
            a = ((avg[tr] > thr) == pair_idx[tr, 2]).mean()
            if a > best_acc: best_acc, best_thr = a, thr
        accs.append(((avg[te] > best_thr) == pair_idx[te, 2]).mean())
    print(f"  score-mean:  {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")

if __name__ == "__main__":
    main()