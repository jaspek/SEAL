import numpy as np
import config as C

data = np.load(C.EMBEDDINGS_FILE, allow_pickle=True)
arc = data["arcface"]
mag = data["magface"]
ada = data["adaface"]
pair_idx = data["pair_idx"]

print(f"Total embeddings: {arc.shape[0]}\n")

# How many embeddings are all-zero (= alignment failed)?
for name, emb in [("ArcFace", arc), ("MagFace", mag), ("AdaFace", ada)]:
    zeros = np.sum(np.all(emb == 0, axis=1))
    norms = np.linalg.norm(emb, axis=1)
    print(f"{name}: zero rows={zeros}, norm range=[{norms.min():.3f}, {norms.max():.3f}], mean norm={norms.mean():.3f}")

print("\n--- Cosine distributions (same vs different pairs) ---")
same = pair_idx[pair_idx[:, 2] == 1]
diff = pair_idx[pair_idx[:, 2] == 0]

for name, emb in [("ArcFace", arc), ("MagFace", mag), ("AdaFace", ada)]:
    s = np.sum(emb[same[:, 0]] * emb[same[:, 1]], axis=1)
    d = np.sum(emb[diff[:, 0]] * emb[diff[:, 1]], axis=1)
    print(f"{name}: same={s.mean():+.3f}±{s.std():.3f}   diff={d.mean():+.3f}±{d.std():.3f}   gap={s.mean()-d.mean():+.3f}")

# Print first 5 dimensions of one ArcFace embedding
print(f"\nArcFace[100] first 5 dims: {arc[100][:5]}")
print(f"AdaFace[100] first 5 dims: {ada[100][:5]}")