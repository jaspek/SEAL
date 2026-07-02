import numpy as np, csv
import config as C

data = np.load(C.EMBEDDINGS_FILE, allow_pickle=True)
arc, mag, ada = data["arcface"], data["magface"], data["adaface"]
pair_idx = data["pair_idx"]
paths = data["paths"]
labels = pair_idx[:, 2]

# Find pairs all three got wrong (using each model's optimal threshold)
def wrong_mask(emb):
    s = np.sum(emb[pair_idx[:, 0]] * emb[pair_idx[:, 1]], axis=1)
    best_acc, best_thr = 0, 0
    for thr in np.arange(-1, 1, 0.005):
        acc = ((s > thr) == labels).mean()
        if acc > best_acc: best_acc, best_thr = acc, thr
    return (s > best_thr) != labels, s

w_arc, s_arc = wrong_mask(arc)
w_mag, s_mag = wrong_mask(mag)
w_ada, s_ada = wrong_mask(ada)

all_wrong = w_arc & w_mag & w_ada
print(f"Total pairs all 3 got wrong: {all_wrong.sum()}\n")

# How many are 'same' (label=1, predicted different) vs 'different' (label=0, predicted same)?
same_wrong = all_wrong & (labels == 1)
diff_wrong = all_wrong & (labels == 0)
print(f"  Labeled 'same' but predicted different: {same_wrong.sum()}  ← model thinks they're different people")
print(f"  Labeled 'different' but predicted same: {diff_wrong.sum()}  ← model thinks they're same person")

print("\nFirst 10 universally-wrong pairs:")
for i in np.where(all_wrong)[0][:10]:
    a_idx, b_idx, lbl = pair_idx[i]
    label_name = "SAME" if lbl == 1 else "DIFF"
    p1 = paths[a_idx].split("\\")[-1]
    p2 = paths[b_idx].split("\\")[-1]
    print(f"  [{label_name}] {p1}  vs  {p2}   "
          f"(arc={s_arc[i]:+.2f} mag={s_mag[i]:+.2f} ada={s_ada[i]:+.2f})")