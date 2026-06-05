import numpy as np
from sklearn.model_selection import KFold
import config as C

data = np.load(C.EMBEDDINGS_FILE, allow_pickle=True)
arc, mag, ada = data["arcface"], data["magface"], data["adaface"]
pair_idx = data["pair_idx"]
labels = pair_idx[:, 2]

def per_pair_predictions(emb, pair_idx, labels):
    """Return a boolean array: True = correct, False = wrong (using one global best threshold)."""
    scores = np.sum(emb[pair_idx[:, 0]] * emb[pair_idx[:, 1]], axis=1)
    # global best threshold (simpler than 10-fold for this diagnostic)
    best_acc, best_thr = 0, 0
    for thr in np.arange(-1, 1, 0.005):
        acc = ((scores > thr) == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    preds = (scores > best_thr).astype(int)
    return preds == labels, best_acc, best_thr

c_arc, acc_arc, _ = per_pair_predictions(arc, pair_idx, labels)
c_mag, acc_mag, _ = per_pair_predictions(mag, pair_idx, labels)
c_ada, acc_ada, _ = per_pair_predictions(ada, pair_idx, labels)

print(f"Single accuracies: ArcFace {acc_arc*100:.2f}%, MagFace {acc_mag*100:.2f}%, AdaFace {acc_ada*100:.2f}%")

w_arc, w_mag, w_ada = ~c_arc, ~c_mag, ~c_ada

print(f"\nWrong by ArcFace only:    {(w_arc & ~w_mag & ~w_ada).sum()}")
print(f"Wrong by MagFace only:    {(~w_arc & w_mag & ~w_ada).sum()}")
print(f"Wrong by AdaFace only:    {(~w_arc & ~w_mag & w_ada).sum()}")
print(f"Wrong by ArcFace+MagFace: {(w_arc & w_mag & ~w_ada).sum()}")
print(f"Wrong by ArcFace+AdaFace: {(w_arc & ~w_mag & w_ada).sum()}")
print(f"Wrong by MagFace+AdaFace: {(~w_arc & w_mag & w_ada).sum()}")
print(f"Wrong by ALL THREE:       {(w_arc & w_mag & w_ada).sum()}")

# Majority-vote oracle: a pair is correct if at least 2 of 3 models got it right
votes = c_arc.astype(int) + c_mag.astype(int) + c_ada.astype(int)
majority_correct = (votes >= 2).mean()
oracle_correct = (c_arc | c_mag | c_ada).mean()
print(f"\nMajority-vote accuracy:   {majority_correct*100:.2f}%")
print(f"Oracle (any one correct): {oracle_correct*100:.2f}%")