import numpy as np
from sklearn.model_selection import KFold

def cosine_scores(emb, pair_idx):
    a = emb[pair_idx[:, 0]]
    b = emb[pair_idx[:, 1]]
    # embeddings already L2-normalized in extract_embeddings.py
    return np.sum(a * b, axis=1)

def evaluate_lfw(emb, pair_idx):
    """10-fold LFW protocol: pick threshold on 9 folds, evaluate on 1."""
    scores = cosine_scores(emb, pair_idx)
    labels = pair_idx[:, 2]

    accs = []
    kf = KFold(n_splits=10, shuffle=False)
    thr_grid = np.arange(-1.0, 1.0, 0.005)

    for train_idx, test_idx in kf.split(scores):
        # find best threshold on train
        train_scores, train_labels = scores[train_idx], labels[train_idx]
        best_acc, best_thr = 0.0, 0.0
        for thr in thr_grid:
            acc = ((train_scores > thr) == train_labels).mean()
            if acc > best_acc:
                best_acc, best_thr = acc, thr
        # eval on test
        test_acc = ((scores[test_idx] > best_thr) == labels[test_idx]).mean()
        accs.append(test_acc)
    return float(np.mean(accs)), float(np.std(accs))