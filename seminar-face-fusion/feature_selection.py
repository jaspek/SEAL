"""
Feature selection over the 1536-d concatenated (ArcFace + MagFace + AdaFace) embedding.

For each FS method:
  - 10-fold CV; selection done on train fold only (no leakage)
  - For each k in K_VALUES:
      - select top-k dims of the concat embedding
      - subset embeddings, L2-renormalize
      - tune cosine threshold on train fold
      - evaluate accuracy on test fold
  - Report mean ± std across folds
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

import config as C


# --- FS scoring functions: each takes (X, y), returns ranking (descending importance) ---
def fs_variance(X, y):
    return np.argsort(-X.var(axis=0))

def fs_anova(X, y):
    f, _ = f_classif(X, y)
    return np.argsort(-np.nan_to_num(f, nan=-np.inf))

def fs_l1lr(X, y):
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=300)
    lr.fit(X, y)
    return np.argsort(-np.abs(lr.coef_[0]))

def fs_rf(X, y):
    rf = RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=0)
    rf.fit(X, y)
    return np.argsort(-rf.feature_importances_)


# --- Helpers ---
def cosine_with_dims(emb, pair_idx, dims):
    sub = emb[:, dims]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-10)
    return np.sum(sub[pair_idx[:, 0]] * sub[pair_idx[:, 1]], axis=1)

def best_thr_acc(scores, labels, train_idx, test_idx):
    best_acc, best_thr = 0, 0
    for thr in np.arange(-1, 1, 0.005):
        acc = ((scores[train_idx] > thr) == labels[train_idx]).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return ((scores[test_idx] > best_thr) == labels[test_idx]).mean()


# --- Evaluation drivers ---
def evaluate_fs(method_fn, name, emb, pair_idx, k_values):
    a = emb[pair_idx[:, 0]]; b = emb[pair_idx[:, 1]]
    pair_X = a * b                  # per-dim product = cosine contribution
    pair_y = pair_idx[:, 2]

    results = {k: [] for k in k_values}
    for tr, te in KFold(n_splits=10, shuffle=False).split(pair_X):
        ranking = method_fn(pair_X[tr], pair_y[tr])
        for k in k_values:
            scores = cosine_with_dims(emb, pair_idx, ranking[:k])
            results[k].append(best_thr_acc(scores, pair_y, tr, te))

    print(f"\n=== {name} ===")
    for k in k_values:
        a, s = np.mean(results[k]), np.std(results[k])
        print(f"  k={k:5d}:  {a*100:.2f}% ± {s*100:.2f}%")
    return results

def evaluate_pca(emb, pair_idx, k_values):
    pair_y = pair_idx[:, 2]
    results = {k: [] for k in k_values}
    for tr, te in KFold(n_splits=10, shuffle=False).split(pair_idx):
        train_img_idxs = np.unique(np.concatenate([pair_idx[tr, 0], pair_idx[tr, 1]]))
        train_emb = emb[train_img_idxs]
        for k in k_values:
            pca = PCA(n_components=k).fit(train_emb)
            proj = pca.transform(emb)
            proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-10)
            scores = np.sum(proj[pair_idx[:, 0]] * proj[pair_idx[:, 1]], axis=1)
            results[k].append(best_thr_acc(scores, pair_y, tr, te))

    print(f"\n=== PCA (projection baseline, not strict feature selection) ===")
    for k in k_values:
        a, s = np.mean(results[k]), np.std(results[k])
        print(f"  k={k:5d}:  {a*100:.2f}% ± {s*100:.2f}%")
    return results


# --- Main ---
def main():
    print(f"Loading {C.EMBEDDINGS_FILE}...")
    data = np.load(C.EMBEDDINGS_FILE, allow_pickle=True)
    arc, mag, ada = data["arcface"], data["magface"], data["adaface"]
    pair_idx = data["pair_idx"]

    concat = np.concatenate([arc, mag, ada], axis=1)
    concat = concat / (np.linalg.norm(concat, axis=1, keepdims=True) + 1e-10)
    print(f"  concat dim: {concat.shape[1]}, pairs: {len(pair_idx)}")

    K_VALUES = [64, 128, 256, 512, 1024, 1536]

    methods = [
        (fs_variance, "Variance threshold"),
        (fs_anova,    "ANOVA F-score"),
        (fs_l1lr,     "L1 Logistic Regression"),
        (fs_rf,       "Random Forest importance"),
    ]

    all_results = {name: evaluate_fs(fn, name, concat, pair_idx, K_VALUES) for fn, name in methods}
    all_results["PCA"] = evaluate_pca(concat, pair_idx, K_VALUES)

    # Summary
    print("\n\n=== Summary Table (mean accuracy %) ===")
    header = f"  {'Method':25s} | " + " | ".join(f"k={k:5d}" for k in K_VALUES)
    print(header); print("-" * len(header))
    for name, res in all_results.items():
        row = f"  {name:25s} | " + " | ".join(f"{np.mean(res[k])*100:6.2f}" for k in K_VALUES)
        print(row)


if __name__ == "__main__":
    main()