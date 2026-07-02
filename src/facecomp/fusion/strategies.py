import numpy as np
from sklearn.decomposition import PCA

def concat(embeddings: list[np.ndarray]) -> np.ndarray:
    """Simple concatenation of N embeddings."""
    return np.concatenate(embeddings)

def weighted_concat(embeddings: list[np.ndarray], weights: list[float]) -> np.ndarray:
    return np.concatenate([w * e for w, e in zip(weights, embeddings)])

class PCAFusion:
    """Fit PCA on concatenated training embeddings, then project."""
    def __init__(self, n_components: int = 512):
        self.pca = PCA(n_components=n_components)

    def fit(self, X: np.ndarray):  # X: (N, total_dim)
        self.pca.fit(X)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.pca.transform(x.reshape(1, -1)).squeeze()