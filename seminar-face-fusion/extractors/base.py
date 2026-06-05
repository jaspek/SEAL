from abc import ABC, abstractmethod
import numpy as np

class EmbeddingExtractor(ABC):
    """All extractors return a 1D numpy array (the embedding)."""

    name: str
    embedding_dim: int

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Input: aligned face image (BGR, 112x112). Output: L2-normalized embedding."""
        ...

    @staticmethod
    def l2_normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-10)