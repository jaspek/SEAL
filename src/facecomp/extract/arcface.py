import insightface
import numpy as np
from .base import EmbeddingExtractor

class ArcFaceExtractor(EmbeddingExtractor):
    name = "arcface"
    embedding_dim = 512

    def __init__(self, model_name="buffalo_l"):
        self.app = insightface.app.FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=(112, 112))

    def extract(self, image: np.ndarray) -> np.ndarray:
        faces = self.app.get(image)
        if not faces:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        # take largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return self.l2_normalize(face.embedding)