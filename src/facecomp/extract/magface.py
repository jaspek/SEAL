import torch
import numpy as np
import cv2
from .base import EmbeddingExtractor

class MagFaceExtractor(EmbeddingExtractor):
    name = "magface"
    embedding_dim = 512

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        # Download from https://github.com/IrvingMeng/MagFace
        # Their repo provides a builder; here is a sketch:
        from magface.models.magface import builder_inf
        self.model = builder_inf(arch="iresnet100", embedding_size=512)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.to(device).eval()
        self.device = device

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        # BGR -> RGB, normalize to [-1, 1], to tensor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        x = self._preprocess(image)
        emb = self.model(x).cpu().numpy().squeeze()
        return self.l2_normalize(emb)