import torch
import numpy as np
import cv2
from .base import EmbeddingExtractor

class AdaFaceExtractor(EmbeddingExtractor):
    name = "adaface"
    embedding_dim = 512

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        # Download from https://github.com/mk-minchul/AdaFace
        from adaface import net
        self.model = net.build_model("ir_101")
        sd = torch.load(checkpoint_path, map_location=device)["state_dict"]
        sd = {k.replace("model.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.model.to(device).eval()
        self.device = device

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ((img / 255.0) - 0.5) / 0.5
        img = np.transpose(img.astype(np.float32), (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> np.ndarray:
        x = self._preprocess(image)
        emb, _ = self.model(x)
        return self.l2_normalize(emb.cpu().numpy().squeeze())