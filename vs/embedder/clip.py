from typing import List

import clip
import torch
from numpy._typing import NDArray


class ClipEmbedder:

    def __init__(
        self,
        batch_size: int,
        normalize: bool = True,
    ) -> None:
        self.model, self.preprocessor = clip.load('ViT-B/32')
        self.batch_size = batch_size
        self.normalize = normalize

    def encode_images(
        self,
        batch_tensor: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            embeds = self.model.encode_image(batch_tensor)
        return embeds

    def embed_images(
        self,
        img_batch: List[NDArray],
    ) -> List[NDArray]:
        images = [self.preprocess(img) for img in img_batch]

        embeddings = []
        batch_tensor = torch.cat(images, dim=0)
        for batch in torch.split(batch_tensor, self.batch_size, dim=0):
            embedded = self.encode_images(batch)
            for emb in embedded:
                emb = self.postprocess(emb)
                embeddings.append(emb.detach().numpy())

        return embeddings

    def preprocess(
        self,
        img: NDArray,
    ) -> None:
        return self.preprocessor(img).unsqueeze(0)

    def postprocess(
        self,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.normalize:
            emb = emb / torch.linalg.norm(emb)
        return emb
