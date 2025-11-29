from typing import List

import clip
import torch
from numpy._typing import NDArray


def normalize(embedding: torch.Tensor) -> torch.Tensor:
    return embedding / torch.linalg.norm(embedding)


class ClipEmbedder:

    def __init__(
        self,
        batch_size: int,
    ) -> None:
        self.model, self.preprocessor = clip.load('ViT-B/32')
        self.batch_size = batch_size

    def encode_images(
        self,
        batch_tensor: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            embeds = self.model.encode_image(batch_tensor)
        return embeds

    def embed_images(
        self,
        img_batch: List[torch.Tensor],
    ) -> List[NDArray]:
        embeddings = []
        batch_tensor = torch.cat(img_batch, dim=0)
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

    def postprocess(self, embedding: torch.Tensor) -> torch.Tensor:
        return normalize(embedding)
