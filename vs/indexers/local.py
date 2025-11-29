import pathlib
from typing import Union, List, Tuple, Any

import torch

from numpy.typing import NDArray
from tqdm import tqdm

from vs.frames import iter_video_frames
import clip

from vs.storage.local import LocalIndexStorage


def normalize(embedding: torch.Tensor) -> torch.Tensor:
    return embedding / torch.linalg.norm(embedding)


class Embedder:

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
            embedded = self.model.encode_image(batch)
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


class LocalIndexPipeline:

    def __init__(
        self,
        embedder: Embedder,
        storage: LocalIndexStorage,
        frame_rate: float = 1.0,
    ) -> None:
        self.rate = frame_rate
        self.embedder = embedder
        self.storage = storage

    def execute(self, video_files: List[str]) -> None:

        all_embeddings = []
        all_meta = []

        for video in tqdm(video_files):
            embeddings, meta = self.execute_one(video)
            all_embeddings.extend(embeddings)
            all_meta.append(meta)

        self.storage.save_metadata(all_meta)
        self.storage.save_index(all_meta)

    def execute_one(
        self,
        video_path: Union[str, pathlib.Path],
    ) -> Tuple[List[torch.Tensor], List[Tuple[str, int]]]:
        list_frames = []
        list_meta = []
        for frame, pos in iter_video_frames(video_path, self.rate):
            img = self.embedder.preprocess(frame)
            list_frames.append(img)
            list_meta.append((video_path, pos))

        embeddings = self.embedder.embed_images(list_frames)
        return embeddings, list_meta
