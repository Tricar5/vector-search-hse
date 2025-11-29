import numpy as np
from typing import List, Tuple, Generator
from fastembed import ImageEmbedding

from numpy.typing import NDArray

from vs.schemas import (
    VideoInfo,
    FrameInfo,
)


class PlainVideoFramesIndexer:
    name: str = 'IndexSingleFrame'

    def __init__(
        self,
    ) -> None:
        self._embedder = ImageEmbedding('Qdrant/clip-ViT-B-32-vision')

    def embed_video(
        self,
        video: VideoInfo,
        normalize: bool = True,
    ) -> Tuple[List[NDArray], List[FrameInfo]]:
        """Generates video frame embeddings."""
        frames_path = [frame.path for frame in video.frames]
        embeddings = self._embed(frames_path)
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings)

        return embeddings, video.frames

    def _embed(self, frames_path: List[str]) -> List[NDArray]:
        return list(self._embedder.embed(frames_path))
