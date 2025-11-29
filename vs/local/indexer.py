import pathlib
from typing import Union, List, Tuple

import torch

from numpy.typing import NDArray
from tqdm import tqdm

from vs.embedder.clip import ClipEmbedder
from vs.frames import iter_video_frames


class LocalIndexPipeline:

    def __init__(
        self,
        embedder: ClipEmbedder,
        frame_rate: float = 1.0,
    ) -> None:
        self.rate = frame_rate
        self.embedder = embedder

    def execute(self, video_files: List[str]) -> Tuple[List[NDArray], List[Tuple[str, int]]]:

        all_embeddings = []
        all_meta = []

        for video in tqdm(video_files):
            embeddings, meta = self.execute_one(video)
            all_embeddings.extend(embeddings)
            all_meta.extend(meta)

        return all_embeddings, all_meta

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
