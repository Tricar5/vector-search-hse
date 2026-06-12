from __future__ import annotations

import pickle
from collections import Counter
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from vs.embedder.clip import (
    AudioCLIPWrapper,
    BaseWrapper,
    CLIPWrapper,
)
from vs.frames import open_and_load_frame


class VideoDescription:
    """Lightweight result descriptor — no pydantic dep in vs package."""

    __slots__ = (
        'name', 'path', 'video_id', 'frame_num', 'frame_num_end',
        'fps', 'start_pos', 'end_pos', 'score',
    )

    def __init__(
        self,
        name: str,
        path: str,
        video_id: int,
        frame_num: int,
        frame_num_end: int,
        fps: int,
        start_pos: float,
        end_pos: float,
        score: float,
    ) -> None:
        self.name = name
        self.path = path
        self.video_id = video_id
        self.frame_num = frame_num
        self.frame_num_end = frame_num_end
        self.fps = fps
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.score = score

    def replace(self, **kwargs: Any) -> 'VideoDescription':
        fields = {s: getattr(self, s) for s in self.__slots__}
        fields.update(kwargs)
        return VideoDescription(**fields)

    def __repr__(self) -> str:
        return f'VideoDescription(name={self.name!r}, score={self.score:.4f})'


def _brute_force_query(
    X: torch.Tensor,
    x: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sims = (x @ X.t()).squeeze(0)
    mask = sims >= threshold
    indices = torch.nonzero(mask).squeeze(1)
    sims_filtered = sims[indices]
    sorted_sims, order = torch.sort(sims_filtered, descending=True)
    return indices[order].cpu(), sorted_sims.float().cpu()


class LocalSearchEngine:
    """
    Canonical video search engine backed by a CLIP/AudioCLIP index.

    Supports text, image and audio queries.
    An optional reranker (sklearn-compatible predict_proba) can be injected.
    """

    def __init__(
        self,
        index: list[NDArray[np.floating[Any]]],
        meta: list[Any],
        thumbnails_meta: dict[str, Any],
        model: BaseWrapper,
        reranker: Any | None = None,
    ) -> None:
        self._model = model
        self._reranker = reranker
        self.dataset = torch.tensor(np.array(index))
        self.thumbnails_meta = thumbnails_meta
        self.all_videos: list[str] = sorted(set(m[0] for m in meta))
        self._video_to_int: dict[str, int] = {v: i for i, v in enumerate(self.all_videos)}
        self._int_to_video: dict[int, str] = {i: v for v, i in self._video_to_int.items()}
        self._meta_video_ids = torch.tensor(
            [self._video_to_int[m[0]] for m in meta], dtype=torch.int32,
        )
        self._meta_frame_starts = torch.tensor(
            [m[1] for m in meta], dtype=torch.int32,
        )
        self._meta_frame_ends = torch.tensor(
            [m[2] if len(m) > 2 else m[1] for m in meta], dtype=torch.int32,
        )
        self._meta_total_frames: Counter[str] = Counter(m[0] for m in meta)

    def search_by_text(
        self,
        text: str,
        frame_threshold: float = 0.26,
        video_threshold: float = 0.5,
        percentile: float = 0.8,
    ) -> list[VideoDescription]:
        if not self._model.text:
            raise NotImplementedError('Loaded model does not support text queries')
        batch = self._model.preprocess_text(text)
        x = self._model.process_text(batch)[0:1]
        return self._query(x, frame_threshold, video_threshold, percentile)

    def search_by_image(
        self,
        image: Any,
        frame_threshold: float = 0.26,
        video_threshold: float = 0.5,
        percentile: float = 0.8,
    ) -> list[VideoDescription]:
        if not self._model.images:
            raise NotImplementedError('Loaded model does not support image queries')
        batch = self._model.preprocess_image(image)
        x = self._model.process_image(batch)[0:1]
        # power-transform for better image retrieval
        x = torch.sign(x) * torch.pow(torch.abs(x), 0.25)
        x /= torch.linalg.norm(x)
        return self._query(x, frame_threshold, video_threshold, percentile)

    def search_by_audio(
        self,
        audio_path: str,
        frame_threshold: float = 0.8,
        video_threshold: float = 0.01,
        percentile: float = 0.9,
    ) -> list[VideoDescription]:
        if not self._model.audio:
            raise NotImplementedError('Loaded model does not support audio queries')
        batch, _ = self._model.preprocess_audio(audio_path)
        x = self._model.process_audio(batch)[0:1]
        return self._query(x, frame_threshold, video_threshold, percentile)

    def _query(
        self,
        x: torch.Tensor,
        frame_threshold: float,
        video_threshold: float,
        percentile: float,
    ) -> list[VideoDescription]:
        indices, certs = _brute_force_query(self.dataset, x, frame_threshold)
        if len(indices) == 0:
            return []

        video_idxs = self._meta_video_ids[indices]
        frame_starts = self._meta_frame_starts[indices]
        frame_ends = self._meta_frame_ends[indices]

        vals, order = torch.sort(video_idxs)
        targets = torch.tensor([self._video_to_int[v] for v in self.all_videos])
        left = torch.bucketize(targets, vals, right=False)
        right = torch.bucketize(targets, vals, right=True)

        lengths = right - left
        valid = lengths > 0
        if not torch.any(valid):
            return []

        perc_offsets = torch.clamp((lengths.float() * (1 - percentile)).long(), min=0)
        perc_idxs = (left + perc_offsets)[valid]
        certs_per_video = certs[order[perc_idxs]]

        passed = certs_per_video >= video_threshold
        if not torch.any(passed):
            return []

        final_video_idxs = torch.nonzero(valid).squeeze(1)[passed]
        final_certs = certs_per_video[passed]
        starts_sorted = frame_starts[order]
        ends_sorted = frame_ends[order]

        videos: list[VideoDescription] = []
        certs_map: dict[str, torch.Tensor] = {}

        for i, vid_idx in enumerate(final_video_idxs.tolist()):
            l, r = left[vid_idx].item(), right[vid_idx].item()
            subset_s = starts_sorted[l:r]
            subset_e = ends_sorted[l:r]
            video_path = self.all_videos[vid_idx]

            certs_map[video_path] = certs[order[l:r]]

            videos.append(VideoDescription(
                name=video_path.split('/')[-1],
                path=video_path,
                video_id=self._video_to_int[video_path],
                frame_num=int(subset_s[0].item()),
                frame_num_end=int(subset_e[0].item()),
                fps=self.thumbnails_meta[video_path][1],
                start_pos=float(subset_s.min().item()),
                end_pos=float(subset_e.max().item()),
                score=float(final_certs[i].item()),
            ))

        videos.sort(key=lambda v: v.score, reverse=True)

        if self._reranker is not None:
            videos = self._reranker.rerank(videos, certs_map, self._meta_total_frames)

        return videos[:25]

    @classmethod
    def from_pickle(
        cls,
        index_path: str,
        metadata_path: str,
        thumbnail_path: str,
        device: str = 'cpu',
        model: BaseWrapper | None = None,
        reranker: Any | None = None,
    ) -> 'LocalSearchEngine':
        with open(index_path, 'rb') as fh:
            index = pickle.load(fh)
        with open(metadata_path, 'rb') as fh:
            meta = pickle.load(fh)
        with open(thumbnail_path, 'rb') as fh:
            thumbnails_meta = pickle.load(fh)

        if model is None:
            model = CLIPWrapper(device=device)

        return cls(index, meta, thumbnails_meta, model=model, reranker=reranker)
