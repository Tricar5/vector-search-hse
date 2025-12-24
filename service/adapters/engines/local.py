# flake8: noqa: WPS111, WPS221, WPS432
import pickle
from typing import Any

import clip
import numpy as np
import torch
from numpy.typing import NDArray

from service.adapters.engines.base import Engine
from service.adapters.engines.schemas import (
    LocalEngineConfig,
    UsedVideo,
)
from service.adapters.files import load_yml_config
from service.domain.videos.schemas import VideoDescription
from service.settings import AppSettings


def brute_force_query_torch(
    X: torch.Tensor,
    x: torch.Tensor,
    certainty_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sims = (x @ X.t()).squeeze(0)  # shape: [N]

    mask = sims >= certainty_threshold
    filtered_indices = torch.nonzero(mask).squeeze(1)  # индексы в X
    filtered_sims = sims[filtered_indices]

    # Сортировка по убыванию
    sorted_sims, order = torch.sort(filtered_sims, descending=True)
    sorted_indices = filtered_indices[order]

    return sorted_indices.cpu(), sorted_sims.float().cpu()


class LocalSearchEngine(Engine):

    def __init__(
        self,
        config: LocalEngineConfig,
        index: list[NDArray[np.floating[Any]]],
        meta: list[Any],
        thumbnails_meta: dict[str, Any],
    ) -> None:
        self.model, self.preprocessor = clip.load(
            'ViT-B/32',
            device=config.device,
        )
        self.dataset = torch.tensor(np.array(index))
        self.thumbnails_meta = thumbnails_meta
        self.meta = meta
        self.device = config.device
        self.all_videos = sorted(set([m[0] for m in meta]))
        self._video_to_int = {v: i for i, v in enumerate(self.all_videos)}
        self._int_to_video = {i: v for v, i in self._video_to_int.items()}
        self._meta_video_ids = torch.tensor(
            [self._video_to_int[m[0]] for m in meta],
            device='cpu',
            dtype=torch.int32,
        )
        self._meta_frame_nums = torch.tensor(
            [m[1] for m in meta],
            device='cpu',
            dtype=torch.int32,
        )
        self.img_cfg = config.image
        self.text_cfg = config.text

    def search_videos_by_text(
        self,
        text: str,
    ) -> list[VideoDescription]:
        x = self._encode_text(text)
        return self.query_videos_by_tensor(
            x,
            self.text_cfg.frame_threshold,
            self.text_cfg.video_threshold,
            self.text_cfg.percentile,
        )

    def search_videos_by_image(
        self,
        img: NDArray[np.floating[Any]],
    ) -> list[VideoDescription]:
        x = self._encode_image(img)
        return self.query_videos_by_tensor(
            x,
            self.img_cfg.frame_threshold,
            self.img_cfg.video_threshold,
            self.img_cfg.percentile,
        )

    def query_videos_by_tensor(
        self,
        x: torch.Tensor,
        frame_threshold: float,
        video_threshold: float,
        percentile: float,
    ) -> list[VideoDescription]:
        indices, certs = brute_force_query_torch(self.dataset, x, frame_threshold)
        video_idxs = self._meta_video_ids[indices]
        video_frames = self._meta_frame_nums[indices]

        vals, order = torch.sort(video_idxs)
        targets = torch.tensor(
            [self._video_to_int[v] for v in self.all_videos]
        )
        left = torch.bucketize(targets, vals, right=False)
        right = torch.bucketize(targets, vals, right=True)
        
        lengths = right - left
        valid = lengths > 0
        if not torch.any(valid):
            return []
        
        perc_offsets = (lengths.float() * (1 - percentile)).long()
        perc_offsets = torch.clamp(perc_offsets, min=0)
    
        perc_idxs = left + perc_offsets
        perc_idxs = perc_idxs[valid]
    
        certs_per_video = certs[order[perc_idxs]]
    
        passed = certs_per_video >= video_threshold
        if not torch.any(passed):
            return []

        final_video_idxs = torch.nonzero(valid).squeeze(1)[passed]
        final_certs = certs_per_video[passed]
        frames_sorted = video_frames[order]
        
        videos = []
        used_videos = {}
        for i, vid_idx in enumerate(final_video_idxs.tolist()):
            l,r  = left[vid_idx], right[vid_idx]
            subset = frames_sorted[l:r]
            start_ = subset.min().item()
            end_ = subset.max().item()
            max_frame = subset[0]
            cert_ = final_certs[i].item()

            video = self.all_videos[vid_idx]
            used_videos[video] = UsedVideo(
                start_pos=start_, end_pos=end_, score=cert_
            )

            video_ = VideoDescription(
                name=video.split('/')[-1],
                path=video,
                video_id=self._video_to_int[video],
                frame_num=max_frame,  # type: ignore
                fps=self.thumbnails_meta[video][1],
                start_pos=start_,
                end_pos=end_,
                score=cert_,
            )
            videos.append(video_)

        videos = sorted(videos, key=lambda x: x.score, reverse=True)

        return videos

    @classmethod
    def build_engine(
        cls,
        settings: AppSettings,
    ) -> 'LocalSearchEngine':

        raw_config = load_yml_config(settings.engine_config_path)

        config = LocalEngineConfig.parse_obj(raw_config['local'])

        with open(config.index_path, 'rb') as pickled_data:
            dataset = pickle.load(pickled_data)

        with open(config.metadata_path, 'rb') as pickled_data:
            meta = pickle.load(pickled_data)

        with open(config.thumbnail_path, 'rb') as pickled_data:
            thumbnails_meta = pickle.load(pickled_data)

        return LocalSearchEngine(config, dataset, meta, thumbnails_meta)

    def _encode_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            tensor_data = self.model.encode_text(clip.tokenize([text]))
        tensor_data /= torch.linalg.norm(tensor_data)
        return tensor_data

    def _encode_image(
        self,
        file: NDArray[np.floating[Any]],
    ) -> torch.Tensor:
        with torch.no_grad():
            tensor_data = self.model.encode_image(self.preprocessor(file).unsqueeze(0))

        tensor_data = torch.sign(tensor_data) * torch.pow(torch.abs(tensor_data), 0.25)
        tensor_data /= torch.linalg.norm(tensor_data)
        return tensor_data
