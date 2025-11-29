import pickle
from typing import Any, List, Tuple, Dict
import clip
from numpy.typing import NDArray
import numpy as np

import torch


def brute_force_query_torch(X, x, certainty_threshold):
    sims = (x @ X.t()).squeeze(0)  # shape: [N]

    # Фильтрация по порогу 0.2
    mask = sims >= certainty_threshold
    filtered_indices = torch.nonzero(mask).squeeze(1)  # индексы в X
    filtered_sims = sims[filtered_indices]

    # Сортировка по убыванию
    sorted_sims, order = torch.sort(filtered_sims, descending=True)
    sorted_indices = filtered_indices[order]

    return sorted_indices, sorted_sims.float()


class LocalSearchEngine:

    def __init__(
        self,
        index: Any,
        meta: Any,
        thumbnails_meta: Any,
        device: str,
    ):
        self.model, self.preprocessor = clip.load('ViT-B/32', device=device)
        self.dataset = torch.tensor(np.array(index))
        self.thumbnails_meta = thumbnails_meta
        self.all_videos = sorted(set([m[0] for m in meta]))
        self.video_to_int = {v: i for i, v in enumerate(self.all_videos)}
        self.int_to_video = {i: v for v, i in self.video_to_int.items()}
        self.meta_video_ids = torch.tensor(
            [self.video_to_int[m[0]] for m in meta],
            device='cpu', dtype=torch.int32)
        self.meta_frame_nums = torch.tensor([m[1] for m in meta],
                                            device='cpu', dtype=torch.int32)

    def encode_text(
        self,
        text: str
    ) -> torch.Tensor:
        with torch.no_grad():
            data = self.model.encode_text(clip.tokenize([text]))
        data = torch.sign(data) * torch.pow(torch.abs(data), 0.25)
        data /= torch.linalg.norm(data)
        return data

    def encode_image(
        self,
        file: NDArray,
    ) -> torch.Tensor:
        with torch.no_grad():
            data = self.model.encode_image(self.preprocessor(file).unsqueeze(0))
        data /= torch.linalg.norm(data)

        return data

    def query_videos_by_tensor(
        self,
        x: torch.Tensor,
        frame_threshold: float,
        percentile: float,
        video_threshold: float,
    ) -> Tuple[List[Tuple[str, str, int]], Dict[str, Tuple[int, int, float]]]:
        idxs, certs = brute_force_query_torch(self.dataset, x, frame_threshold)
        certs = certs.cpu()
        video_idxs = self.meta_video_ids[idxs]
        video_frames = self.meta_frame_nums[idxs]

        video_descriptions = []
        used_videos = {}

        vals, order = torch.sort(video_idxs)
        targets = torch.tensor([self.video_to_int[v] for v in self.all_videos],
                               device=video_idxs.device)
        order = order.cpu()
        left = torch.bucketize(targets, vals, right=False).cpu()
        right = torch.bucketize(targets, vals, right=True).cpu()

        for i, video in enumerate(self.all_videos):
            if left[i] == right[i]:
                continue
            args = order[left[i]:right[i]]
            cert_ = certs[order[left[i] + int((right[i] - left[i]) * (1 - percentile))]]
            if cert_ < video_threshold:
                continue
            subset = video_frames[args]
            start_ = torch.min(subset)
            end_ = torch.max(subset)
            max_frame = subset[0]
            used_videos[video] = (start_.item(), end_.item(), cert_.item())
            frame_request = f'/image?video={self.video_to_int[video]}&frame_number={max_frame}'
            video_descriptions.append((video, frame_request, self.thumbnails_meta[video][1]))

        video_descriptions = sorted(video_descriptions, key=lambda x: used_videos[x[0]][2], reverse=True)[:100]

        return video_descriptions, used_videos


def load_search_index(
    index_path: str,
    metadata_path: str,
    thumbnail_path: str,
    device: str = 'cpu',
) -> LocalSearchEngine:
    with open(index_path, 'rb') as handle:
        dataset = pickle.load(handle)

    with open(metadata_path, 'rb') as handle:
        meta = pickle.load(handle)

    with open(thumbnail_path, 'rb') as handle:
        thumbnails_meta = pickle.load(handle)

    return LocalSearchEngine(dataset, meta, thumbnails_meta, device=device)
