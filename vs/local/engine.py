import pickle
from typing import Any, List, Tuple, Dict
import clip
from numpy.typing import NDArray
import numpy as np

import torch
from vs.embedder.clip import BaseWrapper, AudioCLIPWrapper


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
        index: List[NDArray],
        meta: List[Any],
        thumbnails_meta: Dict[str, Any],
        device: str,
        model: Optional[BaseWrapper]=None
    ):
        self.model = model if model else AudioCLIPWrapper(device=device)
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
        if self.model.text:
            batch = self.model.preprocess_text(text)
            data = self.model.process_text(batch)
            return data[0:1]
        else:
            raise NonImplemetedError('Представленная модель не может обрабатывать текст')

    def encode_image(
        self,
        image: NDArray,
    ) -> torch.Tensor:
        if self.model.images:
            batch = self.model.preprocess_image(image)
            data = self.model.process_image(batch)
            return data[0:1]
        else:
            raise NonImplemetedError('Представленная модель не может обрабатывать изображения')

    def encode_audio(
        self,
        audio_path: Union[str, pathlib.Path], # потому что разные модели могут потребовать разные способы загрузки аудиоданных
    ) -> torch.Tensor:
        if self.model.audio:
            batch, _ = self.model.preprocess_audio(audio_path)
            data = self.model.process_audio(batch)
            return data[0:1]
        else:
            raise NonImplemetedError('Представленная модель не может обрабатывать аудио')

    def query_videos_by_tensor(
        self,
        x: torch.Tensor,
        frame_threshold: float,
        percentile: float,
        video_threshold: float,
    ) -> Tuple[List[Tuple[str, str, int]], Dict[str, Tuple[int, int, float]]]:
        idxs, certs = brute_force_query_torch(self.dataset, x, frame_threshold)
        certs = certs.cpu()
        idxs = idxs.cpu()
        video_idxs = self.meta_video_ids[idxs]
        video_starts = self.meta_frame_starts[idxs]
        video_ends = self.meta_frame_ends[idxs]

        vals, order = torch.sort(video_idxs)
        targets = torch.tensor([video_to_int[v] for v in all_videos])
        left  = torch.bucketize(targets, vals, right=False)
        right = torch.bucketize(targets, vals, right=True)

        num_videos = len(all_videos)

        lengths = right - left
        valid = lengths > 0

        perc_offsets = (lengths.float() * (1 - percentile)).long()
        perc_offsets = torch.clamp(perc_offsets, min=0)

        perc_idxs = left + perc_offsets
        perc_idxs = perc_idxs[valid]
        certs_per_video = certs[order[perc_idxs]]
        passed = certs_per_video >= video_threshold

        final_video_idxs = torch.nonzero(valid)[passed].squeeze(1)
        final_certs = certs_per_video[passed]
        frames_starts_sorted = video_starts[order]
        frames_ends_sorted = video_ends[order]

        used_videos = {}
        video_descriptions = []

        for i, vid_idx in enumerate(final_video_idxs.tolist()):
            l,r  = left[vid_idx], right[vid_idx]
            subset_s = frames_starts_sorted[l:r]
            subset_e = frames_ends_sorted[l:r]
            start = subset_s.min()
            end = subset_e.max()
            max_frame = subset_s[0]

            video = self.all_videos[vid_idx]
            used_videos[video] = (
                start.item(),
                end.item(),
                final_certs[i].item()
            )
            frame_request = (
                f"/image?video={self.video_to_int[video]}"
                f"&frame_number={max_frame.item()}"
            )
            video_descriptions.append(
                (video, frame_request, self.thumbnails_meta[video][1])
            )

        video_descriptions = sorted(
            video_descriptions,
            key=lambda x: used_videos[x[0]][2],
            reverse=True
        )[:100]

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
