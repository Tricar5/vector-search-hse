import logging
import pathlib
import pickle
from typing import (
    List,
    Set,
    Tuple,
    Union,
)

from numpy.typing import NDArray
from tqdm import tqdm

from vs.embedder.clip import BaseWrapper, AudioCLIPWrapper
from vs.frames import (
    iter_video_frames,
    open_and_load_frame,
)

import cv2

logger = logging.getLogger(__name__)


def embed_images_from_one_video(
    embedder: BaseWrapper,
    video_path: Union[str, pathlib.Path],
    seconds_per_embed: float = 1,
) -> Tuple[List[NDArray], List[Tuple[str, int]]]:

    list_frames = []
    list_meta = []
    for frame, pos in iter_video_frames(video_path, rate):
        list_frames.append(embedder.preprocess_image(frame))
        list_meta.append((video_path, pos, pos+1))

    if not list_frames:
        return [], list_meta

    batch = torch.stack(list_frames)
    embeddings_tensor = embedder.process_image(batch)
    embeddings = [emb.cpu().numpy() for emb in embeddings_tensor]

    return embeddings, list_meta


def embed_audio_from_one_video(
    embedder: BaseWrapper,
    video_path: Union[str, pathlib.Path]
) -> Tuple[List[NDArray], List[Tuple[str, int]]]:
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)

    batch, list_meta = embedder.preprocess_audio(video_path)
    embeddings_tensor = embedder.process_image(batch)
    list_meta = [(video_path, item[0]*fps, item[1]*fps) for item in list_meta]
    embeddings = [emb.cpu().numpy() for emb in embeddings_tensor]

    return embeddings, list_meta


def local_index_pipe(
    video_files: List[str],
    seconds_per_embed: float,
    batch_size: int,
    index_path: str,
    metadata_path: str,
) -> None:
    # Build index classes
    embedder = AudioCLIPWrapper('cpu')

    all_embeddings = []
    all_meta = []

    for video in tqdm(video_files):
        if embedder.images:
            embeddings, meta = embed_images_from_one_video(embedder, video, seconds_per_embed)
            all_embeddings.extend(embeddings)
            all_meta.extend(meta)
        if embedder.audio:
            embeddings, meta = embed_audio_from_one_video(embedder, video)
            all_embeddings.extend(embeddings)
            all_meta.extend(meta)

    # Saving to local
    with open(index_path, 'wb') as file:
        pickle.dump(all_embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(metadata_path, 'wb') as file:
        pickle.dump(all_meta, file, protocol=pickle.HIGHEST_PROTOCOL)


def local_thumbnails(
    metadata_path: str = 'metadata.pkl',
    thumbnail_path: str = 'thumbnails.pkl',
) -> None:
    with open(metadata_path, 'rb') as handle:
        meta = pickle.load(handle)
    all_videos = sorted(set([m[0] for m in meta]))

    thumbnails_meta = {}
    for i, video in enumerate(all_videos):
        frame_number = 0 if video[-4:] not in ['.png', '.jpg', 'webp', 'jpeg'] else -1
        frame, fps = open_and_load_frame((video, frame_number))
        thumbnails_meta[video] = [i, fps]

    with open(thumbnail_path, 'wb') as handle:
        pickle.dump(thumbnails_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)
