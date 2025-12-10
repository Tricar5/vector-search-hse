import logging
import pathlib
import pickle
from typing import Union, List, Tuple, Set

from scipy.linalg._decomp_interpolative import NDArray
from tqdm import tqdm

from vs.frames import open_and_load_frame, iter_video_frames
from vs.embedder.clip import ClipEmbedder

logger = logging.getLogger(__name__)


def embed_one_video(
    embedder: ClipEmbedder,
    video_path: Union[str, pathlib.Path],
    rate: float = 1,
) -> Tuple[List[NDArray], List[Tuple[str, int]]]:
    list_frames = []
    list_meta = []
    for frame, pos in iter_video_frames(video_path, rate):
        list_frames.append(frame)
        list_meta.append((video_path, pos))

    embeddings = embedder.embed_images(list_frames)
    return embeddings, list_meta


def local_index_pipe(
    video_files: List[str],
    frame_rate: float,
    batch_size: int,
    index_path: str,
    metadata_path: str,
) -> None:
    # Build index classes
    embedder = ClipEmbedder(
        batch_size=batch_size,
    )

    all_embeddings = []
    all_meta = []

    for video in tqdm(video_files):
        embeddings, meta = embed_one_video(embedder, video, frame_rate)
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
