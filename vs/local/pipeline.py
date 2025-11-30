import logging
import pathlib
import pickle
from typing import Optional

from vs.frames import open_and_load_frame
from vs.local.indexer import LocalIndexPipeline
from vs.embedder.clip import ClipEmbedder
from vs.utils import find_all_files_with_pattern

logger = logging.getLogger(__name__)


def local_index_pipe(
    video_dir: str,
    video_pattern: str,
    frame_rate: float,
    batch_size: int,
    index_path: str,
    metadata_path: str,
    normalize: bool = True,
) -> None:
    video_files = find_all_files_with_pattern(
        pattern=video_pattern,
        folder=pathlib.Path(video_dir),
    )

    # Build index classes
    embedder = ClipEmbedder(
        batch_size=batch_size,
        normalize=normalize,
    )
    indexer = LocalIndexPipeline(
        frame_rate=frame_rate,
        embedder=embedder,
    )

    all_embeddings, all_meta = indexer.make_index_for_videofiles(video_files)

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
