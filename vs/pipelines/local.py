import logging
import pathlib
import pickle
from typing import Optional
from tqdm import tqdm

import numpy as np

from vs.indexers.local import LocalIndexPipeline, Embedder
from vs.storage.local import LocalIndexStorage
from vs.indexers._local import PlainVideoFramesIndexer
from vs.storages._local import LocalVideoInfoStorage
from vs.utils import find_all_files_with_pattern
from vs.video_file import VideoFilesProcessor

logger = logging.getLogger(__name__)


def local_index_pipe(
    video_dir: str = 'data/video',
    video_pattern: str = '*.MOV',
    frame_rate: float = 1.0,
    batch_size: int = 32,
    index_path: str = 'index.pickle',
    metadata_path: str = 'metadata.pickle',
) -> None:
    video_files = find_all_files_with_pattern(
        pattern=video_pattern,
        folder=pathlib.Path(video_dir),
    )
    embedder = Embedder(batch_size=batch_size)
    storage = LocalIndexStorage(index_path=index_path, metadata_path=metadata_path)
    indexer = LocalIndexPipeline(
        frame_rate=frame_rate,
        embedder=embedder,
        storage=storage
    )
    indexer.execute(video_files)


def local_indexing_from_frames(
    video_meta: str = 'video.json',
    index_path: str = 'index.pickle',
    metadata_path: str = 'metadata.pickle',
):
    vf_s = LocalVideoInfoStorage.from_file(video_meta)
    indexer = PlainVideoFramesIndexer()

    embeddings_list = []
    frames_list = []

    for video in vf_s.videos:
        frames_embeddings, frames_meta = indexer.embed_video(video, normalize=True)
        embeddings_list.extend(frames_embeddings)
        frames_list.extend(frames_meta)

    embeddings = np.array(embeddings_list)

    with open(index_path, 'wb') as file:
        pickle.dump(embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)

    frame_meta = [(i, frame.path, frame.video_id) for i, frame in enumerate(frames_list)]
    with open(metadata_path, 'wb') as file:
        pickle.dump(frame_meta, file, protocol=pickle.HIGHEST_PROTOCOL)


def local_frames_pipe(
    video_dir: str = 'video',
    frames_dir: str = 'frames',
    frame_rate: float = 1.0,
    video_pattern: str = '*.MOV',
    metadata_name: Optional[str] = 'video.json',
    limit: Optional[int] = None  # usable for tests
) -> None:
    video_files = find_all_files_with_pattern(
        pattern=video_pattern,
        folder=pathlib.Path(video_dir),
    )
    if limit:
        video_files = video_files[:limit]

    vp = VideoFilesProcessor(
        frame_dir=pathlib.Path(frames_dir),
        frame_rate=frame_rate,
    )
    video_records = []

    for video_path in video_files:
        video = vp.process_single_video(video_path)
        video_records.append(video)

    video_storage = LocalVideoInfoStorage(video_records)

    video_storage.dump_to_file(metadata_name)
