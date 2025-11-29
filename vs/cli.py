from typing import Optional
from typer import Typer

from vs.local.pipeline import local_index_pipe, local_thumbnails
from vs._local.pipe import local_indexing_from_frames, local_frames_pipe

app = Typer()


@app.command(
    'local_frames',
    help='Создание локального индекса для работы локального движка',
)
def make_local_frames(
    video_dir: str = 'data/video',
    frames_dir: str = 'frames',
    frame_rate: float = 1.0,
    video_pattern: str = '*.MOV',
    metadata_name: Optional[str] = 'video.json',
    limit: Optional[int] = None  # usable for tests
) -> None:
    local_frames_pipe(
        video_dir=video_dir,
        frames_dir=frames_dir,
        frame_rate=frame_rate,
        video_pattern=video_pattern,
        metadata_name=metadata_name,
        limit=limit
    )


@app.command(
    'local_index_from_frames',
    help='Cоздание индекса в pickle с помощью готовых нарезанных фреймов',
)
def make_local_index_from_frames(
    video_meta: str = 'video.json',
    index_path: str = 'index.pickle',
    metadata_path: str = 'metadata.pickle',
) -> None:
    local_indexing_from_frames(
        video_meta=video_meta,
        index_path=index_path,
        metadata_path=metadata_path,
    )


@app.command(
    'local_index_pipe',
    help='Создание локального индекса для работы локального движка',
)
def make_local_index(
    video_dir: str = 'data/video',
    video_pattern: str = '*.MOV',
    frame_rate: float = 1.0,
    batch_size: int = 128,
    index_path: str = 'index.pkl',
    frames_meta_path: str = 'metadata.pkl',
    limit: Optional[int] = None
) -> None:
    local_index_pipe(
        video_dir=video_dir,
        video_pattern=video_pattern,
        frame_rate=frame_rate,
        batch_size=batch_size,
        index_path=index_path,
        metadata_path=frames_meta_path,
        limit=limit,
    )


@app.command(
    'local_thumbnail',
    help='Создание локальных превью',
)
def make_local_index(
    metadata_path: str = 'metadata.pkl',
    thumbnail_path: str = 'thumbnails.pkl',
) -> None:
    local_thumbnails(
        metadata_path=metadata_path,
        thumbnail_path=thumbnail_path,
    )


if __name__ == "__main__":
    app()
