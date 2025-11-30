from typing import Optional
from typer import Typer

from vs.local.pipeline import local_index_pipe, local_thumbnails

app = Typer()


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
    metadata_path: str = 'metadata.pkl',
    normalize: bool = True,
) -> None:
    local_index_pipe(
        video_dir=video_dir,
        video_pattern=video_pattern,
        frame_rate=frame_rate,
        batch_size=batch_size,
        index_path=index_path,
        metadata_path=metadata_path,
        normalize=normalize,
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
