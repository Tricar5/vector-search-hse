from typing import List

from typer import Typer, Option, Argument

from vs.local.pipeline import local_index_pipe, local_thumbnails
from vs.utils import find_files_by_extensions

app = Typer()


@app.command(
    'check_files_by_patterns',
    help='Проверка наличия файлов в директории',
)
def check_files_by_patterns(
    folder: str = Option('data', "--folder", "-f", help="Say hi formally."),
    extensions: str = Option('--patterns', help="Say hi formally."),
) -> None:
    extensions = extensions.split(',')
    files = find_files_by_extensions(folder, extensions)
    print(f'Found {len(files)} files in {folder}')
    print(f'File Path Example {files[0]}')


@app.command(
    'local_index_pipe',
    help='Создание локального индекса для работы локального движка',
)
def make_local_index(
    video_dir: str = Option('data/video', '--video-dir', '-v', help='Video dir'),
    extensions: str = Option('mp4,mov', '--extensions', help='File Extension'),
    frame_rate: float = Option(1.0, '--frame-rate', help='Frame Rates to slice'),
    batch_size: int = Option(128, '--batch-size', help='Batch Size'),
    index_path: str = Option('index.pkl', '--index-path', help='Path to store Index'),
    metadata_path: str = Option('metadata.pkl', '--metadata-path', help='Path to store Metadata'),
) -> None:
    extensions = extensions.split(',')
    video_files = find_files_by_extensions(video_dir, extensions)
    print(f'Found {len(video_files)} files in {video_dir}')
    local_index_pipe(
        video_files=video_files,
        frame_rate=frame_rate,
        batch_size=batch_size,
        index_path=index_path,
        metadata_path=metadata_path,
    )


@app.command(
    'local_thumbnail',
    help='Создание локальных превью',
)
def make_local_index(
    metadata_path: str = Option('metadata.pkl', '--metadata-path', help='Path to store Metadata'),
    thumbnail_path: str = Option('thumbnails.pkl', '--thumbnail-path', help='Path to store Thumbnails'),
) -> None:
    local_thumbnails(
        metadata_path=metadata_path,
        thumbnail_path=thumbnail_path,
    )


if __name__ == "__main__":
    app()
