import sys

from typer import (
    Context,
    Option,
    Typer,
)

from vs.local.pipeline import (
    local_index_pipe,
    local_thumbnails,
)
from vs.utils import find_files_by_extensions

app = Typer()
download_app = Typer(help='Скачивание весов моделей и словарей.')
app.add_typer(download_app, name='download-deps')


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
    index_path: str = Option('model/index.pkl', '--index-path', help='Path to store Index'),
    metadata_path: str = Option(
        'model/metadata.pkl', '--metadata-path', help='Path to store Metadata'
    ),
) -> None:
    extensions = extensions.split(',')
    video_files = find_files_by_extensions(video_dir, extensions)
    print(f'Found {len(video_files)} files in {video_dir}')
    local_index_pipe(
        video_files=video_files,
        seconds_per_embed=frame_rate,
        batch_size=batch_size,
        index_path=index_path,
        metadata_path=metadata_path,
    )


@app.command(
    'local_thumbnail',
    help='Создание локальных превью',
)
def make_local_index(
    metadata_path: str = Option(
        'model/metadata.pkl', '--metadata-path', help='Path to store Metadata'
    ),
    thumbnail_path: str = Option(
        'model/thumbnails.pkl', '--thumbnail-path', help='Path to store Thumbnails'
    ),
) -> None:
    local_thumbnails(
        metadata_path=metadata_path,
        thumbnail_path=thumbnail_path,
    )


@app.command(
    'reranker-train',
    help='Обучение реранкера LightGBM с логированием в MLflow.',
    context_settings={
        'allow_extra_args': True,
        'ignore_unknown_options': True,
    },
)
def reranker_train(ctx: Context) -> None:
    from vs.reranker.train import train

    sys.argv = [sys.argv[0]] + ctx.args
    train()


@app.command(
    'reranker-predict',
    help='Инференс PRD-реранкера из MLflow',
    context_settings={'allow_extra_args': True, 'ignore_unknown_options': True},
)
def reranker_predict(ctx: Context) -> None:
    from vs.reranker.predict import app as predict_app

    sys.argv = [sys.argv[0]] + ctx.args
    predict_app()


@download_app.command('clip', help='Скачать веса CLIP ViT-B/32.')
def dl_clip(
    dest_dir: str = Option('model', '--dest-dir', '-d', help='Куда сохранить'),
) -> None:
    from vs.download import download_clip

    download_clip(dest_dir)


@download_app.command('vocab', help='Скачать BPE-словарь (bpe_simple_vocab_16e6.txt.gz).')
def dl_vocab(
    dest_dir: str = Option('model', '--dest-dir', '-d', help='Куда сохранить'),
) -> None:
    from vs.download import download_vocab

    download_vocab(dest_dir)


@download_app.command('audioclip', help='Скачать веса AudioCLIP-Full-Training.pt.')
def dl_audioclip(
    dest_dir: str = Option('model', '--dest-dir', '-d', help='Куда сохранить'),
    url: str = Option('', '--url', '-u', help='Прямая HTTP-ссылка (по умолчанию — GitHub Releases)'),
) -> None:
    from vs.download import download_audioclip

    download_audioclip(dest_dir, url=url or None)


@download_app.command('all', help='Скачать все зависимости: CLIP, BPE-словарь, AudioCLIP.')
def dl_all(
    model_dir: str = Option('model', '--model-dir', '-d', help='Папка для всех артефактов'),
    audioclip_url: str = Option('', '--audioclip-url', help='Прямая HTTP-ссылка для AudioCLIP'),
) -> None:
    from vs.download import (
        download_audioclip,
        download_clip,
        download_vocab,
    )

    download_clip(model_dir)
    download_vocab(model_dir)
    download_audioclip(model_dir, url=audioclip_url or None)


@download_app.command('check', help='Проверить наличие всех файлов зависимостей.')
def dl_check(
    model_dir: str = Option('model', '--model-dir', '-d'),
) -> None:
    from vs.download import check_deps

    check_deps(model_dir)


if __name__ == "__main__":
    app()
