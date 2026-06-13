# flake8: noqa WPS339, WPS110
import csv
import io
import logging
import os
import subprocess
import tempfile
from collections.abc import Generator
from io import BytesIO

import cv2
from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.requests import Request
from starlette.responses import (
    HTMLResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from service.adapters.engines.base import Engine
from service.adapters.engines.local import LocalSearchEngine
from service.di import di
from service.domain.videos.schemas import VideoDescription
from service.services.search import SearchService
from vs.frames import open_and_load_frame


templates = Jinja2Templates(directory='service/templates')

logger = logging.getLogger(__name__)
web_router = APIRouter(prefix='')


def render_main_page(
    request: Request,
    video_descriptions: list[VideoDescription],
    orig_data: str | None = None,
) -> HTMLResponse:
    frames = []
    used_videos: dict[str, tuple[float, float, float]] = {}
    for desc in video_descriptions:
        img_url = f'/image?video={desc.video_id}&frame_number={desc.frame_num}'
        frames.append((desc.path, img_url, desc.fps))
        used_videos[desc.path] = (desc.start_pos, desc.end_pos, desc.score)
    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'frames': frames,
            'used_videos': used_videos,
            'orig_data': orig_data,
        },
    )


@web_router.get('/', response_class=HTMLResponse)
async def main_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse('index.html', {'request': request})


@web_router.post('/load_image')
async def upload_image(
    request: Request,
    file: UploadFile,
    search_service: SearchService = Depends(di.provide(SearchService)),
) -> Response:
    if not file.filename:
        return RedirectResponse(url='/', status_code=303)

    pil_img = Image.open(BytesIO(await file.read()))
    video_desc = await search_service.search_by_image(pil_img)
    return render_main_page(request, video_desc)


@web_router.post('/load_text')
async def upload_text(
    request: Request,
    text: str = Form(...),
    search_service: SearchService = Depends(di.provide(SearchService)),
) -> Response:
    if not text:
        return RedirectResponse('/', status_code=303)

    video_descriptions = await search_service.search_by_text(text)
    return render_main_page(request, video_descriptions, orig_data=text)


@web_router.post('/load_audio')
async def upload_audio(
    request: Request,
    file: UploadFile,
    search_service: SearchService = Depends(di.provide(SearchService)),
) -> Response:
    if not file.filename:
        return RedirectResponse(url='/', status_code=303)

    suffix = os.path.splitext(file.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        video_desc = await search_service.search_by_audio(tmp_path)
    finally:
        os.unlink(tmp_path)

    return render_main_page(request, video_desc)


@web_router.post('/download_csv')
async def download_csv(
    request: Request,
) -> StreamingResponse:
    form = await request.form()
    total = int(form.get('len', 0))
    orig_data = form.get('orig_data', 'results')

    fieldnames = ['idx', 'max', 'mean', 'std', 'perc_90', 'num_passed', 'range', 'rel']

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    import ast
    for i in range(1, total + 1):
        raw_stats = form.get(f'{i}_stats', '{}')
        stats: dict = ast.literal_eval(raw_stats) if isinstance(raw_stats, str) else {}
        rel = form.get(f'{i}_rel', 'off') == 'on'
        writer.writerow({
            'idx': i,
            'max': stats.get('max', ''),
            'mean': stats.get('mean', ''),
            'std': stats.get('std', ''),
            'perc_90': stats.get('perc_90', ''),
            'num_passed': stats.get('num_passed', ''),
            'range': stats.get('range', ''),
            'rel': rel,
        })

    buf.seek(0)
    filename = f'{orig_data}.csv'
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


@web_router.get('/image')
async def serve_image(
    video: int = Query(),
    frame_number: int = Query(),
    thumbnail_size: int | None = Query(default=None),
    engine: LocalSearchEngine = Depends(di.provide(Engine)),
) -> StreamingResponse:
    int_to_video = engine._int_to_video
    video_path = int_to_video[int(video)]

    img, _ = open_and_load_frame((video_path, frame_number), thumbnail_size)

    ok, encoded = cv2.imencode('.jpg', img)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Cannot encode image',
        )

    img_bytes = BytesIO(encoded.tobytes())

    fn = os.path.basename(video_path).rsplit('.', 1)[0]
    if frame_number != -1:
        fn = f'{fn}_frame_{frame_number}'
    if thumbnail_size:
        fn += '_thumbnail'  # noqa: WPS336
    filename = f'{fn}.jpg'

    return StreamingResponse(
        content=img_bytes,
        media_type='image/jpeg',
        headers={'Content-Disposition': f"inline; filename='{filename}'"},
    )


@web_router.get('/video_segment')
async def video_segment(
    path: str = Query(),
    fps: int = Query(),
    frame_start: int = Query(),
    frame_end: int = Query(),
) -> StreamingResponse:
    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Video not found',
        )

    start_seconds = frame_start / fps
    end_seconds = frame_end / fps
    duration = end_seconds - start_seconds

    cmd = [
        'ffmpeg',
        '-ss', str(start_seconds),
        '-i', path,
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-c:a', 'aac',
        '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
        '-f', 'mp4',
        '-crf', '29',
        '-',
    ]

    def generate() -> Generator[bytes, None, None]:  # noqa: WPS430
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            yield from iter(lambda: proc.stdout.read(64 * 1024), b'')  # type: ignore
        except Exception as exc:
            logger.exception(exc)
        finally:
            proc.kill()

    filename = os.path.basename(path).rsplit('.', 1)[0]
    filename = f'{filename}_segment_{frame_start}-{frame_end}.mp4'

    return StreamingResponse(
        generate(),
        media_type='video/mp4',
        headers={
            'Content-Disposition': f"attachment; filename='{filename}'",
            'Accept-Ranges': 'bytes',
        },
    )
