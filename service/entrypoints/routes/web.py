# flake8: noqa WPS339, WPS110
import logging
import os
import subprocess
from collections.abc import Generator
from io import BytesIO
from typing import Any

import cv2
import torch
from fastapi import (
    APIRouter,
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

from service.adapters.files import load_yml_config
from service.settings import settings
from vs.frames import open_and_load_frame
from vs.local.engine import load_search_index


templates = Jinja2Templates(directory='service/templates')

logger = logging.getLogger(__name__)
web_router = APIRouter(
    prefix='',
)

index_config = load_yml_config(settings.ENGINE_CONFIG_PATH)

search_index = load_search_index(
    index_config['local']['index_path'],
    index_config['local']['metadata_path'],
    index_config['local']['thumbnail_path'],
    index_config['local']['device'],
)


def render_main_page(
    request: Request,
    video_descriptions: Any,
    used_videos: Any,
) -> HTMLResponse:
    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'frames': video_descriptions or [],
            'used_videos': used_videos or {},
        },
    )


@web_router.get('/', response_class=HTMLResponse)
async def main_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
        },
    )


@web_router.post('/load_image')
async def upload_image(
    request: Request,
    file: UploadFile,
) -> Response:
    if not file.filename:
        return RedirectResponse(url='/', status_code=303)

    # Read image
    pil_img = Image.open(BytesIO(await file.read()))

    with torch.no_grad():
        query_tensor = search_index.encode_image(pil_img)  # type: ignore

    if len(query_tensor) == 0:
        return RedirectResponse('/', status_code=303)

    # Query videos
    video_desc, used_videos = search_index.query_videos_by_tensor(
        query_tensor,
        video_threshold=0.30,
        frame_threshold=0.2,
        percentile=1,
    )

    return render_main_page(request, video_desc, used_videos)


@web_router.post('/load_text')
async def upload_text(
    request: Request,
    text: str = Form(...),
) -> Response:
    if not text:
        return RedirectResponse('/', status_code=303)

    with torch.no_grad():
        query_tensor = search_index.encode_text(text)

    if len(query_tensor) == 0:
        return RedirectResponse('/', status_code=303)

    video_desc, used_videos = search_index.query_videos_by_tensor(
        query_tensor,
        video_threshold=0.2,
        frame_threshold=0.2,
        percentile=0.8,
    )

    return render_main_page(request, video_desc, used_videos)


@web_router.get('/image')
async def serve_image(
    video: int = Query(),
    frame_number: int = Query(),
    thumbnail_size: int | None = Query(default=None),
) -> StreamingResponse:
    int_to_video = search_index.int_to_video
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
        '-ss',
        str(start_seconds),
        '-i',
        path,
        '-t',
        str(duration),
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-c:a',
        'aac',
        '-movflags',
        'frag_keyframe+empty_moov+default_base_moof',
        '-f',
        'mp4',
        '-crf',
        '29',
        '-',
    ]

    def generate() -> Generator[bytes, None, None]:  # noqa: WPS430
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            FILE_SIZE = 64 * 1024  # noqa: WPS432
            yield from iter(lambda: proc.stdout.read(), b'')  # type: ignore
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
