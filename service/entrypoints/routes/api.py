import os
import tempfile
from io import BytesIO
from typing import (
    Annotated,
    Optional,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from PIL import Image

from service.di import di
from service.domain.auth.auth import check_auth
from service.domain.auth.token import AuthContext
from service.domain.inference.schemas import InferenceFilters
from service.domain.internal.schemas import (
    BaseResponseSchema,
    ForwardRequestSchema,
)
from service.services.search import SearchService


api_router = APIRouter(
    prefix='/api/v1',
    tags=['API'],
    dependencies=[
        # Depends(check_auth)
    ],
)


@api_router.post(
    path='/forward',
    status_code=status.HTTP_200_OK,
)
async def make_forward_predict(
    request_data: ForwardRequestSchema,
    search_service: SearchService = Depends(di.provide(SearchService)),
) -> BaseResponseSchema:
    videos = await search_service.search_by_text(
        text=request_data.query,
        user='unknown',
    )
    return BaseResponseSchema(answer=videos)


@api_router.post(
    path='/forward/image',
    status_code=status.HTTP_200_OK,
)
async def make_forward_predict_image(
    file: UploadFile,
    search_service: SearchService = Depends(di.provide(SearchService)),
) -> BaseResponseSchema:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='No file provided',
        )
    pil_img = Image.open(BytesIO(await file.read()))
    videos = await search_service.search_by_image(pil_img, user='unknown')
    return BaseResponseSchema(answer=videos)


@api_router.post(
    path='/forward/audio',
    status_code=status.HTTP_200_OK,
)
async def make_forward_predict_audio(
    file: UploadFile,
    search_service: SearchService = Depends(di.provide(SearchService)),
) -> BaseResponseSchema:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='No file provided',
        )
    suffix = os.path.splitext(file.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        videos = await search_service.search_by_audio(tmp_path, user='unknown')
    finally:
        os.unlink(tmp_path)

    return BaseResponseSchema(answer=videos)


@api_router.get(
    path='/history',
    status_code=status.HTTP_200_OK,
)
async def get_historical_results(
    filters: Annotated[InferenceFilters, Query()],
    search_service: SearchService = Depends(di.provide(SearchService)),
) -> BaseResponseSchema:
    history = await search_service.get_searches(filters)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Not Founded History',
        )
    return BaseResponseSchema(answer=history)


@api_router.delete(
    path='/history',
    status_code=status.HTTP_200_OK,
)
async def delete_historical_results(
    user: Optional[str] = Query(default=None),
    search_service: SearchService = Depends(di.provide(SearchService)),
    token: AuthContext = Depends(check_auth),
) -> BaseResponseSchema:
    if not user:
        user = token.payload.user
    await search_service.delete_searches(user, token)
    return BaseResponseSchema(answer=None)
