from typing import (
    Annotated,
    Any,
)

from dishka.integrations.fastapi import (
    FromDishka,
    inject,
)
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    status,
)
from pydantic import BaseModel

from service.domain.inference.schemas import InferenceFilters
from service.domain.internal.errors.exc import ModelException
from service.services.search import SearchService


class ForwardRequestSchema(BaseModel):
    query: str


class BaseResponseSchema(BaseModel):
    success: bool = True
    answer: Any


api_router = APIRouter(
    prefix='/api/v1',
    tags=['API'],
)


@api_router.post(
    path='/forward',
    status_code=status.HTTP_200_OK,
)
@inject
async def make_forward_predict(
    request_data: ForwardRequestSchema,
    search_service: FromDishka[SearchService],
) -> BaseResponseSchema:
    try:
        videos = await search_service.search_by_text(request_data.query)
    except Exception:
        raise ModelException('Модель не смогла обработать данные')
    return BaseResponseSchema(answer=videos)


@api_router.get(
    path='/history',
    status_code=status.HTTP_200_OK,
)
@inject
async def get_historical_results(
    filters: Annotated[InferenceFilters, Query()],
    search_service: FromDishka[SearchService],
) -> BaseResponseSchema:
    history = await search_service.get_searches(filters)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='Not Founded History'
        )
    return BaseResponseSchema(answer=history)
