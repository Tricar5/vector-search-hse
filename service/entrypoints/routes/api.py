from typing import Any

from dishka.integrations.fastapi import (
    FromDishka,
    inject,
)
from fastapi import (
    APIRouter,
    status,
)
from pydantic import BaseModel

from service.domain.internal.errors.exc import ModelException
from service.services.search import SearchService


class ForwardRequestSchema(BaseModel):
    query: str


class BaseResponseSchema(BaseModel):
    success: bool = True
    answer: Any


api_router = APIRouter(
    prefix='/api/v1',
)


@api_router.post(
    path='/forward',
    status_code=status.HTTP_201_CREATED,
)
@inject
def make_forward_predict(
    request_data: ForwardRequestSchema,
    search_service: FromDishka[SearchService],
) -> BaseResponseSchema:
    try:
        videos = search_service.search_by_text(request_data.query)
    except Exception:
        raise ModelException('Модель не смогла обработать данные')
    return BaseResponseSchema(answer=videos)
