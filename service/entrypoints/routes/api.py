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

from service.adapters.engines.local import BaselineSearchEngine
from service.domain.internal.errors.exc import ModelException


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
    engine: FromDishka[BaselineSearchEngine],
) -> BaseResponseSchema:
    try:
        videos = engine.search_videos_by_text(request_data.query)
    except Exception:
        raise ModelException('Модель не смогла обработать данные')
    return BaseResponseSchema(answer=videos)
