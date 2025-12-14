from typing import (
    Annotated,
    Any,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import BaseModel

from service.di import di
from service.domain.auth.auth import check_auth
from service.domain.inference.schemas import InferenceFilters
from service.domain.internal.errors.exc import ModelException
from service.services.search import SearchService


class ForwardRequestSchema(BaseModel):
    query: str


class BaseResponseSchema(BaseModel):
    success: bool = True
    answer: Any


api_router = APIRouter(prefix='/api/v1', tags=['API'], dependencies=[Depends(check_auth)])


@api_router.post(
    path='/forward',
    status_code=status.HTTP_200_OK,
)
async def make_forward_predict(
    request_data: ForwardRequestSchema,
    search_service: SearchService = Depends(lambda: di.get(SearchService)),
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
async def get_historical_results(
    filters: Annotated[InferenceFilters, Query()],
    search_service: SearchService = Depends(lambda: di.get(SearchService)),
) -> BaseResponseSchema:
    history = await search_service.get_searches(filters)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='Not Founded History'
        )
    return BaseResponseSchema(answer=history)
