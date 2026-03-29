from typing import (
    Annotated,
    Optional,
)

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)

from service.di import di
from service.domain.auth.auth import check_auth
from service.domain.auth.token import AuthContext
from service.domain.inference.schemas import InferenceFilters
from service.domain.internal.errors.exc import ModelException
from service.domain.internal.schemas import (
    BaseResponseSchema,
    ForwardRequestSchema,
)
from service.services.search import SearchService


api_router = APIRouter(
    prefix='/api/v1',
    tags=['API'],
    dependencies=[Depends(check_auth)],
)


@api_router.post(
    path='/forward',
    status_code=status.HTTP_200_OK,
)
async def make_forward_predict(
    request_data: ForwardRequestSchema,
    search_service: SearchService = Depends(lambda: di.get(SearchService)),
    token: AuthContext = Depends(check_auth),
) -> BaseResponseSchema:
    try:
        videos = await search_service.search_by_text(
            text=request_data.query,
            user=token.payload.user,
        )
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
    token: AuthContext = Depends(check_auth),
) -> BaseResponseSchema:
    filters.user = token.payload.user
    history = await search_service.get_searches(filters)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='Not Founded History'
        )
    return BaseResponseSchema(answer=history)


@api_router.delete(
    path='/history',
    status_code=status.HTTP_200_OK,
)
async def delete_historical_results(
    user: Optional[str] = Query(default=None),
    search_service: SearchService = Depends(lambda: di.get(SearchService)),
    token: AuthContext = Depends(check_auth),
) -> BaseResponseSchema:
    if not user:
        user = token.payload.user
    await search_service.delete_searches(user, token)
    return BaseResponseSchema(answer=None)
