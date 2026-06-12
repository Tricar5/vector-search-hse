import logging
import time
from typing import List

from service.adapters.engines.base import Engine
from service.db.repositories.search import SearchRepository
from service.domain.auth.token import AuthContext
from service.domain.inference.schemas import (
    InferenceCreateSchema,
    InferenceFilters,
    SearchResultSchema,
)
from service.domain.inference.types import InputQueryType
from service.domain.internal.metrics.metrics import (
    text_length_metric,
    token_count_metric,
)
from service.domain.videos.schemas import VideoDescription


logger = logging.getLogger(__name__)


class SearchService:

    def __init__(
        self,
        engine: Engine,
        repo: SearchRepository,
    ) -> None:
        self._engine = engine
        self._repo = repo

    async def search_by_text(
        self,
        text: str,
        user: str = 'unknown',
    ) -> list[VideoDescription]:
        st = time.monotonic()
        videos = self._engine.search_videos_by_text(text)
        processing_time = time.monotonic() - st

        text_length_metric.observe(len(text))
        token_count_metric.observe(len(text.split()))
        try:
            await self._store_results(
                query=text,
                query_type=InputQueryType.TEXT,
                videos=videos,
                processing_time=processing_time,
                user=user,
            )
        except Exception as exc:
            logger.error('Cannot store data: %s', exc)
        return videos

    async def search_by_image(
        self,
        img: object,
        user: str = 'unknown',
    ) -> list[VideoDescription]:
        st = time.monotonic()
        videos = self._engine.search_videos_by_image(img)  # type: ignore[arg-type]
        processing_time = time.monotonic() - st
        try:
            await self._store_results(
                query='<image>',
                query_type=InputQueryType.IMAGE,
                videos=videos,
                processing_time=processing_time,
                user=user,
            )
        except Exception as exc:
            logger.error('Cannot store data: %s', exc)
        return videos

    async def search_by_audio(
        self,
        audio_path: str,
        user: str = 'unknown',
    ) -> list[VideoDescription]:
        st = time.monotonic()
        videos = self._engine.search_videos_by_audio(audio_path)
        processing_time = time.monotonic() - st
        try:
            await self._store_results(
                query='<audio>',
                query_type=InputQueryType.AUDIO,
                videos=videos,
                processing_time=processing_time,
                user=user,
            )
        except Exception as exc:
            logger.error('Cannot store data: %s', exc)
        return videos

    async def get_searches(
        self,
        filters: InferenceFilters,
    ) -> List[SearchResultSchema]:
        return await self._repo.get_by_filters(**filters.model_dump(exclude_none=True))

    async def delete_searches(self, user: str, token: AuthContext) -> None:
        if not token.is_admin and user != token.payload.user:
            return
        await self._repo.delete_history(user_to_delete=user)

    async def _store_results(
        self,
        query: str,
        query_type: InputQueryType,
        videos: list[VideoDescription],
        processing_time: float,
        user: str = 'unknown',
    ) -> SearchResultSchema:
        return await self._repo.create(
            obj_in=InferenceCreateSchema(
                query=query,
                query_type=query_type,
                result={'videos': [v.model_dump() for v in videos]},
                processing_time=processing_time,
                user=user,
            ),
        )
