import time
from typing import List

from service.adapters.engines.base import Engine
from service.db.repositories.search import SearchRepository
from service.domain.inference.schemas import (
    InferenceCreateSchema,
    InferenceFilters,
    SearchResultSchema,
)
from service.domain.inference.types import InputQueryType
from service.domain.videos.schemas import VideoDescription


class SearchService:

    def __init__(
        self,
        engine: Engine,
        repo: SearchRepository,
    ) -> None:
        self._engine = engine
        self._repo = repo

    async def search_by_text(self, text: str, user: str) -> list[VideoDescription]:
        st = time.monotonic()
        videos = self._engine.search_videos_by_text(text)
        processing_time = time.monotonic() - st
        await self._store_results(
            text=text,
            videos=videos,
            processing_time=processing_time,
            user=user,
        )
        return videos

    async def get_searches(
        self,
        filters: InferenceFilters,
    ) -> List[SearchResultSchema]:
        return await self._repo.get_by_filters(**filters.model_dump(exclude_none=True))

    async def _store_results(
        self,
        text: str,
        videos: list[VideoDescription],
        processing_time: float,
        user: str,
    ) -> SearchResultSchema:
        return await self._repo.create(
            obj_in=InferenceCreateSchema(
                query=text,
                query_type=InputQueryType.TEXT,
                result={'videos': videos},
                processing_time=processing_time,
                user=user,
            ),
        )
