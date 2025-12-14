from service.adapters.engines.base import Engine
from service.domain.videos.schemas import VideoDescription


class SearchService:

    def __init__(
        self,
        engine: Engine,
    ) -> None:
        self._engine = engine

    def search_by_text(
        self,
        text: str,
    ) -> list[VideoDescription]:
        return self._engine.search_videos_by_text(text)
