from service.db.models.search import SearchHistoryModel
from service.db.repositories.base import BaseRepository
from service.domain.inference.schemas import (
    InferenceCreateSchema,
    SearchResultSchema,
)


class SearchRepository(
    BaseRepository[SearchHistoryModel, InferenceCreateSchema, SearchResultSchema]
):
    """Репозиторий для работы с логами"""

    model = SearchHistoryModel
    input_entity_schema = InferenceCreateSchema
    entity_schema = SearchResultSchema
