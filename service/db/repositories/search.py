from sqlalchemy import delete

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

    async def delete_history(
        self,
        user_to_delete: str,
    ) -> None:
        stmt = delete(self.model)
        if not 'all' == user_to_delete:
            stmt = stmt.where(self.model.user == user_to_delete)

        async with self.session_factory() as session:
            await session.execute(stmt)
            session.commit()
