import logging
from typing import (
    Any,
    Generic,
    Optional,
    Type,
    TypeVar,
)

from fastapi.encoders import jsonable_encoder
from pydantic import (
    BaseModel,
    ValidationError,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from service.db.connections.base import Connector
from service.db.models.base import Base


ModelType = TypeVar('ModelType', bound=Base)
CreateSchemaType = TypeVar('CreateSchemaType', bound=BaseModel)
SchemaType = TypeVar('SchemaType', bound=BaseModel)

logger = logging.getLogger(__name__)


class BaseRepository(Generic[ModelType, CreateSchemaType, SchemaType]):
    """Base CRUD Class to work with Generics"""

    model: type[ModelType]
    input_entity_schema: Type[CreateSchemaType]
    entity_schema: Type[SchemaType]

    def __init__(self, conn: Connector) -> None:
        """Repository object with default methods to CRUD."""
        self.session_factory = conn.session

    async def create(self, *, obj_in: CreateSchemaType) -> SchemaType:
        async with self.session_factory() as db:
            obj_in_data = jsonable_encoder(obj_in)
            db_obj = self.model(**obj_in_data)

            db.add(db_obj)
            await db.commit()
            await db.refresh(db_obj)

        return self._record_to_entity(db_obj)  # type: ignore

    async def get(self, obj_id: int) -> SchemaType | None:
        async with self.session_factory() as db:
            stmt = select(self.model).filter(self.model.id == obj_id)
            res = await db.execute(stmt)
            db_obj = res.scalar()
        return self._record_to_entity(db_obj)

    async def get_by_filters(
        self,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> list[SchemaType]:
        async with self.session_factory() as db:
            stmt = select(self.model)
            if offset is not None:
                stmt = stmt.offset(offset)
            if limit is not None:
                stmt = stmt.limit(limit)
            res = await db.execute(stmt)
            db_objects = res.scalars().all()
        raw_entities = [self._record_to_entity(db_object) for db_object in db_objects]
        return [entity for entity in raw_entities if entity is not None]

    async def update(
        self,
        db: AsyncSession,
        *,
        db_obj: ModelType,
        obj_in: SchemaType | dict[str, Any],
    ) -> ModelType:
        obj_data = jsonable_encoder(db_obj)
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        async with self.session_factory() as db:
            db.add(db_obj)
            await db.commit()
            await db.refresh(db_obj)

        return db_obj

    async def remove(self, *, obj_id: int) -> ModelType | None:
        async with self.session_factory() as db:
            stmt = select(self.model).filter(self.model.id == obj_id)
            res = await db.execute(stmt)
            db_obj = res.scalar()

            if not db_obj:
                return None

            await db.delete(db_obj)
            await db.commit()

        return db_obj

    def _record_to_entity(self, record: ModelType) -> Optional[SchemaType]:
        try:
            return self.entity_schema.model_validate(record)
        except ValidationError as exc:
            logger.error(
                '[%s] Occured %s, Failed to validate record: %s',
                self.__class__.__name__,
                str(exc),
                str(record),
            )
        return None
