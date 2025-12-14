import datetime
from typing import (
    Any,
    Dict,
    Optional,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class InferenceCreateSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    query_type: str = Field(description='Тип запроса: текстовый или изображение')
    query: str = Field(description='Запрос')
    result: Dict[str, Any] = Field(  # noqa: WPS110
        description='Результат выполнения запроса',
    )
    processing_time: float = Field(description='Время выполнения в секундах')


class SearchResultSchema(InferenceCreateSchema):
    id: int = Field(description='ID запроса в базе данных')
    created_at: datetime.datetime = Field(
        description='Время совершения запроса',
    )


class InferenceFilters(BaseModel):
    limit: Optional[int] = Field(
        default=10,
    )
    offset: Optional[int] = Field(
        default=0,
    )
    query: Optional[str] = Field(
        default=None,
        description='Запрос',
    )
    query_type: Optional[str] = Field(
        default=None,
    )
