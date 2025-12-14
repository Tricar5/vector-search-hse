import datetime
from typing import (
    Any,
    Dict,
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


class InferenceSchema(InferenceCreateSchema):
    id: int = Field(description='ID запроса в базе данных')
    created_at: datetime.datetime = Field(
        description='Время совершения запроса',
    )
