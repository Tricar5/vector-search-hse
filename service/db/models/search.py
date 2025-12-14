# flake8: noqa:WPS432
from typing import (
    Any,
    Dict,
)

from sqlalchemy import (
    Float,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from service.db.models.base import Base
from service.domain.inference.types import InputQueryType


class SearchHistoryModel(Base):
    __tablename__ = 'search_history'

    query_type: Mapped[InputQueryType] = mapped_column(String(50), index=True)
    query: Mapped[str] = mapped_column(String(50))
    result: Mapped[Dict[str, Any]] = mapped_column(JSONB)  # noqa: WPS110
    processing_time: Mapped[float] = mapped_column(
        Float,
    )
