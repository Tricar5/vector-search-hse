import datetime
from types import MappingProxyType
from typing import Mapping

from sqlalchemy import (
    TIMESTAMP,
    Integer,
    MetaData,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
)


CONVENTIONS: Mapping[str, str] = MappingProxyType(
    {
        'ix': 'ix_%(column_0_label)s',
        'uq': 'uq_%(table_name)s_%(column_0_name)s',
        'ck': 'ck_%(table_name)s_%(constraint_name)s',
        'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s',
        'pk': 'pk_%(table_name)s',
    },
)
metadata = MetaData(naming_convention=CONVENTIONS)


class Base(DeclarativeBase):
    metadata = metadata

    __abstract__ = True

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        unique=True,
        doc='Unique index of element (type UUID)',
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),  # pylint: disable=E1102
        nullable=False,
        doc='Date and time of create (type TIMESTAMP)',
    )

    def __repr__(self) -> str:
        columns = {
            column.name: getattr(self, column.name) for column in self.__table__.columns
        }
        cols_repr = ', '.join(
            map(lambda atr: f'{atr[0]}={atr[1]}', columns.items())  # noqa: WPS432
        )  # noqa: WPS111, WPS221
        return f'<{self.__tablename__}: {cols_repr}>'
