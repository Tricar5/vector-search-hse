"""Connection class for postgresql database"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from service.db.connections.base import Connector


logger = logging.getLogger(__name__)


class Postgres(Connector):
    """
    Postgres Connection
    """

    def __init__(self, db_dsn: str) -> None:
        self.engine: AsyncEngine = create_async_engine(db_dsn)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                logger.exception("Session rollback because of exception")
                await session.rollback()
                raise
            finally:
                await session.commit()
