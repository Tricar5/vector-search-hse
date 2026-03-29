import abc
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncGenerator,
)


class Connector(abc.ABC):
    @abc.abstractmethod
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[Any, Any]:
        """Getting connection pool in asynchronous context."""
        try:
            yield
        except Exception as exc:
            raise exc
