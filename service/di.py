from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    Type,
    TypeVar,
)

from punq import (
    Container,
    Scope,
)

from service.adapters.engines.base import Engine
from service.adapters.engines.local import LocalSearchEngine
from service.db.connections.postgres import Postgres
from service.db.repositories.search import SearchRepository
from service.services.auth_service import AuthService
from service.services.search import SearchService
from service.settings import (
    AppSettings,
    settings,
)

ClassT = TypeVar('ClassT')


class DI:
    def __init__(self) -> None:
        self._container = Container()

    def register_all(self, settings: AppSettings) -> 'DI':
        self._container.register(
            AppSettings,
            instance=settings,
            scope=Scope.singleton,
        )
        self._register_infrastructure(settings)
        self._register_repos()
        self._register_services(settings)
        return self

    def override(
        self,
        class_type: Type[ClassT],
        **kwargs: Any,
    ) -> None:
        self._container.register(class_type, **kwargs)
        self._container.resolve_all(class_type)

    def resolve(self, class_type: Type[ClassT]) -> ClassT:
        return self._container.resolve(class_type)

    def provide(self, class_type: Type[ClassT]) -> Callable[[], ClassT]:
        return lambda: self._container.resolve(class_type)

    def _register_infrastructure(self, settings: AppSettings) -> None:
        self._container.register(
            Postgres,
            factory=lambda: Postgres(settings.db),
            scope=Scope.singleton,
        )

    def _register_repos(self) -> None:
        self._container.register(
            SearchRepository,
            scope=Scope.singleton,
        )

    def _register_services(self, settings: AppSettings) -> None:
        self._container.register(
            AuthService,
            scope=Scope.singleton,
        )
        self._container.register(
            SearchService,
            scope=Scope.singleton,
        )
        self._container.register(
            Engine,
            factory=LocalSearchEngine.build_engine,
            settings=settings,
            scope=Scope.singleton,
        )


ROOT_PATH = Path(__file__).parent.parent
SETTINGS_PATH = ROOT_PATH / 'settings'
di: Final[DI] = DI().register_all(settings)
