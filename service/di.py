from dishka import (
    Provider,
    Scope,
    make_container,
    provide,
)

from service.adapters.engines.base import Engine
from service.adapters.engines.local import LocalSearchEngine
from service.db.connections.base import Connector
from service.db.connections.postgres import Postgres
from service.db.repositories.search import SearchRepository
from service.domain.internal.metrics.instrumentator import MetricsInstrumentator
from service.services.auth_service import AuthService
from service.services.search import SearchService
from service.settings import (
    AppSettings,
    get_settings,
)


class AppProvider(Provider):

    @provide(scope=Scope.APP)
    def get_settings(self) -> AppSettings:
        return get_settings()

    @provide(scope=Scope.APP)
    def get_metrics(self) -> MetricsInstrumentator:
        return MetricsInstrumentator()

    @provide(scope=Scope.APP)
    def get_engine(self, settings: AppSettings) -> Engine:
        return LocalSearchEngine.build_engine(settings)

    @provide(scope=Scope.APP)
    def get_connection(self, settings: AppSettings) -> Connector:
        return Postgres(settings.db)

    @provide(scope=Scope.APP)
    def get_history_repo(self, conn: Connector) -> SearchRepository:
        return SearchRepository(conn)

    @provide(scope=Scope.APP)
    def get_search_service(self, engine: Engine, repo: SearchRepository) -> SearchService:
        return SearchService(engine, repo)

    @provide(scope=Scope.APP)
    def get_auth_service(self, settings: AppSettings) -> AuthService:
        return AuthService(settings.auth)


di = make_container(AppProvider())
