from dishka import (
    Provider,
    Scope,
    make_async_container,
    provide,
)

from service.adapters.engines.base import Engine
from service.adapters.engines.local import BaselineSearchEngine
from service.db.connections.base import Connector
from service.db.connections.postgres import Postgres
from service.db.repositories.search import SearchRepository
from service.services.search import SearchService
from service.settings import (
    AppSettings,
    get_settings,
)


class ApplicationSearchProvider(Provider):

    @provide(scope=Scope.APP)
    def get_settings(self) -> AppSettings:
        return get_settings()

    @provide(scope=Scope.APP)
    def get_engine(self, settings: AppSettings) -> Engine:
        return BaselineSearchEngine.build_engine(settings)

    @provide(scope=Scope.APP)
    def get_connection(self, settings: AppSettings) -> Connector:
        return Postgres(settings.async_dsn)

    @provide(scope=Scope.APP)
    def get_history_repo(self, conn: Connector) -> SearchRepository:
        return SearchRepository(conn)

    @provide(scope=Scope.REQUEST)
    def get_service(self, engine: Engine, repo: SearchRepository) -> SearchService:
        return SearchService(engine, repo)


provider = ApplicationSearchProvider()

main_container = make_async_container(provider)
