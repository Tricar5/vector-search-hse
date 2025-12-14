from dishka import (
    Provider,
    Scope,
    make_async_container,
    provide,
)

from service.adapters.engines.base import Engine
from service.adapters.engines.local import BaselineSearchEngine
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

    @provide(scope=Scope.REQUEST)
    def get_service(self, engine: Engine) -> SearchService:
        return SearchService(engine)


provider = ApplicationSearchProvider()

main_container = make_async_container(provider)
