from dishka import (
    Provider,
    Scope,
    make_async_container,
)

from service.adapters.engines.local import BaselineSearchEngine
from service.settings import (
    AppSettings,
    get_settings,
)


main_provider = Provider()

main_provider.provide(
    source=get_settings,
    provides=AppSettings,
    scope=Scope.APP,
)
main_provider.provide(
    source=BaselineSearchEngine.load_search_index,
    provides=BaselineSearchEngine,
    scope=Scope.APP,
)

main_container = make_async_container(main_provider)
