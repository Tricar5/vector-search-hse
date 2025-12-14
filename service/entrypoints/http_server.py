from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI

from service.di import di
from service.domain.internal.errors.registry import DEFAULT_EXCEPTION_HANDLERS
from service.domain.internal.metrics.instrumentator import MetricsInstrumentator
from service.entrypoints.routes.api import api_router
from service.settings import (
    AppSettings,
    settings,
)


def create_app(settings: AppSettings) -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        docs_url='/docs',
    )
    # app.include_router(web_router)
    app.include_router(api_router)

    metrics = di.get(MetricsInstrumentator)

    for exc_handler in DEFAULT_EXCEPTION_HANDLERS:
        app.add_exception_handler(exc_handler.exc, exc_handler.exc_handler)
    setup_dishka(di, app)
    metrics.setup(app, settings.app_name)
    return app


app: FastAPI = create_app(settings)
