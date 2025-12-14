from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI

from service.di import di
from service.domain.internal.errors.registry import DEFAULT_EXCEPTION_HANDLERS
from service.entrypoints.routes.api import api_router
from service.settings import (
    AppSettings,
    settings,
)


def create_app(settings: AppSettings) -> FastAPI:
    app = FastAPI(
        title=settings.name,
        docs_url='/docs',
    )
    # app.include_router(web_router)
    app.include_router(api_router)

    for exc_handler in DEFAULT_EXCEPTION_HANDLERS:
        app.add_exception_handler(exc_handler.exc, exc_handler.exc_handler)
    setup_dishka(di, app)
    return app


app: FastAPI = create_app(settings)
