import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from service.adapters.engines.base import Engine
from service.domain.internal.metrics.collector import MetricsCollector
from service.domain.internal.metrics.default import DefaultMetrics
from service.di import di
from service.domain.internal.errors.registry import DEFAULT_EXCEPTION_HANDLERS
from service.entrypoints.routes.api import api_router
from service.entrypoints.routes.web import web_router
from service.settings import (
    AppSettings,
    get_settings,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info('Loading engine and dependencies...')
    di.resolve(Engine)
    di.resolve(MetricsCollector)
    logger.info('Engine loaded, service is ready.')
    yield
    logger.info('Shutting down.')


def create_app(settings: AppSettings) -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        docs_url='/docs',
        lifespan=lifespan,
    )

    app.include_router(api_router)
    app.include_router(web_router)

    for exc_handler in DEFAULT_EXCEPTION_HANDLERS:
        app.add_exception_handler(exc_handler.exc, exc_handler.exc_handler)

    DefaultMetrics().setup(app, settings.app_name)

    return app


app: FastAPI = create_app(get_settings())
