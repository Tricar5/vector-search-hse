from fastapi import FastAPI

from service.entrypoints.routes.web import web_router
from service.settings import (
    AppSettings,
    settings,
)


def create_app(settings: AppSettings) -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        docs_url='/docs',
    )
    app.include_router(web_router)

    return app


app: FastAPI = create_app(settings)
