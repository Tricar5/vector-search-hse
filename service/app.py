from fastapi import FastAPI

from service.api.web import web_router

app = FastAPI(
    docs_url="/docs",
)
app.include_router(web_router)
