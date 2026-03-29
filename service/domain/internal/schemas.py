from typing import Any

from pydantic import BaseModel


class ForwardRequestSchema(BaseModel):
    query: str


class BaseResponseSchema(BaseModel):
    success: bool = True
    answer: Any
