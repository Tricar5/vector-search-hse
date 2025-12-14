from enum import StrEnum
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
)


class AdminUsers(StrEnum):
    HSE_ADMIN = 'hse-admin'


class TokenPayload(BaseModel):
    user: str = Field(alias='u')
    issuer: str = Field(default='vector-search-service', alias='i')


class TokenContext(BaseModel):
    payload: Optional[TokenPayload] = None
    is_admin: bool = False
