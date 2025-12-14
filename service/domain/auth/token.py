from enum import StrEnum

from pydantic import (
    BaseModel,
    Field,
)


class AdminUsers(StrEnum):
    HSE_ADMIN = 'hse-vector'


class TokenPayload(BaseModel):
    user: str = Field(alias='u')
    issuer: str = Field(default='vector-search-service', alias='i')


class TokenContext(BaseModel):
    payload: TokenPayload
    is_admin: bool = False
