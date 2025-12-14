from typing import (
    Annotated,
    Optional,
)

from fastapi import Depends
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from service.di import di
from service.domain.auth.token import TokenContext
from service.domain.auth.user import (
    CTX_AUTH_USER,
    set_token_context,
)
from service.services.auth_service import AuthService


AUTH_HEADER = HTTPBearer(scheme_name='Bearer', auto_error=False)


def check_auth(
    http_credentials: Annotated[  # noqa: WPS320
        Optional[HTTPAuthorizationCredentials],
        Depends(AUTH_HEADER),
    ],
    auth_service: Annotated[AuthService, Depends(lambda: di.get(AuthService))],
) -> None:
    try:
        token_context = CTX_AUTH_USER.get()
    except LookupError:
        token_context = None
    if token_context is None:
        token_context = TokenContext()
        set_token_context(token_context)
    token = http_credentials.credentials if http_credentials else None
    payload = auth_service.validate_token(token)
    token_context.payload = payload
    token_context.is_admin = auth_service.is_admin(payload)
