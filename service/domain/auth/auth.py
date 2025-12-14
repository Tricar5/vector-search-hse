from typing import (
    Annotated,
    Optional,
)

from fastapi import (
    Depends,
    Request,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from service.di import di
from service.domain.auth.token import AuthContext
from service.domain.auth.user import (
    CTX_AUTH_USER,
    set_token_context,
)
from service.services.auth_service import AuthService


AUTH_HEADER = HTTPBearer(scheme_name='Bearer', auto_error=False)


def check_auth(
    request: Request,
    http_credentials: Annotated[  # noqa: WPS320
        Optional[HTTPAuthorizationCredentials],
        Depends(AUTH_HEADER),
    ],
    auth_service: Annotated[AuthService, Depends(lambda: di.get(AuthService))],
) -> AuthContext:
    try:
        token_context = CTX_AUTH_USER.get()
    except LookupError:
        token_context = None

    token = http_credentials.credentials if http_credentials else None
    payload = auth_service.validate_token(token)
    token_context = AuthContext(payload=payload)
    token_context.is_admin = auth_service.is_admin(payload)
    set_token_context(token_context)
    request.scope['token_context'] = CTX_AUTH_USER
    return token_context
