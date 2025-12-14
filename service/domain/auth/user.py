from contextvars import ContextVar
from typing import Optional

from service.domain.auth.token import TokenContext


CTX_AUTH_USER: ContextVar[TokenContext] = ContextVar('auth_user')


def get_auth_user() -> Optional[TokenContext]:
    context = get_token_context()
    if context is None:
        return None
    return context


def set_token_context(context: TokenContext) -> None:
    CTX_AUTH_USER.set(context)


def get_token_context() -> Optional[TokenContext]:
    try:
        return CTX_AUTH_USER.get()
    except LookupError:
        return None
