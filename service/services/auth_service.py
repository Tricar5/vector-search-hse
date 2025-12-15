from logging import getLogger
from typing import Optional

from fastapi import status
from jose import (
    constants,
    jwt,
)

from service.domain.auth.token import (
    AdminUsers,
    TokenPayload,
)
from service.settings import AuthConfig


logger = getLogger(__name__)


class UnauthorizedException(Exception):
    status_code = status.HTTP_401_UNAUTHORIZED
    message = 'Invalid credentials'


class AuthService:
    UNAUTHORIZED = UnauthorizedException

    def __init__(self, config: AuthConfig) -> None:
        self._pub_key = config.auth_public_key
        self._private_key = config.auth_private_key

    def validate_token(self, token: Optional[str]) -> TokenPayload:
        if not token:
            raise self.UNAUTHORIZED
        clear_token = self._replace_prefix(token)
        payload = self._decode_token(clear_token)
        return payload

    def is_admin(self, token_payload: TokenPayload) -> bool:
        try:
            AdminUsers(token_payload.user)
            return True
        except ValueError:  # noqa: WPS329
            return False

    def _replace_prefix(self, token: str) -> str:
        return token.replace('Bearer ', '').strip()

    def _decode_token(self, token: str) -> TokenPayload:
        try:
            payload = jwt.decode(
                token,
                key=self._pub_key,
                algorithms=[constants.ALGORITHMS.RS256],
            )
            return TokenPayload.model_validate(payload)
        except Exception as exc:
            logger.exception('Malformed token')
            raise self.UNAUTHORIZED from exc
