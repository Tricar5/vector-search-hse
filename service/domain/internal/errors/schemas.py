from typing import Any

from pydantic import BaseModel


class ErrorMessage(BaseModel):
    message: str

    @classmethod
    def from_pydantic_error(
        cls,
        err: dict[str, Any],
    ) -> 'ErrorMessage':
        return cls(
            message=(
                f'Validation error: {err.get("msg")}. '
                f'Loc: {err.get("loc")}. '
                f'Type: {err.get("type")}.'
            ),
        )
