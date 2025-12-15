from fastapi import (
    Request,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import (
    RequestValidationError,
    ResponseValidationError,
)
from fastapi.responses import ORJSONResponse

from service.domain.internal.errors.schemas import ErrorMessage


def error_response(
    status_code: int,
    errors: list[ErrorMessage],
) -> ORJSONResponse:
    return ORJSONResponse(
        status_code=status_code,
        content={'errors': jsonable_encoder(errors)},
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError | ResponseValidationError,
) -> ORJSONResponse:
    """Перехватчик ошибок валидации Pydantic"""
    validation_errors = [
        ErrorMessage.from_pydantic_error(error) for error in exc.errors()
    ]
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    if isinstance(exc, RequestValidationError):
        status_code = status.HTTP_400_BAD_REQUEST
    return error_response(
        status_code=status_code,
        errors=validation_errors,
    )


async def model_exception_handler(
    request: Request,
    exc: Exception,
) -> ORJSONResponse:
    """Перехватчик общих ошибок"""
    return error_response(
        status_code=status.HTTP_403_FORBIDDEN,
        errors=[ErrorMessage(message=str(exc))],
    )
