from typing import (
    Any,
    Callable,
    Type,
)

from fastapi import Request
from fastapi.exceptions import (
    RequestValidationError,
    ResponseValidationError,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from service.domain.internal.errors.handlers import (
    model_exception_handler,
    validation_exception_handler,
)


class ServiceExceptionHandler(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    exc: Type[Exception] | int = Field(description='Класс ошибок')
    exc_handler: Callable[[Request, Any], Any] = Field(description='Обработчик ошибки')


DEFAULT_EXCEPTION_HANDLERS = (
    ServiceExceptionHandler(
        exc=Exception,
        exc_handler=model_exception_handler,
    ),
    ServiceExceptionHandler(
        exc=RequestValidationError,
        exc_handler=validation_exception_handler,
    ),
    ServiceExceptionHandler(
        exc=ResponseValidationError,
        exc_handler=validation_exception_handler,
    ),
)
