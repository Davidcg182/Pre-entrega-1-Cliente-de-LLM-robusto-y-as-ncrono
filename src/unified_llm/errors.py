"""Traducción de excepciones de SDK a la taxonomía propia.

Por qué existe este módulo: si el `AsyncLLMManager` hiciera
`except openai.RateLimitError`, la abstracción sería falsa — el código de
arriba seguiría acoplado a un proveedor concreto. Aquí se traduce una vez y el
resto del sistema sólo conoce `ErrorKind`.

No se importan los SDKs a propósito: se clasifica por nombre de excepción y
código HTTP. Así el paquete arranca aunque sólo esté instalado uno de los dos
proveedores, y `openai` y `anthropic` comparten esa jerarquía de nombres por
construir ambos sobre httpx.
"""

from __future__ import annotations

import asyncio

from .schemas import ErrorInfo, ErrorKind, Provider

_BY_EXCEPTION_NAME: dict[str, ErrorKind] = {
    "AuthenticationError": ErrorKind.AUTH,
    "PermissionDeniedError": ErrorKind.AUTH,
    "RateLimitError": ErrorKind.RATE_LIMIT,
    "APITimeoutError": ErrorKind.TIMEOUT,
    "APIConnectionError": ErrorKind.CONNECTION,
    "APIConnectionTimeoutError": ErrorKind.TIMEOUT,
    "BadRequestError": ErrorKind.BAD_REQUEST,
    "UnprocessableEntityError": ErrorKind.BAD_REQUEST,
    "NotFoundError": ErrorKind.BAD_REQUEST,
    "InternalServerError": ErrorKind.PROVIDER,
    "APIStatusError": ErrorKind.PROVIDER,
}

_BY_STATUS_CODE: dict[int, ErrorKind] = {
    400: ErrorKind.BAD_REQUEST,
    401: ErrorKind.AUTH,
    403: ErrorKind.AUTH,
    404: ErrorKind.BAD_REQUEST,
    408: ErrorKind.TIMEOUT,
    422: ErrorKind.BAD_REQUEST,
    429: ErrorKind.RATE_LIMIT,
}

# Reintentar un 401 es insistir en una contraseña equivocada; reintentar un 400
# es repetir una petición mal formada. Sólo se reintenta lo que puede cambiar
# por sí solo con el paso del tiempo.
_RETRYABLE: frozenset[ErrorKind] = frozenset(
    {ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT, ErrorKind.CONNECTION, ErrorKind.PROVIDER}
)


def classify(exc: BaseException, provider: Provider, attempts: int = 1) -> ErrorInfo:
    """Convierte cualquier excepción en un `ErrorInfo` accionable."""
    status_code = getattr(exc, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        kind = ErrorKind.TIMEOUT
    elif (name_kind := _BY_EXCEPTION_NAME.get(type(exc).__name__)) is not None:
        kind = name_kind
    elif status_code is not None:
        kind = _BY_STATUS_CODE.get(status_code, ErrorKind.PROVIDER)
    elif isinstance(exc, (ConnectionError, OSError)):
        kind = ErrorKind.CONNECTION
    else:
        kind = ErrorKind.UNKNOWN

    # Un 429 declarado como APIStatusError genérico debe ganar por código.
    if status_code is not None and status_code in _BY_STATUS_CODE:
        kind = _BY_STATUS_CODE[status_code]

    return ErrorInfo(
        kind=kind,
        message=f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__,
        provider=provider,
        retryable=kind in _RETRYABLE,
        attempts=attempts,
        status_code=status_code,
    )
