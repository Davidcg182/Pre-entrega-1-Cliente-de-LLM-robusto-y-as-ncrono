"""Contratos de datos del cliente unificado.

Se definen ANTES que los clientes a propósito: son la frontera que impide que
un `dict` anidado del SDK de un proveedor se escape hacia el resto del sistema.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FAKE = "fake"


Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """Un turno de conversación, en formato neutro respecto al proveedor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: Annotated[str, Field(min_length=1)]


class ModelConfig(BaseModel):
    """Configuración validada de una llamada.

    Los rangos no son decorativos: `temperature` fuera de [0, 2] es un 400 del
    proveedor: un viaje de red y una espera para descubrir un error que se
    conocía antes de salir del proceso. Validar aquí es fallar barato.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str = Field(min_length=1)

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=32_000)

    # --- Resiliencia (ver README §Resiliencia) ---
    timeout_s: float = Field(default=30.0, gt=0)
    """Presupuesto total de una llamada NO-streaming."""

    stream_idle_timeout_s: float = Field(default=15.0, gt=0)
    """Silencio máximo tolerado ENTRE tokens. No es un límite total: una
    respuesta larga y sana puede tardar minutos y no debe cortarse por eso."""

    max_retries: int = Field(default=2, ge=0, le=5)


class ErrorKind(StrEnum):
    """Taxonomía de fallos, independiente del proveedor.

    Existe para que quien llama decida por CLASE de fallo y no por el nombre de
    la excepción de un SDK concreto: `AUTH` no se reintenta nunca por mucho que
    insistas; `RATE_LIMIT` sí, pero sólo con backoff.
    """

    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    BAD_REQUEST = "bad_request"
    PROVIDER = "provider"
    UNKNOWN = "unknown"


class ErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ErrorKind
    message: str
    provider: Provider
    retryable: bool
    attempts: int = 1
    status_code: int | None = None


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelResponse(BaseModel):
    """Contrato uniforme: SIEMPRE se devuelve esto, haya ido bien o mal.

    Es la política de errores A del módulo (ver `modules/01-.../NOTES.md`): el
    fallo se absorbe y se devuelve como dato, no como excepción. Coste asumido:
    quien llama debe mirar `ok`; a cambio, un fallo de un proveedor no puede
    tumbar el bucle de arriba.
    """

    model_config = ConfigDict(frozen=True)

    ok: bool
    provider: Provider
    model: str
    content: str = ""
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    error: ErrorInfo | None = None


class StreamChunk(BaseModel):
    """Un fragmento del stream.

    El stream también respeta el contrato uniforme: un fallo a mitad de emisión
    llega como un chunk final con `error`, no como una excepción lanzada dentro
    del `async for` de quien consume.
    """

    model_config = ConfigDict(frozen=True)

    text: str = ""
    done: bool = False
    error: ErrorInfo | None = None
    ttft_s: float | None = None
    """Time To First Token. Sólo viaja en el primer chunk."""
