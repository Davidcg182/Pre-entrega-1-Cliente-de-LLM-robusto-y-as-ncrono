"""El puerto: interfaz común a todos los proveedores.

Reparto de responsabilidades (y la razón de que la clase base no sea un simple
`ABC` con un método abstracto):

- **La base** posee la POLÍTICA: timeout, reintentos con backoff, clasificación
  de errores y contrato uniforme de salida. Se escribe una vez.
- **El adaptador** posee la TRADUCCIÓN: convertir `ChatMessage` al formato de
  su SDK y su respuesta de vuelta. No sabe nada de resiliencia.

Es el patrón Template Method, y es lo que evita que cada proveedor nuevo
reimplemente —de forma sutilmente distinta— su propio retry.
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from pydantic import SecretStr

from .errors import classify
from .schemas import ChatMessage, ErrorKind, ModelConfig, ModelResponse, StreamChunk, Usage

_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 8.0


def _backoff_delay(attempt: int) -> float:
    """Backoff exponencial con jitter completo.

    El jitter no es un adorno: sin él, N clientes que reciben el mismo 429 a la
    vez reintentan a la vez, y se reconstruye el pico que causó el 429. Se
    llama «thundering herd» y el jitter es lo único que lo rompe.
    """
    ceiling = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (2 ** (attempt - 1)))
    return random.uniform(0.0, ceiling)


class BaseLLMClient(ABC):
    """Interfaz común. Los adaptadores sólo implementan los dos métodos `_raw`."""

    def __init__(self, config: ModelConfig, api_key: SecretStr | str,
                 base_url: str | None = None) -> None:
        self.config = config
        # La clave se guarda ENVUELTA y se desenvuelve sólo al construir el
        # cliente del SDK. Así no puede aparecer en un repr, un log ni un
        # traceback de este objeto por descuido de nadie.
        self._api_key = api_key if isinstance(api_key, SecretStr) else SecretStr(api_key)
        self._base_url = base_url

    # ------------------------------------------------------------------
    # Contrato que implementa cada proveedor. Ambos DEBEN dejar escapar las
    # excepciones de su SDK: clasificarlas es trabajo de la base.
    # ------------------------------------------------------------------

    @abstractmethod
    async def _generate_raw(self, messages: Sequence[ChatMessage]) -> tuple[str, Usage]: ...

    @abstractmethod
    def _stream_raw(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]: ...

    async def aclose(self) -> None:
        """Cerrar el pool de conexiones del SDK, si lo hay."""

    # ------------------------------------------------------------------
    # Política común
    # ------------------------------------------------------------------

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Llamada completa. Nunca lanza: devuelve `ModelResponse.ok == False`."""
        started = time.perf_counter()
        last_error = None

        for attempt in range(1, self.config.max_retries + 2):
            try:
                async with asyncio.timeout(self.config.timeout_s):
                    content, usage = await self._generate_raw(messages)
            except Exception as exc:  # noqa: BLE001 - la frontera absorbe todo
                last_error = classify(exc, self.config.provider, attempts=attempt)
                if last_error.retryable and attempt <= self.config.max_retries:
                    await asyncio.sleep(_backoff_delay(attempt))
                    continue
                break
            else:
                return ModelResponse(
                    ok=True,
                    provider=self.config.provider,
                    model=self.config.model,
                    content=content,
                    usage=usage,
                    latency_s=round(time.perf_counter() - started, 3),
                )

        assert last_error is not None
        return ModelResponse(
            ok=False,
            provider=self.config.provider,
            model=self.config.model,
            latency_s=round(time.perf_counter() - started, 3),
            error=last_error,
        )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """Emite tokens conforme llegan. Nunca lanza: el fallo llega como chunk.

        El timeout NO es total, es de INACTIVIDAD: se rearma en cada token. Un
        límite total aquí penalizaría a la respuesta larga y sana por el mismo
        motivo que a la conexión muerta, que son problemas distintos.

        No se reintenta un stream ya empezado: el usuario ya vio texto en
        pantalla y reintentar lo duplicaría. El retry vive en `generate()`.
        """
        started = time.perf_counter()
        ttft_s: float | None = None

        # El generador del proveedor se cierra en `finally`, SIEMPRE: si se deja
        # que lo recoja el recolector de basura, el `GeneratorExit` se inyecta
        # en un punto arbitrario del stream de httpx y revienta el cierre del
        # pool de conexiones. Cerrarlo aquí lo hace en un punto controlado,
        # tanto si el consumidor agota el stream como si abandona a mitad.
        source = self._stream_raw(messages)
        iterator = source.__aiter__()
        try:
            while True:
                try:
                    async with asyncio.timeout(self.config.stream_idle_timeout_s):
                        token = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                except Exception as exc:  # noqa: BLE001
                    yield StreamChunk(done=True, error=classify(exc, self.config.provider))
                    return

                if ttft_s is None:
                    ttft_s = round(time.perf_counter() - started, 3)
                    yield StreamChunk(text=token, ttft_s=ttft_s)
                else:
                    yield StreamChunk(text=token)

            yield StreamChunk(done=True)
        finally:
            if (aclose := getattr(source, "aclose", None)) is not None:
                await aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.aclose()
