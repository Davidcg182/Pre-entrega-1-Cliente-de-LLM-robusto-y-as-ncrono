"""Adaptador de pruebas: simula un LLM sin red, sin clave y sin coste.

No es relleno. Permite dos cosas que con un proveedor real son caras o
imposibles: ejecutar el repositorio sin credenciales, y **provocar un 429 a
voluntad** para demostrar que el backoff funciona (`fail_times`).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Sequence

from ..base import BaseLLMClient
from ..schemas import ChatMessage, ModelConfig, Usage

_RESPUESTA = (
    "La entropía es una medida del desorden o, con más precisión, del número de "
    "microestados compatibles con un estado macroscópico dado. En teoría de la "
    "información mide la incertidumbre media de una fuente: cuántos bits hacen "
    "falta, en promedio, para describir su próximo símbolo."
)


class _SimulatedRateLimit(Exception):
    """Imita la forma de un `RateLimitError` de SDK: se clasifica por status_code."""

    status_code = 429


class FakeClient(BaseLLMClient):
    def __init__(
        self,
        config: ModelConfig,
        api_key: str = "",
        base_url: str | None = None,
        *,
        fail_times: int = 0,
        latency_s: float = 0.4,
    ) -> None:
        super().__init__(config, api_key, base_url)
        self._remaining_failures = fail_times
        self._latency_s = latency_s

    def _maybe_fail(self) -> None:
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise _SimulatedRateLimit("simulated rate limit (429)")

    async def _generate_raw(self, messages: Sequence[ChatMessage]) -> tuple[str, Usage]:
        await asyncio.sleep(self._latency_s)
        self._maybe_fail()
        prompt_chars = sum(len(m.content) for m in messages)
        return _RESPUESTA, Usage(
            prompt_tokens=prompt_chars // 4,
            completion_tokens=len(_RESPUESTA) // 4,
        )

    async def _stream_raw(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        await asyncio.sleep(self._latency_s)
        self._maybe_fail()
        for palabra in _RESPUESTA.split(" "):
            await asyncio.sleep(random.uniform(0.01, 0.05))
            yield palabra + " "
