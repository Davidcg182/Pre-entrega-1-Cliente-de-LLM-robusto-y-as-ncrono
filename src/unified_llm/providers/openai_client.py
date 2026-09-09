"""Adaptador de OpenAI (SDK `openai`, cliente `AsyncOpenAI`).

Sirve también para cualquier API compatible con el protocolo de OpenAI vía
`base_url` — Gemini, Groq, Together, un vLLM propio. Ese es el rendimiento
práctico de programar contra un protocolo y no contra un proveedor.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from pydantic import SecretStr

from ..base import BaseLLMClient
from ..schemas import ChatMessage, ModelConfig, Usage


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: SecretStr | str,
                 base_url: str | None = None) -> None:
        super().__init__(config, api_key, base_url)

        # Import perezoso: permite usar el paquete con sólo un SDK instalado.
        from openai import AsyncOpenAI

        # `max_retries=0`: el retry lo gobierna la clase base. Dos capas de
        # reintento se multiplican (3 x 3 = 9 llamadas) y la de arriba pierde
        # el control del presupuesto de latencia.
        # `.get_secret_value()` en el único punto donde hace falta: el SDK.
        self._client = AsyncOpenAI(
            api_key=self._api_key.get_secret_value(), base_url=base_url, max_retries=0
        )

    def _payload(self, messages: Sequence[ChatMessage]) -> dict:
        return {
            "model": self.config.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    async def _generate_raw(self, messages: Sequence[ChatMessage]) -> tuple[str, Usage]:
        # `await` sobre el método ASÍNCRONO del cliente asíncrono. Usar aquí el
        # `OpenAI` síncrono es el error que el enunciado señala: bloquearía el
        # event loop entero durante toda la generación.
        completion = await self._client.chat.completions.create(**self._payload(messages))

        content = completion.choices[0].message.content or ""
        raw_usage = completion.usage
        usage = Usage(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
        )
        return content, usage

    async def _stream_raw(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            **self._payload(messages), stream=True
        )
        # `async with` sobre el AsyncStream: libera la conexión HTTP aunque el
        # consumidor deje de leer a mitad (un usuario que cierra la pestaña).
        async with stream:
            async for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if (text := getattr(delta, "content", None)):
                    yield text

    async def aclose(self) -> None:
        await self._client.close()
