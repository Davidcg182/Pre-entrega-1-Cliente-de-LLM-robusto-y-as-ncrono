"""Adaptador de Anthropic (SDK `anthropic`, cliente `AsyncAnthropic`).

Aquí se ve para qué sirve realmente la capa de abstracción. Anthropic NO es
OpenAI con otro nombre:

- El `system prompt` es un parámetro de nivel superior, no un mensaje con
  `role="system"`. Mandarlo dentro de `messages` es un 400.
- `max_tokens` es OBLIGATORIO, no opcional.
- El streaming emite eventos tipados (`content_block_delta`), no `choices`.

Sin esta capa, esas tres diferencias se filtran al código de negocio.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from pydantic import SecretStr

from ..base import BaseLLMClient
from ..schemas import ChatMessage, ModelConfig, Usage


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: SecretStr | str,
                 base_url: str | None = None) -> None:
        super().__init__(config, api_key, base_url)

        from anthropic import AsyncAnthropic

        # `.get_secret_value()` en el único punto donde hace falta: el SDK.
        self._client = AsyncAnthropic(
            api_key=self._api_key.get_secret_value(), base_url=base_url, max_retries=0
        )

    def _payload(self, messages: Sequence[ChatMessage]) -> dict:
        system_parts = [m.content for m in messages if m.role == "system"]
        turns = [m.model_dump() for m in messages if m.role != "system"]

        payload = {
            "model": self.config.model,
            "messages": turns,
            "temperature": min(self.config.temperature, 1.0),  # Anthropic acota en 1.0
            "max_tokens": self.config.max_tokens,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        return payload

    async def _generate_raw(self, messages: Sequence[ChatMessage]) -> tuple[str, Usage]:
        message = await self._client.messages.create(**self._payload(messages))

        content = "".join(block.text for block in message.content if block.type == "text")
        usage = Usage(
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )
        return content, usage

    async def _stream_raw(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        async with self._client.messages.stream(**self._payload(messages)) as stream:
            async for text in stream.text_stream:
                yield text

    async def aclose(self) -> None:
        await self._client.close()
