"""`AsyncLLMManager`: punto único de entrada y composition root.

Qué problema resuelve: sin él, cada punto del sistema que quiera hablar con un
LLM tiene que saber qué proveedor hay configurado, de qué variable de entorno
sale su clave y cómo se construye su cliente. Ese conocimiento se duplica y
diverge. El manager lo concentra en un sitio y expone `generate` / `stream`.

Es un Factory: `provider` es un dato de configuración, no una rama `if` repartida
por el código. Añadir un proveedor nuevo es registrar una entrada en `_REGISTRY`
y escribir su adaptador; ningún consumidor cambia. (Open/Closed, en la práctica.)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence

from pydantic import SecretStr

from .base import BaseLLMClient
from .providers import AnthropicClient, FakeClient, OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider, StreamChunk
from .settings import Settings

_REGISTRY: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
    Provider.FAKE: FakeClient,
}

_API_KEY_ENV: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.FAKE: "",
}

_BASE_URL_ENV: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_BASE_URL",
    Provider.ANTHROPIC: "ANTHROPIC_BASE_URL",
    Provider.FAKE: "",
}

_DEFAULT_MODEL: dict[Provider, str] = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-sonnet-5",
    Provider.FAKE: "fake-1",
}


class MissingAPIKeyError(RuntimeError):
    """Se lanza al construir, no al llamar.

    Es deliberado: una clave ausente es un error de despliegue, no un fallo de
    ejecución. Debe reventar al arrancar el proceso —fail fast— y no cinco
    minutos después, dentro de una petición de usuario.
    """


class AsyncLLMManager:
    """Fachada asíncrona sobre el proveedor configurado."""

    def __init__(self, config: ModelConfig, api_key: SecretStr | str | None = None,
                 base_url: str | None = None, **client_kwargs) -> None:
        self.config = config

        key = api_key if api_key is not None else self._key_from_env(config.provider)
        url = base_url if base_url is not None else self._url_from_env(config.provider)

        client_cls = _REGISTRY[config.provider]
        self._client: BaseLLMClient = client_cls(config, key, url, **client_kwargs)

    # ------------------------------------------------------------------
    # Construcción
    # ------------------------------------------------------------------

    @staticmethod
    def _key_from_env(provider: Provider) -> SecretStr:
        """Camino de construcción directa. `from_env()` usa `Settings`, que
        además valida tipos y rangos; esto es el mínimo para quien instancia el
        manager a mano con una config propia."""
        if provider is Provider.FAKE:
            return SecretStr("")
        var = _API_KEY_ENV[provider]
        key = os.getenv(var, "").strip()
        if not key:
            raise MissingAPIKeyError(
                f"Falta la variable de entorno {var} para el proveedor '{provider}'. "
                f"Copia .env.example a .env y rellénala."
            )
        return SecretStr(key)

    @staticmethod
    def _url_from_env(provider: Provider) -> str | None:
        var = _BASE_URL_ENV.get(provider, "")
        return (os.getenv(var, "").strip() or None) if var else None

    @classmethod
    def from_settings(cls, settings: Settings, **overrides) -> "AsyncLLMManager":
        """Construye a partir de una configuración ya validada.

        Se separa de `from_env()` para que quien llama pueda construir el
        `Settings` a su manera —con sobrescrituras, o en un test— sin que este
        método vuelva a leer el entorno por su cuenta. Un `Settings` por
        proceso, y una sola fuente de verdad."""
        return cls(
            settings.to_model_config(),
            api_key=settings.api_key,
            base_url=settings.base_url,
            **overrides,
        )

    @classmethod
    def from_env(cls, **overrides) -> "AsyncLLMManager":
        """Atajo: lee y valida el entorno, y construye.

        Toda la validación ocurre aquí, antes de la primera petición: tipos,
        rangos, presencia de la clave del proveedor activo. Y la clave llega
        envuelta en `SecretStr`, no como `str` suelto."""
        return cls.from_settings(Settings(), **overrides)

    # ------------------------------------------------------------------
    # Uso
    # ------------------------------------------------------------------

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages)

    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamChunk]:
        return self._client.stream(messages)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncLLMManager":
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"AsyncLLMManager(provider={self.config.provider}, model={self.config.model!r})"
