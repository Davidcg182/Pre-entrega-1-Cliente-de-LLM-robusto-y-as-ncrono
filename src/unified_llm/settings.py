"""Configuración del proceso, validada al arrancar.

Qué problema resuelve, y por qué no basta con `os.getenv`:

1. **`os.getenv` devuelve `str | None` y no valida nada.** Un `LLM_MAX_TOKENS=abc`
   revienta con un `ValueError` sin contexto en el primer uso, no al arrancar.
2. **Una API key en un `str` corriente se filtra sola.** Acaba en un `repr()`, en
   un log de excepción, en un `print` de depuración o en un reporte de Sentry sin
   que nadie lo haya decidido. `SecretStr` la envuelve: se muestra como
   `SecretStr('**********')` y sólo entrega su valor a quien lo pide
   explícitamente con `.get_secret_value()`.

La diferencia entre las dos no es cosmética: la primera es comodidad, la segunda
es que el secreto deja de viajar en claro por sitios que no controlas.
"""

from __future__ import annotations

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .schemas import ModelConfig, Provider

_DEFAULT_MODEL: dict[Provider, str] = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-sonnet-5",
    Provider.FAKE: "fake-1",
}


class Settings(BaseSettings):
    """Entorno completo del proceso. Falla al construirse, no al usarse."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_provider: Provider = Provider.FAKE
    llm_model: str | None = None
    """Vacío o ausente = el modelo por defecto del proveedor."""

    # `SecretStr`: el valor no aparece en repr, str, logs ni tracebacks.
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    openai_base_url: str | None = None
    anthropic_base_url: str | None = None

    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0, le=32_000)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_stream_idle_timeout_s: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def _blank_key_is_absent(cls, value: object) -> object:
        """`OPENAI_API_KEY=` en el `.env` significa «no la tengo», no «es vacía».
        Sin esto, la comprobación de abajo pasaría y el fallo llegaría como un
        401 del proveedor: un viaje de red para descubrir algo que ya sabíamos."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_model", "openai_base_url", "anthropic_base_url", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _require_key_of_active_provider(self) -> "Settings":
        """Sólo se exige la clave del proveedor ACTIVO.

        Pedir las dos obligaría a tener cuenta en ambos para probar uno, y
        exigir credenciales que no se van a usar empuja a rellenarlas con
        basura, que es peor que no tenerlas."""
        if self.llm_provider is Provider.FAKE:
            return self
        if self.api_key is None:
            variable = f"{self.llm_provider.value.upper()}_API_KEY"
            raise ValueError(
                f"Falta {variable}: es la clave del proveedor activo "
                f"(LLM_PROVIDER={self.llm_provider.value}). "
                f"Copia .env.example a .env y rellénala."
            )
        return self

    # ------------------------------------------------------------------

    @property
    def api_key(self) -> SecretStr | None:
        """La clave del proveedor activo, todavía envuelta."""
        return {
            Provider.OPENAI: self.openai_api_key,
            Provider.ANTHROPIC: self.anthropic_api_key,
            Provider.FAKE: SecretStr(""),
        }[self.llm_provider]

    @property
    def base_url(self) -> str | None:
        return {
            Provider.OPENAI: self.openai_base_url,
            Provider.ANTHROPIC: self.anthropic_base_url,
            Provider.FAKE: None,
        }[self.llm_provider]

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            provider=self.llm_provider,
            model=self.llm_model or _DEFAULT_MODEL[self.llm_provider],
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            timeout_s=self.llm_timeout_s,
            stream_idle_timeout_s=self.llm_stream_idle_timeout_s,
            max_retries=self.llm_max_retries,
        )
