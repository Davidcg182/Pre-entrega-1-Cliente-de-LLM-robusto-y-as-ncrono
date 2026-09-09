"""Script de validación del Unified Async LLM Client.

Ejecuta la misma pregunta en los dos modos que exige el enunciado —completo y
streaming— contra el proveedor que diga `LLM_PROVIDER` en el `.env`.

    python main.py                    # usa el .env
    python main.py --provider fake    # sin claves ni coste
    python main.py --demo-errores     # fuerza un 429 y enseña el backoff
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from unified_llm import (  # noqa: E402
    AsyncLLMManager,
    ChatMessage,
    MissingAPIKeyError,
    ModelConfig,
    Provider,
    Settings,
)

PREGUNTA = "¿Qué es la entropía?"

MENSAJES = [
    ChatMessage(role="system", content="Eres un divulgador científico. Responde en 3 frases."),
    ChatMessage(role="user", content=PREGUNTA),
]


def titulo(texto: str) -> None:
    print(f"\n{'=' * 68}\n{texto}\n{'=' * 68}")


def mostrar_configuracion(settings: Settings) -> None:
    """Evidencia de que la clave no se filtra ni cuando se imprime a propósito."""
    titulo("0. CONFIGURACIÓN VALIDADA")
    print(f"  proveedor         : {settings.llm_provider.value}")
    print(f"  modelo            : {settings.to_model_config().model}")
    print(f"  openai_api_key    : {settings.openai_api_key!r}")
    print(f"  anthropic_api_key : {settings.anthropic_api_key!r}")
    print(f"  temperature       : {settings.llm_temperature} · max_tokens: {settings.llm_max_tokens}")
    print("\n  Arriba no hay ninguna clave: SecretStr sólo la entrega a quien llama")
    print("  explícitamente a .get_secret_value(), y eso ocurre en un único punto,")
    print("  al construir el cliente del SDK.")


async def modo_completo(manager: AsyncLLMManager) -> None:
    titulo(f"1. MODO COMPLETO — {manager!r}")
    respuesta = await manager.generate(MENSAJES)

    if not respuesta.ok:
        assert respuesta.error is not None
        print(f"✗ FALLO CONTROLADO [{respuesta.error.kind}] tras "
              f"{respuesta.error.attempts} intento(s): {respuesta.error.message}")
        print("  El programa sigue vivo: eso es el criterio de resiliencia.")
        return

    print(respuesta.content)
    print(f"\n  latencia={respuesta.latency_s}s · "
          f"tokens in/out={respuesta.usage.prompt_tokens}/{respuesta.usage.completion_tokens}")


async def modo_streaming(manager: AsyncLLMManager) -> None:
    titulo(f"2. MODO STREAMING — {manager!r}")
    inicio = time.perf_counter()
    ttft = None
    n_chunks = 0

    async for chunk in manager.stream(MENSAJES):
        if chunk.error is not None:
            print(f"\n✗ FALLO CONTROLADO EN STREAM [{chunk.error.kind}]: {chunk.error.message}")
            return
        if chunk.ttft_s is not None:
            ttft = chunk.ttft_s
        if chunk.text:
            n_chunks += 1
            print(chunk.text, end="", flush=True)

    total = time.perf_counter() - inicio
    print(f"\n\n  TTFT={ttft}s · total={total:.2f}s · {n_chunks} fragmentos")
    print("  El TTFT es lo que percibe el usuario; el total, lo que cuesta la respuesta.")


async def demo_errores() -> None:
    """Dos 429 seguidos contra un proveedor simulado: el backoff los absorbe."""
    titulo("3. RESILIENCIA — 429 simulado y reintento con backoff + jitter")

    config = ModelConfig(provider=Provider.FAKE, model="fake-1", max_retries=2)
    async with AsyncLLMManager(config, fail_times=2, latency_s=0.1) as manager:
        inicio = time.perf_counter()
        respuesta = await manager.generate(MENSAJES)
        print(f"ok={respuesta.ok} · recuperado en {time.perf_counter() - inicio:.2f}s "
              f"(incluye las esperas del backoff)")

    print("\n  Y ahora el mismo fallo sin reintentos suficientes:")
    config_sin_retry = ModelConfig(provider=Provider.FAKE, model="fake-1", max_retries=0)
    async with AsyncLLMManager(config_sin_retry, fail_times=2, latency_s=0.1) as manager:
        respuesta = await manager.generate(MENSAJES)
        assert respuesta.error is not None
        print(f"  ok={respuesta.ok} · kind={respuesta.error.kind} · "
              f"retryable={respuesta.error.retryable} · sin excepción propagada")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Validación del cliente unificado.")
    parser.add_argument("--provider", choices=[p.value for p in Provider],
                        help="Sobrescribe LLM_PROVIDER del .env.")
    parser.add_argument("--demo-errores", action="store_true",
                        help="Añade la demostración de retry con backoff.")
    args = parser.parse_args()

    load_dotenv()

    # La sobrescritura viaja al constructor de Settings, no al entorno del
    # proceso: pydantic-settings lee el fichero .env, así que manipular
    # os.environ no bastaría para cambiar de proveedor.
    sobrescrituras = {"llm_provider": args.provider, "llm_model": None} if args.provider else {}

    # Los errores de CONFIGURACIÓN se tratan aquí, y no dentro del cliente,
    # porque no son fallos de una llamada: son un despliegue mal hecho. Deben
    # matar el proceso al arrancar (fail fast), pero con un mensaje legible en
    # lugar de un traceback.
    try:
        settings = Settings(**sobrescrituras)
        mostrar_configuracion(settings)
        manager = AsyncLLMManager.from_settings(settings)
    except MissingAPIKeyError as exc:
        print(f"✗ CONFIGURACIÓN: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except ValidationError as exc:
        print("✗ CONFIGURACIÓN inválida en el .env:", file=sys.stderr)
        for error in exc.errors():
            campo = ".".join(str(part) for part in error["loc"]) or "(configuración)"
            detalle = error["msg"].removeprefix("Value error, ")
            print(f"    {campo}: {detalle}", file=sys.stderr)
        raise SystemExit(2) from None

    async with manager:
        await modo_completo(manager)
        await modo_streaming(manager)

    if args.demo_errores:
        await demo_errores()


if __name__ == "__main__":
    asyncio.run(main())
