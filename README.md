# Unified Async LLM Client

Capa de abstracción asíncrona sobre proveedores de LLM (OpenAI y Anthropic) con validación
de esquemas, streaming de tokens y manejo de errores no propagantes.

**Pre-entrega 1 · Módulo 1 — Coderhouse AI Engineering.**

---

## El problema

Hablar con dos proveedores de LLM parece lo mismo y no lo es. Anthropic recibe el *system
prompt* como parámetro de nivel superior y no como un mensaje con `role="system"`; exige
`max_tokens`; y su stream emite eventos tipados en lugar de `choices[].delta`. Sin una capa
intermedia, esas diferencias se filtran al código de negocio y cambiar de proveedor deja de
ser una decisión de configuración para convertirse en una refactorización.

A eso se suma que una llamada a un LLM es una dependencia de red lenta, cara y con cuota: un
`429` o una conexión colgada no deberían tumbar el proceso que los recibe.

Este repositorio resuelve las dos cosas: **una interfaz, varios proveedores**, y **el fallo
como dato en lugar de como excepción**.

---

## Arquitectura

Ports & adapters. El puerto es `BaseLLMClient`; cada proveedor es un adaptador.

```
                    ┌─────────────────────┐
   configuración →  │  AsyncLLMManager    │  Factory + fachada. Es lo único
                    │  (composition root) │  que conoce el código de negocio.
                    └──────────┬──────────┘
                               │  elige por config, no por `if`
                    ┌──────────▼──────────┐
                    │   BaseLLMClient     │  POLÍTICA: timeout, retry con
                    │  (puerto abstracto) │  backoff+jitter, clasificación de
                    └──────────┬──────────┘  errores, contrato uniforme.
              ┌────────────────┼────────────────┐
      ┌───────▼──────┐ ┌───────▼───────┐ ┌──────▼──────┐
      │ OpenAIClient │ │AnthropicClient│ │ FakeClient  │  TRADUCCIÓN: sólo
      └──────────────┘ └───────────────┘ └─────────────┘  hablar con su SDK.
```

La base concentra la resiliencia y los adaptadores sólo traducen. Es un *Template Method*, y
es lo que impide que cada proveedor acabe con su propio `retry` sutilmente distinto.

```
src/unified_llm/
├── schemas.py     Pydantic: ChatMessage, ModelConfig, ModelResponse, StreamChunk, ErrorInfo
├── errors.py      Traducción de excepciones de SDK → taxonomía propia (ErrorKind)
├── base.py        BaseLLMClient: timeout, retry, contrato uniforme
├── manager.py     AsyncLLMManager: factory por configuración + from_env()
└── providers/     Adaptadores: openai · anthropic · fake
main.py            Script de validación (modo completo + streaming)
```

---

## Instalación

Requiere **Python 3.12+**.

```bash
python3 -m venv .venv && source .venv/bin/activate   # o: uv venv --python 3.12 .venv
pip install -r requirements.txt                      # o: uv pip install -r requirements.txt
cp .env.example .env
```

Dependencias: `openai`, `anthropic`, `pydantic`, `python-dotenv`.

---

## Variables de entorno

Todas viven en `.env` (que está en `.gitignore`; la plantilla versionada es `.env.example`).

| Variable | Obligatoria | Por defecto | Qué hace |
| --- | --- | --- | --- |
| `LLM_PROVIDER` | no | `fake` | `openai` · `anthropic` · `fake` |
| `LLM_MODEL` | no | según proveedor | `gpt-4o-mini` · `claude-sonnet-5` · `fake-1` |
| `OPENAI_API_KEY` | sí, si `LLM_PROVIDER=openai` | — | Clave de OpenAI |
| `ANTHROPIC_API_KEY` | sí, si `LLM_PROVIDER=anthropic` | — | Clave de Anthropic |
| `OPENAI_BASE_URL` | no | — | Endpoint alternativo compatible con OpenAI |
| `ANTHROPIC_BASE_URL` | no | — | Endpoint alternativo de Anthropic |
| `LLM_TEMPERATURE` | no | `0.7` | Validada en `[0, 2]` |
| `LLM_MAX_TOKENS` | no | `1024` | Validada `> 0` |
| `LLM_TIMEOUT_S` | no | `60` | Presupuesto total de una llamada no-streaming |
| `LLM_STREAM_IDLE_TIMEOUT_S` | no | `30` | Silencio máximo **entre** tokens |
| `LLM_MAX_RETRIES` | no | `2` | Reintentos ante fallo recuperable |

Una configuración inválida falla **al arrancar**, no en la primera petición:

```
$ LLM_TEMPERATURE=5 python main.py
✗ CONFIGURACIÓN inválida en el .env:
    temperature: Input should be less than or equal to 2 (recibido: 5.0)
```

---

## Ejecución

```bash
python main.py                    # usa el proveedor del .env
python main.py --provider fake    # sin claves, sin red, sin coste
python main.py --provider openai
python main.py --demo-errores     # además, fuerza un 429 y enseña el backoff
```

El script lanza la misma pregunta —«¿Qué es la entropía?»— en los dos modos exigidos:

Salida real contra la API (recogida completa en `salida-validacion.txt`):

```
====================================================================
2. MODO STREAMING — AsyncLLMManager(provider=openai, model='gemini-3.6-flash')
====================================================================
La entropía es la medida física que cuantifica el grado de desorden, aleatoriedad
y dispersión de la energía en un sistema. […]

  TTFT=6.595s · total=7.06s · 5 fragmentos
```

### Probar sin pagar

El SDK de OpenAI habla un **protocolo**, no sólo con OpenAI. Con una clave gratuita de
[Google AI Studio](https://aistudio.google.com/apikey) se ejecuta el mismo `OpenAIClient`
—mismo `AsyncOpenAI`, mismo `async for` del streaming— contra Gemini:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gemini-3.6-flash
OPENAI_API_KEY=<tu clave>
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

**Si el proveedor es un modelo de razonamiento**, dos parámetros por defecto se
quedan cortos, y ambos fallos son confusos porque no parecen fallos:

- `max_tokens` se consume **pensando**. Con 512, la respuesta llegaba truncada con
  `finish_reason=length` tras una sola frase, y parte del razonamiento se filtraba
  al contenido. El presupuesto de salida no es sólo lo que lees.
- El **TTFT sube**: el modelo piensa antes de emitir el primer token. Con un timeout
  de inactividad de 15 s se cortaban streams perfectamente sanos. Por eso el valor
  por defecto es 30 s.

---

## Decisiones de diseño

**El fallo se devuelve, no se lanza.** `generate()` siempre devuelve un `ModelResponse` con
`ok: bool`; `stream()` emite un `StreamChunk` final con `error` en vez de lanzar dentro del
`async for` de quien consume. Coste asumido: quien llama debe mirar `ok`. A cambio, un fallo
de un proveedor no puede romper el bucle de arriba, que es exactamente el criterio de
resiliencia del enunciado. La alternativa —propagar excepciones tipadas— es más honesta para
una librería de terceros y obliga a un `try/except` en cada punto de llamada.

**El timeout del streaming es de inactividad, no total.** Un límite total castiga por igual a
la conexión muerta y a la respuesta larga y sana, que son problemas distintos: la latencia de
un LLM es proporcional a los tokens que genera. Se mide, en cambio, el **TTFT** (lo que
percibe el usuario) y se vigila el silencio entre tokens, rearmando el reloj en cada uno.

**Backoff exponencial con jitter, y sólo para lo reintentable.** Un `401` no mejora por
insistir y un `400` tampoco: sólo se reintentan `RATE_LIMIT`, `TIMEOUT`, `CONNECTION` y
`PROVIDER`. El jitter no es cosmético: sin él, N clientes que reciben el mismo `429`
reintentan a la vez y reconstruyen el pico que lo provocó.

**Un solo nivel de reintentos.** Los SDKs se instancian con `max_retries=0` y el retry lo
gobierna la clase base. Dos capas se multiplican (3 × 3 = 9 llamadas) y quien está arriba
pierde el control de su presupuesto de latencia.

**Los errores de configuración no son errores de ejecución.** Una clave ausente o una
temperatura fuera de rango matan el proceso al arrancar, con mensaje legible y código de
salida 2. Un fallo de despliegue debe verse en el despliegue, no dentro de la petición de un
usuario cinco minutos después.

**`FakeClient` no es relleno.** Permite ejecutar el repositorio sin credenciales y provocar
un `429` a voluntad (`fail_times`), que es la única forma barata de demostrar que el backoff
hace lo que dice.

---

## Limitaciones conocidas

- **No hay rate limiting.** El retry reacciona al `429`; no lo previene. Un control de caudal
  real necesita un *token bucket* sobre RPM **y** TPM, y el TPM depende del tamaño del prompt
  y de la respuesta.
- **No hay límite de concurrencia** en el manager: si se lanzan 10.000 llamadas, se lanzan
  10.000. El control de admisión (`asyncio.Semaphore`) se trabajó en la práctica de la unidad
  1.1 y no se integró aquí porque el enunciado no lo pide.
- **Sin tests automatizados.** `FakeClient` está diseñado para hacerlos triviales, pero el
  entregable no los exige.
- El conteo de tokens en `FakeClient` es una aproximación por caracteres, no un tokenizador.
- **Dependencias fijadas a `openai<2` y `anthropic<1`** a propósito. Las versiones
  mayores actuales arrastran `httpx2`, y `httpcore2 2.12.0` falla al finalizar el
  cuerpo de una respuesta en streaming: ensucia stderr con un `RuntimeError:
  generator didn't stop after athrow()` al apagar el event loop. La salida es
  correcta, pero el ruido es indistinguible de un crash. Comprobado que no ocurre
  con la pila estable, y que no procede de este código: el SDK usado directamente,
  sin esta capa, lo reproduce igual.
