# Pre-entrega 5: Agente de razonamiento cíclico con memoria persistente

Agente de inventario de **Supermercados del Valle** hecho con **LangGraph** y **Claude** (Anthropic), en Python 3.12 con `asyncio`. Responde preguntas sobre stock y pedidos a proveedores usando herramientas propias, en un ciclo **ReAct**: razona, usa una herramienta, observa el resultado (`ToolMessage`) y repite hasta poder responder. La memoria de cada conversación se guarda **en disco** (SQLite) y se identifica con un `thread_id`.

## Cómo verificar cada criterio

Todas las evidencias salen de **una ejecución real** de `python main.py` con Claude y están en el repo:

| Archivo | Qué contiene |
|---|---|
| [`traza.log`](traza.log) | La traza ReAct legible, turno por turno, con el tipo de mensaje de cada paso. |
| [`traza.json`](traza.json) | La misma traza en formato estructurado. |
| [`ejecucion.log`](ejecucion.log) | Los logs del agente: cada decisión del LLM, su duración y qué se recuperó del checkpointer. |
| [`memoria.db`](memoria.db) | La base SQLite que generó el `AsyncSqliteSaver` en esa ejecución, con los checkpoints de `sesion-1` y `sesion-2`. |

Para reproducirlas, seguí la [Instalación](#instalación) y corré `python main.py`. Se vuelven a generar los cuatro archivos.

### 1. Autonomía y tool calling

El agente decide solo si usa una herramienta. El código no tiene ningún `if/else` que elija la herramienta según la pregunta: la arista `tools_condition` ([`agent.py` L74](agent.py#L74)) solo mira si el LLM devolvió `tool_calls`.

**Dónde verlo** en [`ejecucion.log`](ejecucion.log): para la misma lógica, el LLM a veces pide herramientas y a veces responde directo, según la pregunta:

```text
INFO main: thread_id=sesion-1 | mensajes recuperados de memoria.db: 0 | pregunta: ¿Qué producto tiene menos stock y cuándo llega su próximo pedido?
INFO agente: nodo modelo: 1.83s | 1 mensajes enviados | pide herramientas ['consultar_stock']
INFO agente: nodo modelo: 1.58s | 3 mensajes enviados | pide herramientas ['buscar_pedidos']
INFO agente: nodo modelo: 1.29s | 5 mensajes enviados | responde al usuario
...
INFO main: thread_id=sesion-1 | mensajes recuperados de memoria.db: 6 | pregunta: ¿Y quién es el proveedor de ese producto?
INFO agente: nodo modelo: 1.52s | 7 mensajes enviados | responde al usuario
```

En la última pregunta decidió **no** usar herramientas, porque ya tenía el dato en la memoria.

### 2. Ciclo ReAct y ciclo de retorno

**Dónde verlo** en [`traza.log`](traza.log), primer turno de `sesion-1`. Es la secuencia completa `AIMessage` con `tool_calls` → `ToolMessage` → síntesis final, repetida 2 veces (razonamiento multi-paso):

```text
Usuario (HumanMessage): "¿Qué producto tiene menos stock y cuándo llega su próximo pedido?"
→ El agente razona (AIMessage): Voy a consultar el stock de todos los productos para ver cuál tiene menos unidades.
→ El agente decide usar la herramienta: consultar_stock(producto='todos')  [tool_call_id=toolu_01DA...]
→ La herramienta devuelve (ToolMessage, tool_call_id=toolu_01DA...): {"stock": {..., "aceite": 8, ...}}
→ El agente razona (AIMessage): El producto con menos stock es el aceite, con 8 unidades. Ahora busco su próximo pedido al proveedor.
→ El agente decide usar la herramienta: buscar_pedidos(producto='aceite')  [tool_call_id=toolu_01QG...]
→ La herramienta devuelve (ToolMessage, tool_call_id=toolu_01QG...): {"producto": "aceite", "proveedor": "Aceites Cuyo", ...}
Respuesta (AIMessage): "El producto con menos stock es el aceite, con solo 8 unidades. ..."
```

**Ciclo de retorno**, en los turnos de `sesion-2` de [`traza.log`](traza.log): la herramienta devuelve un error por "fideos", el agente pide una aclaración y, cuando el usuario responde "quise decir arroz", vuelve a llamar a las herramientas.

### 3. Persistencia y resiliencia de estado (SqliteSaver + `thread_id`)

`main.py` corre **dos ejecuciones, y cada una abre una conexión nueva** a `memoria.db`, como si el programa se cerrara y se volviera a abrir.

**Dónde verlo** en [`traza.log`](traza.log), al final:

```text
=== Ejecución 2: nueva conexión a memoria.db (AsyncSqliteSaver) ===

[thread_id=sesion-1 | mensajes recuperados de memoria.db: 6]
Usuario (HumanMessage): "¿Y quién es el proveedor de ese producto?"
Respuesta (AIMessage): "El proveedor del aceite es **Aceites Cuyo**. ..."
```

Con el mismo `thread_id` (`sesion-1`), el checkpointer recuperó del disco los 6 mensajes de la ejecución anterior y el agente entendió "ese producto" (el aceite) sin volver a consultar. En [`ejecucion.log`](ejecucion.log), en cambio, la primera pregunta de `sesion-2` arranca con `mensajes recuperados: 0`: cada `thread_id` es una sesión separada.

`memoria.db` es un archivo binario y LangGraph guarda los mensajes en un formato interno, así que no se puede leer abriéndolo directo. Para ver su contenido en texto (no necesita API key):

```bash
python ver_memoria.py            # todas las conversaciones guardadas
python ver_memoria.py sesion-1   # solo un thread_id
```

Muestra, por cada `thread_id`, cuántos checkpoints hay y todos los mensajes guardados (`HumanMessage`, `AIMessage` con sus `tool_calls`, `ToolMessage`).

## Cumplimiento de la consigna

| Requisito | Dónde se cumple | Evidencia |
|---|---|---|
| Proyecto Python 3.12+ con venv | [`requirements.txt`](requirements.txt), sección [Instalación](#instalación) | — |
| `StateGraph` que hereda de `MessagesState` | [`agent.py` L27](agent.py#L27) (`AgentState`), [`agent.py` L66-78](agent.py#L66-L78) (grafo) | `tests/test_agente.py` |
| Nodo modelo + nodo herramientas + `tools_condition` | [`agent.py` L69-76](agent.py#L69-L76) | `test_razonamiento_multipaso` |
| Herramientas propias con `@tool` y docstring descriptivo | [`tools.py`](tools.py), ver sección [Herramientas](#herramientas-tool) | `tests/test_tools.py` |
| LLM vinculado con `llm.bind_tools()` | [`agent.py` L37-38](agent.py#L37-L38) | — |
| Persistencia en disco con `SqliteSaver` + `thread_id` | [`main.py` L108-113](main.py#L108-L113), [`chat.py` L19-20](chat.py#L19-L20) | "Ejecución 2" de [`traza.log`](traza.log), `test_la_memoria_sobrevive_al_reiniciar` |
| Razonamiento multi-paso (herramienta ≥ 2 veces) | Preguntas en [`main.py` L26](main.py#L26) | Primer turno de [`traza.log`](traza.log): `consultar_stock` → `buscar_pedidos` |
| `recursion_limit` definido | [`config.py` L17](config.py#L17) (`RECURSION_LIMIT = 10`), usado en [`main.py` L113](main.py#L113) | `test_recursion_limit_corta_bucles_infinitos` |
| **Autonomía** (sin `if/else` manuales) | El LLM decide; `tools_condition` solo mira si la respuesta trae `tool_calls` ([`agent.py` L74](agent.py#L74)) | [`traza.log`](traza.log) |
| **Ciclo de retorno** (reintento o aclaración) | Las herramientas devuelven `error` + `productos_disponibles`; el system prompt indica reintentar o pedir aclaración ([`agent.py` L18-24](agent.py#L18-L24)) | Turnos de `sesion-2` en [`traza.log`](traza.log), `test_reintenta_si_la_herramienta_da_error` |
| **Resiliencia de estado** (recuerda con el mismo `thread_id`) | `AsyncSqliteSaver` + `thread_id` en el `config` | "Ejecución 2" de [`traza.log`](traza.log), `test_recuerda_con_el_mismo_thread_id` |
| **Código limpio**: Python 3.12, type hints, `asyncio` | Todas las funciones tipadas; `async def` + `ainvoke` + `asyncio.run` | — |
| Estado sucio (limpiar el contexto) | `trim_messages` en [`agent.py` L44](agent.py#L44) (máximo 20 mensajes al LLM) | `test_recorta_el_historial_largo` |
| Sin API keys en el repo | `.env` en [`.gitignore`](.gitignore), validación en [`config.py`](config.py) | — |
| Traza ReAct (.json y log) | [`traza.json`](traza.json), [`traza.log`](traza.log) y [`ejecucion.log`](ejecucion.log) | Generadas con `python main.py` |

## Estructura

| Archivo | Qué hace |
|---|---|
| `config.py` | Carga el `.env`, valida que exista `ANTHROPIC_API_KEY` y define las constantes (modelo, `recursion_limit`, base SQLite). |
| `tools.py` | Las 2 herramientas (`@tool`) que simulan la base de datos del supermercado. |
| `agent.py` | El estado (`AgentState`, hereda de `MessagesState`), el nodo del modelo y el `StateGraph`. |
| `main.py` | Prueba de ejecución asíncrona: corre una sesión ReAct y genera `traza.json`, `traza.log`, `ejecucion.log` y `memoria.db`. |
| `chat.py` | Chat interactivo por consola, con memoria persistente en `memoria.db`. |
| `ver_memoria.py` | Muestra en texto lo que hay guardado en `memoria.db`, por `thread_id`. |
| `tests/` | Tests con pytest. Usan un LLM falso, así que **no consumen API**. |
| `traza.json` / `traza.log` | Traza ReAct de una ejecución real con Claude. |
| `ejecucion.log` | Logs de esa ejecución: decisiones del LLM, tiempos y mensajes recuperados del checkpointer. |
| `memoria.db` | Base SQLite generada por el `AsyncSqliteSaver` en esa ejecución (evidencia de la persistencia). |

## Herramientas (`@tool`)

Están en [`tools.py`](tools.py). Simulan consultas a la base de datos de Supermercados del Valle: los datos son inventados y viven en dos diccionarios (`STOCK` y `PEDIDOS`). El LLM decide cuál usar **solo a partir del docstring**, por eso cada uno explica cuándo usarla, qué recibe y qué devuelve, incluido el caso de error.

### `consultar_stock(producto: str) -> dict`

Consulta cuántas unidades hay en stock.

| Llamada | Devuelve |
|---|---|
| `consultar_stock("arroz")` | `{"producto": "arroz", "stock": 80}` |
| `consultar_stock("todos")` | `{"stock": {"leche": 120, "pan": 35, "arroz": 80, "aceite": 8, "yerba": 45}}` |
| `consultar_stock("fideos")` | `{"error": "El producto 'fideos' no existe.", "productos_disponibles": [...]}` |

Con `"todos"` el agente puede comparar productos, por ejemplo para saber cuál tiene menos stock.

### `buscar_pedidos(producto: str) -> dict`

Busca el próximo pedido a proveedor de un producto.

| Llamada | Devuelve |
|---|---|
| `buscar_pedidos("aceite")` | `{"producto": "aceite", "proveedor": "Aceites Cuyo", "fecha_entrega": "2026-10-12", "cantidad": 60}` |
| `buscar_pedidos("fideos")` | `{"error": "No hay pedidos para 'fideos'.", "productos_disponibles": [...]}` |

### Por qué devuelven `error` en vez de lanzar una excepción

Si el producto no existe, las herramientas no cortan el programa. Devuelven un `error` y la lista de `productos_disponibles`. LangGraph pasa esa respuesta al LLM como `ToolMessage` y el agente decide qué hacer: reintentar con el nombre correcto o pedirle una aclaración al usuario. Eso es el **ciclo de retorno**.

## Cómo funciona el grafo

```text
START → modelo ──(tools_condition)──→ tools ─┐
          ↑                                  │
          └──────────────────────────────────┘
          │
          └──(sin tool_calls)──→ END
```

- **Autonomía**: el LLM (`ChatAnthropic` con `llm.bind_tools(TOOLS)`) decide solo si llama a una herramienta y a cuál. No hay `if/else` en el código: la arista condicional `tools_condition` solo mira si la respuesta del modelo trae `tool_calls`. Si las trae, va al nodo `tools` (`ToolNode`), que ejecuta la herramienta y devuelve un `ToolMessage`. Después vuelve al modelo: ese es el ciclo.
- **Estado**: `MessagesState` usa el reducer `add_messages`, que **agrega** los mensajes nuevos a la lista en vez de reemplazarla. Cumple la función del `operator.add` que menciona la consigna y además maneja los IDs de los mensajes.
- **Límite de recursión**: todas las invocaciones usan `recursion_limit=10` para evitar bucles infinitos.
- **Estado sucio**: antes de llamar al LLM, `trim_messages` recorta el historial a los últimos 20 mensajes para que el contexto no crezca sin límite. El estado completo sigue guardado en SQLite.
- **Logs**: cada llamada al LLM registra su duración, cuántos mensajes se enviaron y qué decidió el modelo.
- **Sobre `temperature`**: no se pasa a propósito. `claude-sonnet-5-5` no acepta valores distintos del default y la API devuelve error. El modelo se crea en un solo lugar ([`agent.py` L37](agent.py#L37)).

## Persistencia con SqliteSaver

El checkpointer es **`AsyncSqliteSaver`**, del paquete `langgraph-checkpoint-sqlite`. Es la versión asíncrona de `SqliteSaver`, necesaria porque el grafo se ejecuta con `await agente.ainvoke(...)`, y el `SqliteSaver` común no funciona con `ainvoke`.

```python
async with AsyncSqliteSaver.from_conn_string("memoria.db") as checkpointer:
    agente = construir_grafo(checkpointer)
    config = {"configurable": {"thread_id": "sesion-1"}, "recursion_limit": 10}
    await agente.ainvoke({"messages": [HumanMessage("...")]}, config)
```

- El estado se guarda **en disco**, en el archivo `memoria.db`. No se usa `MemorySaver`, que guarda todo en la RAM y lo pierde al cerrar el programa.
- Después de cada paso del grafo, LangGraph guarda un *checkpoint* asociado al `thread_id`. Al invocar de nuevo con el mismo `thread_id`, se recupera toda la conversación.
- **Prueba en la traza**: `main.py` hace dos ejecuciones, y **cada una abre una conexión nueva** a `memoria.db`, como si se cerrara y se volviera a abrir el programa. En la "Ejecución 2", el agente recupera 6 mensajes del disco para `sesion-1` y responde "¿quién es el proveedor de ese producto?" sin volver a llamar a las herramientas.
- `memoria.db` se incluye en el repo como evidencia de la ejecución que generó la traza. Se vuelve a crear al correr `main.py`. Los archivos temporales de SQLite (`memoria.db-wal`, `memoria.db-shm`) están en el `.gitignore`.

## Instalación

Requiere **Python 3.12** y una API key de Anthropic.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version                 # debe decir Python 3.12.x
pip install -r requirements.txt
copy .env.example .env
notepad .env                     # completar ANTHROPIC_API_KEY
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version                 # debe decir Python 3.12.x
pip install -r requirements.txt
cp .env.example .env
nano .env                        # completar ANTHROPIC_API_KEY
```

Si falta la API key, el programa se detiene al inicio con un mensaje que explica qué hacer.

## Ejecución

Con el entorno virtual activado (los comandos son iguales en Windows, Linux y macOS):

```bash
# 1. Prueba de ejecución asíncrona (asyncio.run): corre la sesión ReAct
#    y genera traza.json, traza.log, ejecucion.log y memoria.db
python main.py

# 2. Chat interactivo con memoria persistente (el argumento es el thread_id)
python chat.py mi-sesion

# 3. Ver lo que quedó guardado en memoria.db
python ver_memoria.py

# 4. Tests (no necesitan API key ni consumen créditos)
python -m pytest
```

Para probar la persistencia a mano: en `python chat.py mi-sesion` preguntá algo, salí con `salir`, volvé a correr `python chat.py mi-sesion` y preguntá "¿de qué hablamos?". El agente lo recuerda porque la conversación quedó en `memoria.db`. Con otro `thread_id` la sesión arranca vacía.

## Traza ReAct

[`traza.log`](traza.log) es la versión legible y [`traza.json`](traza.json) la estructurada. Cada paso indica qué tipo de mensaje de LangChain lo generó:

| Paso ReAct | Mensaje | En el log |
|---|---|---|
| Input del usuario | `HumanMessage` | `Usuario (HumanMessage)` |
| Pensamiento del LLM | `AIMessage` (texto) | `El agente razona (AIMessage)` |
| Decisión de usar una herramienta | `AIMessage.tool_calls` | `El agente decide usar la herramienta` + `tool_call_id` |
| Resultado de la herramienta | `ToolMessage` (mismo `tool_call_id`) | `La herramienta devuelve (ToolMessage)` |
| Respuesta final | `AIMessage` sin `tool_calls` | `Respuesta (AIMessage)` |

Los extractos de cada caso están en [Cómo verificar cada criterio](#cómo-verificar-cada-criterio).
