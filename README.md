# Pre-entrega 5: Agente de razonamiento cíclico con memoria persistente

Agente de inventario de **Supermercados del Valle** hecho con **LangGraph**. Responde preguntas sobre stock y pedidos a proveedores usando herramientas propias, en un ciclo ReAct (razonar → usar herramienta → observar → repetir). Guarda la memoria de cada conversación en SQLite.

## Estructura

| Archivo | Qué hace |
|---|---|
| `tools.py` | Las 2 herramientas (`@tool`) que simulan la base de datos: `consultar_stock` y `buscar_pedidos`. |
| `agent.py` | El estado (`AgentState`, hereda de `MessagesState`), el nodo del modelo y el `StateGraph`. |
| `main.py` | Prueba automática: corre las preguntas de ejemplo y genera `traza.json` y `traza.log`. |
| `chat.py` | Chat interactivo por consola, con memoria persistente. |
| `traza.json` / `traza.log` | Traza ReAct de una ejecución real. |

## Cómo funciona

```
START → modelo ──(tools_condition)──→ tools ─┐
          ↑                                  │
          └──────────────────────────────────┘
          │
          └──(sin tool_calls)──→ END
```

- **Autonomía**: el LLM (`ChatAnthropic` con `llm.bind_tools(TOOLS)`) decide solo si llama a una herramienta. No hay `if/else` en el código: la arista condicional `tools_condition` mira si la respuesta del modelo trae `tool_calls`.
- **Ciclo de retorno**: si la herramienta no encuentra el producto, devuelve un `"error"` con la lista de `productos_disponibles`. El agente reintenta con el nombre correcto o pide una aclaración al usuario.
- **Estado**: `MessagesState` usa el reducer `add_messages`, que **agrega** los mensajes nuevos a la lista en vez de reemplazarla (cumple la función del `operator.add` de la consigna y además maneja los IDs de los mensajes).
- **Persistencia**: `AsyncSqliteSaver` (la versión asíncrona de `SqliteSaver`, del paquete `langgraph-checkpoint-sqlite`) guarda el estado en `memoria.db`. Con el mismo `thread_id` el agente recuerda la conversación, incluso si cerrás y volvés a abrir el programa. Se usa la versión asíncrona porque el `SqliteSaver` común no funciona con `ainvoke`.
- **Límite de recursión**: todas las invocaciones usan `recursion_limit=10` para evitar bucles infinitos.
- **Estado sucio**: antes de llamar al LLM, `trim_messages` recorta el historial a los últimos 20 mensajes para que el contexto no crezca sin límite.

## Cómo levantar el entorno

Requiere **Python 3.12+** y una API key de Anthropic.

```bash
# 1. Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar la API key
copy .env.example .env          # Windows  (cp en Linux / Mac)
# y completar ANTHROPIC_API_KEY en .env
```

## Cómo ejecutarlo

```bash
# Prueba automática + genera traza.json y traza.log
python main.py

# Chat interactivo (el argumento es el thread_id; por defecto "chat-1")
python chat.py mi-sesion
```

## Ejemplo de traza (`traza.log`)

```
[thread_id=sesion-1]
Usuario: "¿Qué producto tiene menos stock y cuándo llega su próximo pedido?"
→ El agente decide usar la herramienta: consultar_stock(producto='todos')
→ La herramienta devuelve: {"stock": {"leche": 120, "pan": 35, "arroz": 80, "aceite": 8, "yerba": 45}}
→ El agente decide usar la herramienta: buscar_pedidos(producto='aceite')
→ La herramienta devuelve: {"producto": "aceite", "proveedor": "Aceites Cuyo", "fecha_entrega": "2026-10-12", "cantidad": 60}
Respuesta: "El producto con menos stock es el **aceite**, con solo 8 unidades. ..."
(llamadas a herramientas: 2)

[thread_id=sesion-1]
Usuario: "¿Y quién es el proveedor de ese producto?"
Respuesta: "El proveedor del aceite es **Aceites Cuyo**."
(llamadas a herramientas: 0)
```

1. **Razonamiento multi-paso**: para la primera pregunta el agente llama a 2 herramientas. Primero averigua qué producto tiene menos stock y con ese dato busca el pedido.
2. **Memoria**: en la segunda pregunta ("ese producto") no llama a ninguna herramienta, porque recuerda la respuesta anterior gracias al `thread_id`.
3. **Ciclo de retorno** (`sesion-2` en `traza.log`): el usuario pregunta por "fideos", las herramientas devuelven error, y el agente pide una aclaración. Cuando el usuario responde "quise decir arroz", vuelve a llamar a las herramientas y contesta.

La traza completa, con el detalle de cada paso, está en [`traza.json`](traza.json).
