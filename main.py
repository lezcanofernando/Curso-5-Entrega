"""Prueba del agente: corre una sesión ReAct y guarda la traza en traza.json y traza.log.

La prueba tiene dos "ejecuciones". Cada una abre su propia conexión a memoria.db (SQLite),
como si el programa se cerrara y se volviera a abrir. En la ejecución 2 el agente recuerda
lo hablado en la ejecución 1 porque el estado quedó guardado en disco con el thread_id.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agent import construir_grafo
from config import DB_PATH, MODEL, RECURSION_LIMIT

logger = logging.getLogger("main")

# Cada ejecución es una lista de (thread_id, pregunta)
EJECUCIONES: list[list[tuple[str, str]]] = [
    [
        # Multi-paso: necesita consultar_stock("todos") y después buscar_pedidos(...)
        ("sesion-1", "¿Qué producto tiene menos stock y cuándo llega su próximo pedido?"),
        # Ciclo de retorno: "fideos" no existe, la herramienta da error y el agente pide aclaración
        ("sesion-2", "¿Cuánto stock hay de fideos y cuándo llega el próximo pedido?"),
        # El usuario aclara y el agente vuelve a intentar (recordando qué se preguntaba)
        ("sesion-2", "Perdón, quise decir arroz."),
    ],
    [
        # Persistencia: nueva conexión a memoria.db; "ese producto" solo se entiende
        # si el agente recupera del disco la conversación de sesion-1
        ("sesion-1", "¿Y quién es el proveedor de ese producto?"),
    ],
]


def pasos_de(mensajes: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convierte los mensajes de un turno en los pasos del ciclo ReAct."""
    pasos: list[dict[str, Any]] = []
    for msg in mensajes:
        if isinstance(msg, HumanMessage):
            pasos.append({"paso": "input_usuario", "mensaje": "HumanMessage", "contenido": msg.text})
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            if msg.text:
                pasos.append({"paso": "pensamiento", "mensaje": "AIMessage", "contenido": msg.text})
            for llamada in msg.tool_calls:
                pasos.append({
                    "paso": "accion",
                    "mensaje": "AIMessage.tool_calls",
                    "herramienta": llamada["name"],
                    "argumentos": llamada["args"],
                    "tool_call_id": llamada["id"],
                })
        elif isinstance(msg, AIMessage):
            pasos.append({"paso": "respuesta_final", "mensaje": "AIMessage", "contenido": msg.text})
        elif isinstance(msg, ToolMessage):
            pasos.append({
                "paso": "observacion",
                "mensaje": "ToolMessage",
                "herramienta": msg.name,
                "tool_call_id": msg.tool_call_id,
                "contenido": json.loads(msg.text),
            })
    return pasos


def a_log(turno: dict[str, Any]) -> str:
    """Formatea un turno como el ejemplo de la consigna."""
    lineas = [
        f"[thread_id={turno['thread_id']} | mensajes recuperados de {DB_PATH}: "
        f"{turno['mensajes_previos_en_memoria']}]"
    ]
    for paso in turno["pasos"]:
        match paso["paso"]:
            case "input_usuario":
                lineas.append(f'Usuario (HumanMessage): "{paso["contenido"]}"')
            case "pensamiento":
                lineas.append(f"→ El agente razona (AIMessage): {paso['contenido']}")
            case "accion":
                args = ", ".join(f"{k}={v!r}" for k, v in paso["argumentos"].items())
                lineas.append(
                    f"→ El agente decide usar la herramienta: {paso['herramienta']}({args})"
                    f"  [tool_call_id={paso['tool_call_id']}]"
                )
            case "observacion":
                contenido = json.dumps(paso["contenido"], ensure_ascii=False)
                lineas.append(
                    f"→ La herramienta devuelve (ToolMessage, tool_call_id={paso['tool_call_id']}): {contenido}"
                )
            case "respuesta_final":
                lineas.append(f'Respuesta (AIMessage): "{paso["contenido"]}"')
    lineas.append(
        f"(llamadas a herramientas: {turno['llamadas_a_herramientas']}"
        f" | duración: {turno['duracion_segundos']}s)"
    )
    return "\n".join(lineas)


async def ejecutar(numero: int, preguntas: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Abre una conexión nueva a la base SQLite y hace las preguntas."""
    turnos: list[dict[str, Any]] = []
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        agente = construir_grafo(checkpointer)
        logger.info("=== Ejecución %d: nueva conexión a %s (AsyncSqliteSaver) ===", numero, DB_PATH)

        for thread_id, pregunta in preguntas:
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

            # Lo que el checkpointer ya tenía guardado en disco para este thread_id
            estado = await agente.aget_state(config)
            antes = len(estado.values.get("messages", []))
            logger.info(
                "thread_id=%s | mensajes recuperados de %s: %d | pregunta: %s", thread_id, DB_PATH, antes, pregunta
            )

            inicio = time.perf_counter()
            try:
                resultado = await agente.ainvoke({"messages": [HumanMessage(pregunta)]}, config)
            except GraphRecursionError:
                print(f"Se alcanzó el recursion_limit ({RECURSION_LIMIT}) en: {pregunta}")
                continue
            duracion = time.perf_counter() - inicio

            pasos = pasos_de(resultado["messages"][antes:])
            turnos.append({
                "ejecucion": numero,
                "thread_id": thread_id,
                "mensajes_previos_en_memoria": antes,
                "llamadas_a_herramientas": sum(p["paso"] == "accion" for p in pasos),
                "duracion_segundos": round(duracion, 2),
                "pasos": pasos,
            })
    return turnos


async def main() -> None:
    # Solo al inicio de la prueba se borra la base, para que la traza sea reproducible.
    # Entre la ejecución 1 y la 2 NO se borra: así se ve que la memoria persiste en disco.
    Path(DB_PATH).unlink(missing_ok=True)

    turnos: list[dict[str, Any]] = []
    bloques_log: list[str] = []
    for numero, preguntas in enumerate(EJECUCIONES, start=1):
        nuevos = await ejecutar(numero, preguntas)
        turnos.extend(nuevos)
        titulo = f"=== Ejecución {numero}: nueva conexión a {DB_PATH} (AsyncSqliteSaver) ==="
        bloques_log.append("\n\n".join([titulo, *(a_log(t) for t in nuevos)]))
        print(bloques_log[-1], end="\n\n")

    traza = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": MODEL,
        "checkpointer": f"AsyncSqliteSaver ({DB_PATH})",
        "recursion_limit": RECURSION_LIMIT,
        "turnos": turnos,
    }
    Path("traza.json").write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("traza.log").write_text("\n\n".join(bloques_log) + "\n", encoding="utf-8")
    print("Traza guardada en traza.json y traza.log")


if __name__ == "__main__":
    # Los logs se muestran en consola y quedan guardados en ejecucion.log como evidencia
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("ejecucion.log", mode="w", encoding="utf-8")],
    )
    # Solo los logs del agente (sin el detalle interno de las librerías HTTP)
    for nombre in ("httpx", "httpx2"):
        logging.getLogger(nombre).setLevel(logging.WARNING)
    asyncio.run(main())
