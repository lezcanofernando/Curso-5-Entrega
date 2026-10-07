"""Prueba del agente: corre unas preguntas y guarda la traza ReAct en traza.json y traza.log."""

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agent import construir_grafo

DB_PATH = "memoria.db"
RECURSION_LIMIT = 10  # techo de pasos del grafo para no gastar API de más

# (thread_id, pregunta)
PREGUNTAS: list[tuple[str, str]] = [
    # Multi-paso: necesita consultar_stock("todos") y después buscar_pedidos(...)
    ("sesion-1", "¿Qué producto tiene menos stock y cuándo llega su próximo pedido?"),
    # Memoria: "ese producto" solo se entiende con el contexto del mismo thread_id
    ("sesion-1", "¿Y quién es el proveedor de ese producto?"),
    # Ciclo de retorno: "fideos" no existe, la herramienta da error y el agente pide aclaración
    ("sesion-2", "¿Cuánto stock hay de fideos y cuándo llega el próximo pedido?"),
    # El usuario aclara y el agente vuelve a intentar (recordando qué se preguntaba)
    ("sesion-2", "Perdón, quise decir arroz."),
]


def pasos_de(mensajes: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convierte los mensajes de un turno en pasos legibles de la traza."""
    pasos: list[dict[str, Any]] = []
    for msg in mensajes:
        if isinstance(msg, HumanMessage):
            pasos.append({"tipo": "usuario", "contenido": msg.text})
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                if msg.text:
                    pasos.append({"tipo": "razonamiento", "contenido": msg.text})
                for llamada in msg.tool_calls:
                    pasos.append({
                        "tipo": "accion",
                        "herramienta": llamada["name"],
                        "argumentos": llamada["args"],
                    })
            else:
                pasos.append({"tipo": "respuesta", "contenido": msg.text})
        elif isinstance(msg, ToolMessage):
            pasos.append({
                "tipo": "observacion",
                "herramienta": msg.name,
                "contenido": json.loads(msg.text),
            })
    return pasos


def a_log(turno: dict[str, Any]) -> str:
    """Formatea un turno como el ejemplo de la consigna."""
    lineas = [f"[thread_id={turno['thread_id']}]"]
    for paso in turno["pasos"]:
        match paso["tipo"]:
            case "usuario":
                lineas.append(f'Usuario: "{paso["contenido"]}"')
            case "razonamiento":
                lineas.append(f"→ El agente razona: {paso['contenido']}")
            case "accion":
                args = ", ".join(f"{k}={v!r}" for k, v in paso["argumentos"].items())
                lineas.append(f"→ El agente decide usar la herramienta: {paso['herramienta']}({args})")
            case "observacion":
                contenido = json.dumps(paso["contenido"], ensure_ascii=False)
                lineas.append(f"→ La herramienta devuelve: {contenido}")
            case "respuesta":
                lineas.append(f'Respuesta: "{paso["contenido"]}"')
    lineas.append(f"(llamadas a herramientas: {turno['llamadas_a_herramientas']})")
    return "\n".join(lineas)


async def main() -> None:
    # Arrancamos con una base limpia para que la traza sea reproducible
    Path(DB_PATH).unlink(missing_ok=True)

    traza: list[dict[str, Any]] = []
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        agente = construir_grafo(checkpointer)

        for thread_id, pregunta in PREGUNTAS:
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

            # Cuántos mensajes había antes, para quedarnos solo con los de este turno
            estado = await agente.aget_state(config)
            antes = len(estado.values.get("messages", []))

            try:
                resultado = await agente.ainvoke({"messages": [HumanMessage(pregunta)]}, config)
            except GraphRecursionError:
                print(f"Se alcanzó el recursion_limit ({RECURSION_LIMIT}) en: {pregunta}")
                continue

            pasos = pasos_de(resultado["messages"][antes:])
            turno = {
                "thread_id": thread_id,
                "pregunta": pregunta,
                "llamadas_a_herramientas": sum(p["tipo"] == "accion" for p in pasos),
                "pasos": pasos,
            }
            traza.append(turno)
            print(a_log(turno), end="\n\n")

    Path("traza.json").write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("traza.log").write_text("\n\n".join(a_log(t) for t in traza) + "\n", encoding="utf-8")
    print("Traza guardada en traza.json y traza.log")


if __name__ == "__main__":
    asyncio.run(main())
