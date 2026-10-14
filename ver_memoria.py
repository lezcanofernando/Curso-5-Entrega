"""Muestra lo que hay guardado en memoria.db, conversación por conversación.

memoria.db es una base SQLite (archivo binario) y los mensajes se guardan en un formato
interno de LangGraph, así que no se pueden leer abriéndola directo. Este script la lee
con el mismo AsyncSqliteSaver que usa el agente y muestra los mensajes en texto.

Uso: python ver_memoria.py [thread_id]
No necesita API key: solo lee el archivo.
"""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DB_PATH = "memoria.db"


def describir(msg: BaseMessage) -> str:
    """Una línea legible por mensaje."""
    tipo = type(msg).__name__
    if isinstance(msg, AIMessage) and msg.tool_calls:
        llamadas = ", ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in msg.tool_calls)
        pensamiento = f" {msg.text}" if msg.text else ""
        return f"{tipo}:{pensamiento} → pide herramientas: {llamadas}"
    if isinstance(msg, ToolMessage):
        return f"{tipo} ({msg.name}): {msg.text}"
    return f"{tipo}: {msg.text}"


async def main(solo_thread: str | None) -> None:
    if not Path(DB_PATH).exists():
        print(f"No existe {DB_PATH}. Corré primero: python main.py")
        return

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        # Cuántos checkpoints (uno por paso del grafo) hay guardados por thread_id
        checkpoints = Counter([c.config["configurable"]["thread_id"] async for c in checkpointer.alist(None)])
        threads = [solo_thread] if solo_thread else sorted(checkpoints)

        for thread_id in threads:
            tupla = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
            if tupla is None:
                print(f"No hay nada guardado para thread_id={thread_id}")
                continue
            mensajes: list[BaseMessage] = tupla.checkpoint["channel_values"].get("messages", [])
            print(f"=== thread_id={thread_id} | {checkpoints[thread_id]} checkpoints | {len(mensajes)} mensajes ===")
            for numero, msg in enumerate(mensajes, start=1):
                print(f"{numero:>2}. {describir(msg)}")
            print()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
