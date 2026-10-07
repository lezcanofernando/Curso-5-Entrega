"""Chat interactivo con el agente. La memoria queda guardada en memoria.db.

Uso: python chat.py [thread_id]
Si cerrás el programa y lo volvés a abrir con el mismo thread_id, el agente recuerda la charla.
"""

import asyncio
import sys

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agent import construir_grafo


async def main(thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    async with AsyncSqliteSaver.from_conn_string("memoria.db") as checkpointer:
        agente = construir_grafo(checkpointer)
        print(f"Sesión '{thread_id}'. Escribí 'salir' para terminar.")
        while (pregunta := input("\nVos: ").strip()).lower() != "salir":
            if not pregunta:
                continue
            try:
                resultado = await agente.ainvoke({"messages": [HumanMessage(pregunta)]}, config)
                print(f"Agente: {resultado['messages'][-1].text}")
            except GraphRecursionError:
                print("Agente: no pude resolverlo dentro del límite de pasos.")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "chat-1"))
