"""Definición del agente: estado, nodos y grafo de LangGraph."""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS

load_dotenv()

# Cuántos mensajes del historial le mandamos al LLM como máximo (evita el "estado sucio")
MAX_MENSAJES = 20

SYSTEM_PROMPT = """Sos el asistente de inventario de Supermercados del Valle.
Respondé en español, de forma breve y clara.
- Usá las herramientas para obtener datos; nunca inventes stock, fechas ni proveedores.
- Si una pregunta necesita varios datos, llamá a las herramientas las veces que haga falta.
- Si una herramienta devuelve "error", revisá "productos_disponibles" y reintentá con el
  nombre correcto. Si no hay un producto parecido, pedile una aclaración al usuario."""


class AgentState(MessagesState):
    """Estado del agente.

    Hereda de MessagesState la clave `messages`, que usa el reducer `add_messages`:
    cada nodo devuelve mensajes nuevos y LangGraph los agrega a la lista (no la pisa).
    """


llm = ChatAnthropic(model=os.getenv("MODEL", "claude-sonnet-5-5"))
llm_con_tools = llm.bind_tools(TOOLS)


async def nodo_modelo(state: AgentState) -> dict[str, list[BaseMessage]]:
    """Llama al LLM. Él decide si responde o si pide usar una herramienta."""
    # Solo mandamos los últimos mensajes para que el contexto no crezca sin límite
    historial = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,  # cuenta mensajes en vez de tokens
        max_tokens=MAX_MENSAJES,
        start_on="human",
    )
    respuesta = await llm_con_tools.ainvoke([SystemMessage(SYSTEM_PROMPT), *historial])
    return {"messages": [respuesta]}


def construir_grafo(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Arma el ciclo ReAct: modelo -> (tools -> modelo)* -> fin."""
    grafo = StateGraph(AgentState)
    grafo.add_node("modelo", nodo_modelo)
    grafo.add_node("tools", ToolNode(TOOLS))

    grafo.add_edge(START, "modelo")
    # Si el LLM pidió una herramienta va a "tools"; si no, termina (END)
    grafo.add_conditional_edges("modelo", tools_condition)
    # Después de ejecutar la herramienta vuelve al modelo: este es el ciclo
    grafo.add_edge("tools", "modelo")

    return grafo.compile(checkpointer=checkpointer)
