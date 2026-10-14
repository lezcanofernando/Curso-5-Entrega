"""Tests del grafo con un LLM falso (no consumen API).

El LLM falso devuelve respuestas armadas de antemano, así podemos verificar el
ciclo ReAct, el reintento ante errores, la memoria por thread_id y el recursion_limit.
"""

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

import agent


class LLMFalso:
    """Reemplaza a llm_con_tools: devuelve las respuestas del guion en orden."""

    def __init__(self, guion: list[AIMessage]) -> None:
        self.guion = guion
        self.recibidos: list[list[BaseMessage]] = []  # lo que "vio" el LLM en cada llamada

    async def ainvoke(self, mensajes: list[BaseMessage]) -> AIMessage:
        self.recibidos.append(mensajes)
        return self.guion.pop(0)


def pedir(herramienta: str, producto: str, id_: str) -> AIMessage:
    """Respuesta del LLM que pide usar una herramienta."""
    return AIMessage(content="", tool_calls=[{"name": herramienta, "args": {"producto": producto}, "id": id_}])


async def preguntar(db: Path, preguntas: list[tuple[str, str]], recursion_limit: int = 10) -> list[dict]:
    """Hace las preguntas (thread_id, texto) con un checkpointer SQLite y devuelve los resultados."""
    resultados = []
    async with AsyncSqliteSaver.from_conn_string(str(db)) as checkpointer:
        grafo = agent.construir_grafo(checkpointer)
        for thread_id, texto in preguntas:
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
            resultados.append(await grafo.ainvoke({"messages": [HumanMessage(texto)]}, config))
    return resultados


def observaciones(resultado: dict) -> list[dict]:
    """Devuelve lo que contestaron las herramientas, como diccionarios."""
    return [json.loads(m.text) for m in resultado["messages"] if isinstance(m, ToolMessage)]


def test_razonamiento_multipaso(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = LLMFalso([
        pedir("consultar_stock", "todos", "1"),
        pedir("buscar_pedidos", "aceite", "2"),
        AIMessage("El aceite tiene menos stock y llega el 12/10."),
    ])
    monkeypatch.setattr(agent, "llm_con_tools", llm)

    [resultado] = asyncio.run(preguntar(tmp_path / "m.db", [("t1", "¿Qué tiene menos stock y cuándo llega?")]))

    obs = observaciones(resultado)
    assert len(obs) == 2  # la herramienta se usó dos veces
    assert obs[0]["stock"]["aceite"] == 8
    assert obs[1]["proveedor"] == "Aceites Cuyo"
    assert resultado["messages"][-1].text == "El aceite tiene menos stock y llega el 12/10."


def test_reintenta_si_la_herramienta_da_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = LLMFalso([
        pedir("buscar_pedidos", "aceite de girasol", "1"),
        pedir("buscar_pedidos", "aceite", "2"),
        AIMessage("Llega el 12/10."),
    ])
    monkeypatch.setattr(agent, "llm_con_tools", llm)

    [resultado] = asyncio.run(preguntar(tmp_path / "m.db", [("t1", "¿Cuándo llega el aceite de girasol?")]))

    primero, segundo = observaciones(resultado)
    assert "error" in primero
    assert segundo["producto"] == "aceite"
    # En el segundo intento el LLM vio el error de la herramienta
    assert any(isinstance(m, ToolMessage) and "error" in m.text for m in llm.recibidos[1])


def test_recuerda_con_el_mismo_thread_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = LLMFalso([AIMessage("Hola Ana."), AIMessage("Te llamás Ana."), AIMessage("No lo sé.")])
    monkeypatch.setattr(agent, "llm_con_tools", llm)

    asyncio.run(preguntar(tmp_path / "m.db", [
        ("sesion-a", "Me llamo Ana"),
        ("sesion-a", "¿Cómo me llamo?"),
        ("sesion-b", "¿Cómo me llamo?"),
    ]))

    textos_sesion_a = [m.text for m in llm.recibidos[1]]
    textos_sesion_b = [m.text for m in llm.recibidos[2]]
    assert "Me llamo Ana" in textos_sesion_a  # mismo thread_id: recuerda
    assert "Me llamo Ana" not in textos_sesion_b  # otro thread_id: sesión separada


def test_la_memoria_sobrevive_al_reiniciar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "m.db"
    llm = LLMFalso([AIMessage("Anotado."), AIMessage("Hablamos del pan.")])
    monkeypatch.setattr(agent, "llm_con_tools", llm)

    # Dos conexiones distintas a la misma base, como si se cerrara y abriera el programa
    asyncio.run(preguntar(db, [("s1", "Quiero saber del pan")]))
    asyncio.run(preguntar(db, [("s1", "¿De qué hablamos?")]))

    assert "Quiero saber del pan" in [m.text for m in llm.recibidos[1]]


def test_recursion_limit_corta_bucles_infinitos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Un LLM que siempre pide una herramienta y nunca responde
    llm = LLMFalso([pedir("consultar_stock", "pan", str(i)) for i in range(50)])
    monkeypatch.setattr(agent, "llm_con_tools", llm)

    with pytest.raises(GraphRecursionError):
        asyncio.run(preguntar(tmp_path / "m.db", [("t1", "Bucle")], recursion_limit=10))
    assert len(llm.recibidos) <= 10


def test_recorta_el_historial_largo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preguntas = [("t1", f"Pregunta {i}") for i in range(15)]
    llm = LLMFalso([AIMessage(f"Respuesta {i}") for i in range(15)])
    monkeypatch.setattr(agent, "llm_con_tools", llm)

    asyncio.run(preguntar(tmp_path / "m.db", preguntas))

    # El estado guarda 30 mensajes, pero al LLM le llegan como máximo MAX_MENSAJES (+ el system)
    assert len(llm.recibidos[-1]) <= agent.MAX_MENSAJES + 1
    assert llm.recibidos[-1][-1].text == "Pregunta 14"
