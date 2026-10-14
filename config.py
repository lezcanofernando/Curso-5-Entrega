"""Configuración del proyecto. Valida las variables de entorno apenas se importa."""

import os

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    raise SystemExit(
        "Falta la variable de entorno ANTHROPIC_API_KEY.\n"
        "Copiá .env.example a .env y completá tu API key de Anthropic."
    )

MODEL: str = os.getenv("MODEL") or "claude-sonnet-5-5"
DB_PATH: str = "memoria.db"
RECURSION_LIMIT: int = 10  # techo de pasos del grafo para no gastar API de más
MAX_MENSAJES: int = 20  # cuántos mensajes del historial se mandan al LLM como máximo
