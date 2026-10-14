"""Configuración de pytest: los tests no llaman a la API real."""

import os
import sys
from pathlib import Path

# Key de mentira para que config.py no corte la ejecución (el LLM se reemplaza en los tests)
os.environ.setdefault("ANTHROPIC_API_KEY", "key-de-prueba")

# Permite importar los módulos de la raíz del proyecto (agent, tools, config)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
