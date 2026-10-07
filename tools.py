"""Herramientas del agente.

Simulan consultas a la base de datos de Supermercados del Valle.
Los datos son inventados y viven en diccionarios (no hay base de datos real).
"""

from typing import Any

from langchain_core.tools import tool

# "Tabla" de stock: producto -> unidades disponibles
STOCK: dict[str, int] = {
    "leche": 120,
    "pan": 35,
    "arroz": 80,
    "aceite": 8,
    "yerba": 45,
}

# "Tabla" de pedidos a proveedores: producto -> próximo pedido
PEDIDOS: dict[str, dict[str, Any]] = {
    "leche": {"proveedor": "Lácteos del Sur", "fecha_entrega": "2026-10-10", "cantidad": 200},
    "pan": {"proveedor": "Panificadora Central", "fecha_entrega": "2026-10-08", "cantidad": 100},
    "arroz": {"proveedor": "Distribuidora Norte", "fecha_entrega": "2026-10-15", "cantidad": 150},
    "aceite": {"proveedor": "Aceites Cuyo", "fecha_entrega": "2026-10-12", "cantidad": 60},
    "yerba": {"proveedor": "Yerbatera Misiones", "fecha_entrega": "2026-10-20", "cantidad": 90},
}


@tool
def consultar_stock(producto: str) -> dict[str, Any]:
    """Consulta cuántas unidades hay en stock de un producto del supermercado.

    Usá esta herramienta cuando el usuario pregunte por stock, existencias,
    unidades disponibles o qué producto tiene más o menos stock.

    Args:
        producto: nombre simple del producto en minúsculas (ej: "leche", "arroz").
            Pasá "todos" para obtener el stock de TODOS los productos, por
            ejemplo para saber cuál tiene menos o más stock.

    Returns:
        Un diccionario con el stock. Si el producto no existe devuelve una
        clave "error" y la lista de "productos_disponibles": en ese caso volvé
        a llamar a la herramienta con el nombre correcto.
    """
    nombre = producto.strip().lower()
    if nombre == "todos":
        return {"stock": STOCK}
    if nombre not in STOCK:
        return {
            "error": f"El producto '{producto}' no existe.",
            "productos_disponibles": list(STOCK),
        }
    return {"producto": nombre, "stock": STOCK[nombre]}


@tool
def buscar_pedidos(producto: str) -> dict[str, Any]:
    """Busca el próximo pedido a proveedor de un producto del supermercado.

    Usá esta herramienta cuando el usuario pregunte cuándo llega mercadería,
    quién es el proveedor de un producto o cuántas unidades se pidieron.

    Args:
        producto: nombre simple del producto en minúsculas (ej: "leche", "aceite").

    Returns:
        Un diccionario con proveedor, fecha_entrega (AAAA-MM-DD) y cantidad.
        Si el producto no existe devuelve una clave "error" y la lista de
        "productos_disponibles": en ese caso volvé a llamar a la herramienta
        con el nombre correcto o pedile una aclaración al usuario.
    """
    nombre = producto.strip().lower()
    if nombre not in PEDIDOS:
        return {
            "error": f"No hay pedidos para '{producto}'.",
            "productos_disponibles": list(PEDIDOS),
        }
    return {"producto": nombre, **PEDIDOS[nombre]}


TOOLS = [consultar_stock, buscar_pedidos]
