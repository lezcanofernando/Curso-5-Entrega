"""Tests de las herramientas simuladas."""

from tools import buscar_pedidos, consultar_stock


def test_stock_de_un_producto() -> None:
    assert consultar_stock.invoke({"producto": "arroz"}) == {"producto": "arroz", "stock": 80}


def test_stock_ignora_mayusculas_y_espacios() -> None:
    assert consultar_stock.invoke({"producto": "  Leche "})["stock"] == 120


def test_stock_de_todos_los_productos() -> None:
    stock = consultar_stock.invoke({"producto": "todos"})["stock"]
    assert min(stock, key=stock.get) == "aceite"


def test_stock_producto_inexistente_devuelve_error() -> None:
    resultado = consultar_stock.invoke({"producto": "fideos"})
    assert "error" in resultado
    assert "arroz" in resultado["productos_disponibles"]


def test_pedido_de_un_producto() -> None:
    pedido = buscar_pedidos.invoke({"producto": "aceite"})
    assert pedido["proveedor"] == "Aceites Cuyo"
    assert pedido["fecha_entrega"] == "2026-10-12"


def test_pedido_producto_inexistente_devuelve_error() -> None:
    resultado = buscar_pedidos.invoke({"producto": "fideos"})
    assert "error" in resultado
    assert "aceite" in resultado["productos_disponibles"]
