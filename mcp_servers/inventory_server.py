"""ReplenishIQ inventory MCP server (FastMCP, Streamable HTTP transport).

Wraps the store inventory / sales / supplier database behind a small set of
MCP tools. The server is stateless (every request opens its own read-only
SQLite connection), so in production you could run several replicas behind a
load balancer.

Run:  python -m mcp_servers.inventory_server          # http://localhost:8000/mcp
Auth: set MCP_AUTH_TOKEN to require `Authorization: Bearer <token>`.
"""
from __future__ import annotations

import re
import sqlite3
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastmcp import FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402

import config  # noqa: E402

SKU_RE = re.compile(r"^[A-Z]{2,4}-\d{3}$")
STORE_RE = re.compile(r"^S\d{3}$")
READ_ONLY = {"readOnlyHint": True, "openWorldHint": False}


def _auth():
    """Optional static bearer-token auth (demo-grade; use OAuth 2.1 / an IdP in production)."""
    if not config.MCP_AUTH_TOKEN:
        return None
    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

    return StaticTokenVerifier(
        tokens={
            config.MCP_AUTH_TOKEN: {
                "client_id": "replenishiq-agents",
                "scopes": ["inventory:read", "orders:write"],
            }
        }
    )


mcp = FastMCP(
    name="replenishiq-inventory",
    instructions=(
        "Store inventory, sales history and supplier terms for the ReplenishIQ pilot stores. "
        "Read tools are side-effect free. create_purchase_order writes data and must only be "
        "called after a human has approved the order."
    ),
    auth=_auth(),
)


# ---------------------------------------------------------------- helpers
def _connect(read_only: bool = True) -> sqlite3.Connection:
    if not config.DB_PATH.exists():
        raise ToolError(f"Database not found at {config.DB_PATH}. Run scripts/generate_data.py first.")
    uri = f"file:{config.DB_PATH}?mode={'ro' if read_only else 'rw'}"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with closing(_connect()) as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _check_store(store: str) -> str:
    store = store.strip().upper()
    if not STORE_RE.match(store):
        raise ToolError(f"Invalid store id '{store}'. Expected format like 'S001'.")
    if not _rows("SELECT 1 FROM inventory WHERE store = ? LIMIT 1", (store,)):
        raise ToolError(f"Unknown store '{store}'.")
    return store


def _check_sku(sku: str) -> str:
    sku = sku.strip().upper()
    if not SKU_RE.match(sku):
        raise ToolError(f"Invalid SKU '{sku}'. Expected format like 'GRO-001'.")
    return sku


_STOCK_SQL = """
SELECT i.sku, p.name, p.category, i.store, i.on_hand, i.on_order, i.shelf_capacity, i.unit_cost
FROM inventory i JOIN products p USING (sku)
"""


# ---------------------------------------------------------------- tools
@mcp.tool(annotations=READ_ONLY)
def get_stock(sku: str, store: str) -> dict[str, Any]:
    """Current stock position for one SKU at one store: on_hand, on_order, shelf_capacity, unit_cost."""
    sku, store = _check_sku(sku), _check_store(store)
    rows = _rows(_STOCK_SQL + " WHERE i.sku = ? AND i.store = ?", (sku, store))
    if not rows:
        raise ToolError(f"SKU {sku} is not stocked at store {store}.")
    return rows[0]


@mcp.tool(annotations=READ_ONLY)
def list_store_inventory(store: str) -> dict[str, Any]:
    """Stock position for every SKU stocked at a store."""
    store = _check_store(store)
    return {"store": store, "items": _rows(_STOCK_SQL + " WHERE i.store = ? ORDER BY i.sku", (store,))}


@mcp.tool(annotations=READ_ONLY)
def list_low_stock(store: str, days_of_cover: float = 7.0) -> dict[str, Any]:
    """SKUs at a store whose (on_hand + on_order) covers fewer than `days_of_cover` days of
    average demand (average of the last 28 days of sales)."""
    store = _check_store(store)
    if not 0 < days_of_cover <= 60:
        raise ToolError("days_of_cover must be between 0 and 60.")
    sql = """
    WITH recent AS (
        SELECT sku, AVG(units_sold) AS avg_daily FROM sales_history
        WHERE store = ? AND date > date((SELECT MAX(date) FROM sales_history), '-28 day')
        GROUP BY sku)
    SELECT i.sku, p.name, p.category, i.on_hand, i.on_order,
           ROUND(r.avg_daily, 1) AS avg_daily_sales,
           ROUND((i.on_hand + i.on_order) / r.avg_daily, 1) AS days_of_cover
    FROM inventory i JOIN products p USING (sku) JOIN recent r USING (sku)
    WHERE i.store = ? AND (i.on_hand + i.on_order) < ? * r.avg_daily
    ORDER BY days_of_cover
    """
    return {"store": store, "threshold_days": days_of_cover, "items": _rows(sql, (store, store, days_of_cover))}


@mcp.tool(annotations=READ_ONLY)
def get_sales_history(sku: str, store: str, days: int = 56) -> dict[str, Any]:
    """Daily units sold for a SKU at a store over the most recent `days` days (max 180)."""
    sku, store = _check_sku(sku), _check_store(store)
    if not 1 <= days <= 180:
        raise ToolError("days must be between 1 and 180.")
    rows = _rows(
        """SELECT date, units_sold FROM sales_history WHERE sku = ? AND store = ?
           ORDER BY date DESC LIMIT ?""",
        (sku, store, days),
    )
    rows.reverse()
    return {"sku": sku, "store": store, "days": len(rows), "history": rows}


@mcp.tool(annotations=READ_ONLY)
def get_supplier_terms(sku: str) -> dict[str, Any]:
    """Supplier, lead time (days), minimum order quantity (units) and case pack for a SKU."""
    sku = _check_sku(sku)
    rows = _rows("SELECT * FROM suppliers WHERE sku = ?", (sku,))
    if not rows:
        raise ToolError(f"No supplier terms for SKU {sku}.")
    return rows[0]


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
def create_purchase_order(store: str, lines: list[dict[str, Any]], approved_by: str, note: str = "") -> dict[str, Any]:
    """Record an APPROVED purchase order. `lines` = [{"sku": "GRO-001", "quantity": 48}, ...].
    Only call this after a human approved the plan; `approved_by` is required."""
    store = _check_store(store)
    if not approved_by.strip():
        raise ToolError("approved_by is required: orders need a human approver.")
    if not lines:
        raise ToolError("Purchase order has no lines.")
    po_id = f"PO-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = 0.0
    with closing(_connect(read_only=False)) as conn, conn:
        for line in lines:
            sku, qty = _check_sku(str(line.get("sku", ""))), int(line.get("quantity", 0))
            row = conn.execute(
                """SELECT i.unit_cost, s.case_pack, s.moq FROM inventory i JOIN suppliers s USING (sku)
                   WHERE i.sku = ? AND i.store = ?""",
                (sku, store),
            ).fetchone()
            if row is None:
                raise ToolError(f"SKU {sku} is not stocked at {store}.")
            if qty < row["moq"] or qty % row["case_pack"]:
                raise ToolError(f"{sku}: quantity {qty} violates MOQ {row['moq']} / case pack {row['case_pack']}.")
            conn.execute(
                "INSERT INTO purchase_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (po_id, created, store, sku, qty, row["unit_cost"], approved_by, note),
            )
            total += qty * row["unit_cost"]
    return {"po_id": po_id, "store": store, "lines": len(lines), "total_cost": round(total, 2), "status": "submitted"}


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    """Liveness probe for notebooks, Docker and Kubernetes."""
    return JSONResponse({"status": "ok", "db": config.DB_PATH.exists()})


if __name__ == "__main__":
    mcp.run(
        transport="http",  # Streamable HTTP
        host=config.MCP_HOST,
        port=config.MCP_PORT,
        path="/mcp",
        stateless_http=True,
        show_banner=False,
    )
