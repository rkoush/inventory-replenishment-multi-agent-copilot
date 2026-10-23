"""Inventory agent: stock positions and supplier terms, fetched through MCP tools."""
from __future__ import annotations

from typing import Any

from agents.mcp_client import mcp_session
from agents.state import ReplenishState


def make_inventory_agent(client: Any = None):
    async def inventory_agent(state: ReplenishState) -> dict[str, Any]:
        store = state["store"]
        async with mcp_session(client, role="planner") as tools:
            stock = (await tools.call("list_store_inventory", store=store))["items"]
            low = (await tools.call("list_low_stock", store=store, days_of_cover=7))["items"]
            for item in stock:
                terms = await tools.call("get_supplier_terms", sku=item["sku"])
                item.update({k: terms[k] for k in ("supplier", "lead_time_days", "moq", "case_pack")})
        low_txt = ", ".join(f"{i['sku']} ({i['days_of_cover']}d)" for i in low) or "none"
        return {
            "inventory": stock,
            "low_stock": low,
            "trace": [
                f"inventory_agent: MCP as {tools.auth}; tools visible {tools.names}",
                f"inventory_agent: {len(stock)} SKUs at {store}; {len(tools.calls)} tool calls; "
                f"under 7 days of cover: {low_txt}",
            ],
        }

    return inventory_agent
