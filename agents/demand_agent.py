"""Demand agent: deterministic per-SKU forecast from MCP sales history (no LLM)."""
from __future__ import annotations

from typing import Any

from agents.forecasting import forecast_sku
from agents.mcp_client import mcp_session
from agents.state import ReplenishState


def make_demand_agent(client: Any = None):
    async def demand_agent(state: ReplenishState) -> dict[str, Any]:
        store, horizon = state["store"], state.get("horizon_days", 7)
        forecasts: dict[str, dict[str, Any]] = {}
        async with mcp_session(client, role="planner") as tools:
            for item in state["inventory"]:
                hist = await tools.call("get_sales_history", sku=item["sku"], store=store, days=84)
                forecasts[item["sku"]] = forecast_sku(hist["history"], item["lead_time_days"], horizon)
        wape = sum(f["backtest_wape_7d"] for f in forecasts.values()) / len(forecasts)
        return {
            "forecast": forecasts,
            "trace": [
                f"demand_agent: forecast {len(forecasts)} SKUs over lead time + {horizon}d "
                f"(DOW-seasonal exponential smoothing); mean 7-day backtest WAPE {wape:.1%}; MCP as {tools.auth}"
            ],
        }

    return demand_agent
