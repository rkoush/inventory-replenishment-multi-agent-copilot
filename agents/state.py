"""Typed shared state that flows through the LangGraph supervisor graph."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class ReplenishState(TypedDict, total=False):
    # --- request (parsed by the supervisor) ---
    request: str
    store: str
    budget: Optional[float]
    horizon_days: int
    plan: list[str]  # ordered worker names chosen by the supervisor
    steps: int  # supervisor visits (loop guard)

    # --- worker outputs (each worker owns exactly one key) ---
    inventory: list[dict[str, Any]]  # inventory_agent: stock + supplier terms per SKU
    low_stock: list[dict[str, Any]]  # inventory_agent: SKUs under the cover threshold
    forecast: dict[str, dict[str, Any]]  # demand_agent: sku -> forecast
    policy: dict[str, Any]  # policy_agent: constraints + citations
    order_plan: dict[str, Any]  # optimizer_agent: MILP solution

    # --- human in the loop + output ---
    approval: dict[str, Any]
    purchase_order: dict[str, Any]
    report: str
    report_facts: dict[str, Any]

    # --- append-only audit trail ---
    trace: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
