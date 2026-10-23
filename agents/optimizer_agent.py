"""Optimizer agent: assembles solver inputs from state and runs the MILP."""
from __future__ import annotations

from typing import Any

import config
from agents.milp import SkuInput, solve_replenishment
from agents.state import ReplenishState

PENALTY_MULTIPLIER = {"Grocery": 4.0}  # service-critical staples (replenishment policy)


def build_inputs(state: ReplenishState) -> list[SkuInput]:
    policy, forecast = state["policy"], state["forecast"]
    items = []
    for inv in state["inventory"]:
        fc = forecast[inv["sku"]]
        uplift = policy["promo_uplift"].get(inv["sku"], 0.0)
        ss_days = policy["safety_stock_days"].get(inv["category"], config.DEFAULT_SAFETY_STOCK_DAYS)
        items.append(SkuInput(
            sku=inv["sku"], name=inv["name"], category=inv["category"], unit_cost=inv["unit_cost"],
            on_hand=inv["on_hand"], on_order=inv["on_order"], shelf_capacity=inv["shelf_capacity"],
            case_pack=inv["case_pack"], moq=inv["moq"], lead_time_days=inv["lead_time_days"],
            demand_lead=fc["demand_lead"],
            demand_cover=fc["demand_lead"] + fc["demand_horizon"] * (1 + uplift),  # promo applies to the planning week
            safety_stock=ss_days * fc["avg_daily"],
            penalty_per_unit=inv["unit_cost"] * PENALTY_MULTIPLIER.get(inv["category"], config.SHORTAGE_PENALTY_MULTIPLIER),
        ))
    return items


def optimizer_agent(state: ReplenishState) -> dict[str, Any]:
    policy = state["policy"]
    cap = policy.get("weekly_budget_cap")
    budget, warnings = state.get("budget"), []
    if budget is None:
        budget = cap
        warnings.append(f"No budget in request; using policy cap ${cap:,.0f}.")
    elif cap and budget > cap:
        warnings.append(f"Requested budget ${budget:,.0f} exceeds policy cap ${cap:,.0f}; capped (needs regional approval).")
        budget = cap
    plan = solve_replenishment(build_inputs(state), budget=budget, max_sku_share=policy["max_sku_budget_share"])
    if plan["status"] != "Optimal":
        raise RuntimeError(f"Solver returned status {plan['status']}")
    t = plan["totals"]
    short = [l["sku"] for l in plan["lines"] if l["shortfall_vs_target"] > 0]
    return {
        "order_plan": plan,
        "warnings": warnings,
        "trace": [
            f"optimizer_agent [PuLP/{plan['solver']}]: {plan['status']}; {t['skus_ordered']} SKUs, "
            f"{t['units_ordered']} units, ${t['total_cost']:,.2f} of ${t['budget']:,.2f} "
            f"({t['budget_utilization_pct']}%); below target: {short or 'none'}"
        ],
    }
