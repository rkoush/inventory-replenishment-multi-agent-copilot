"""Supervisor: the LLM interprets the request and plans; code enforces the guardrails.

The LLM decides *what* to do (parse store/budget/horizon, order the workers).
Deterministic code then validates the parse, adds any worker the plan forgot,
enforces dependency order, and caps the number of steps, so a bad LLM output
can't skip the optimizer or loop forever.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

import config
from agents import rules
from agents.llm import RequestPlan, llm_name
from agents.state import ReplenishState

WORKERS = rules.WORKERS
OUTPUT_KEY = {"inventory_agent": "inventory", "demand_agent": "forecast",
              "policy_agent": "policy", "optimizer_agent": "order_plan"}
DEPENDS_ON = {"inventory_agent": [], "policy_agent": [], "demand_agent": ["inventory_agent"],
              "optimizer_agent": ["inventory_agent", "demand_agent", "policy_agent"]}

SYSTEM = f"""You are the supervisor of a retail replenishment planning team.
Workers you can schedule (use these exact names):
- inventory_agent: reads stock positions and supplier terms from the inventory MCP server
- demand_agent: statistical demand forecast per SKU (needs inventory_agent first)
- policy_agent: retrieves replenishment, budget, supplier and promotion policies (RAG)
- optimizer_agent: MILP solver that computes order quantities (needs all three above)
Extract the store id, budget (USD) and planning horizon (days) from the request.
Never invent values: if the budget is not stated, return null. 'next week' means 7 days.
Return the ordered list of workers to run. Default horizon: {config.DEFAULT_HORIZON_DAYS} days."""


def _fix_plan(plan: list[str]) -> list[str]:
    """Keep known workers in the LLM's order, append missing ones, then respect dependencies."""
    ordered = [w for w in dict.fromkeys(plan) if w in WORKERS] + [w for w in WORKERS if w not in plan]
    result: list[str] = []
    while ordered:
        for w in ordered:
            if all(d in result for d in DEPENDS_ON[w]):
                result.append(w)
                ordered.remove(w)
                break
    return result


def make_supervisor(llm: Any):
    def supervisor(state: ReplenishState) -> Command[Literal[
        "inventory_agent", "demand_agent", "policy_agent", "optimizer_agent", "human_approval", "report"
    ]]:
        steps = state.get("steps", 0) + 1
        update: dict[str, Any] = {"steps": steps}
        if steps > config.MAX_SUPERVISOR_STEPS:
            return Command(goto="report", update={**update, "warnings": ["Supervisor step limit reached."]})

        if not state.get("plan"):  # first visit: plan with the LLM
            msgs = [SystemMessage(SYSTEM), HumanMessage(state["request"])]
            warnings: list[str] = []
            try:
                parsed = llm.with_structured_output(RequestPlan).invoke(msgs)
            except Exception as exc:  # LLM down / bad JSON -> deterministic fallback
                warnings.append(f"Supervisor LLM failed ({type(exc).__name__}); used rule-based parse.")
                parsed = RequestPlan(**rules.parse_request(state["request"]))
            store = parsed.store.strip().upper()
            if not re.fullmatch(r"S\d{3}", store):
                raise ValueError(f"Could not identify a valid store id in the request (got '{parsed.store}').")
            budget = parsed.budget if parsed.budget and parsed.budget > 0 else None
            horizon = parsed.horizon_days if 1 <= parsed.horizon_days <= 28 else config.DEFAULT_HORIZON_DAYS
            plan = _fix_plan(parsed.plan)
            if plan != parsed.plan:
                warnings.append(f"Supervisor plan corrected from {parsed.plan} to {plan}.")
            update.update(store=store, budget=budget, horizon_days=horizon, plan=plan, warnings=warnings)
            update["trace"] = [
                f"supervisor [{'rules fallback' if warnings and 'failed' in warnings[0] else llm_name(llm)}]: store={store}, budget={budget}, horizon={horizon}d, "
                f"plan={' -> '.join(plan)} | {parsed.reasoning}"
            ]
            state = {**state, **update}

        for worker in state["plan"]:  # route to the first worker whose output is missing
            if not state.get(OUTPUT_KEY[worker]):
                update.setdefault("trace", []).append(f"supervisor: routing to {worker}")
                return Command(goto=worker, update=update)
        update.setdefault("trace", []).append("supervisor: plan complete, requesting human approval")
        return Command(goto="human_approval", update=update)

    return supervisor
