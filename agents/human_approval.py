"""Human-in-the-loop gate (LangGraph interrupt) and the order-placement step.

The graph pauses here with the proposed plan; the checkpointer persists state,
and the caller resumes with Command(resume=...):
    "approve" | "reject" | {"decision": "approve"|"reject"|"revise", "budget": 4000,
                            "approver": "jane.doe", "comment": "..."}
Nothing is written to the supplier system unless a human approves.
"""
from __future__ import annotations

import sys
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.types import Command, interrupt

from agents.mcp_client import mcp_session
from agents.state import ReplenishState


def _normalise(answer: Any) -> dict[str, Any]:
    if isinstance(answer, str):
        a = answer.strip().lower()
        decision = "approve" if a in {"y", "yes", "approve", "approved", "ok"} else "reject"
        return {"decision": decision, "approver": "planner"}
    answer = dict(answer or {})
    answer.setdefault("approver", "planner")
    answer["decision"] = str(answer.get("decision", "reject")).lower()
    return answer


def human_approval(state: ReplenishState, config: RunnableConfig) -> Command[Literal["place_order", "optimizer_agent", "report"]]:
    if sys.version_info < (3, 11):
        # Python 3.10: the runnable config contextvar isn't propagated into sync nodes
        # run from an async graph, which interrupt() relies on. Set it explicitly.
        var_child_runnable_config.set(config)
    plan = state["order_plan"]
    answer = _normalise(interrupt({
        "question": "Approve this replenishment plan? (approve / reject / revise with a new budget)",
        "store": state["store"],
        "totals": plan["totals"],
        "lines": [{k: l[k] for k in ("sku", "name", "order_qty", "order_cost", "shortfall_vs_target")}
                  for l in plan["lines"] if l["order_qty"] > 0 or l["shortfall_vs_target"] > 0],
        "warnings": state.get("warnings", []),
    }))
    trace = [f"human_approval: {answer['decision']} by {answer['approver']}"]
    if answer["decision"] == "revise" and answer.get("budget"):
        return Command(goto="optimizer_agent", update={
            "budget": float(answer["budget"]), "approval": answer,
            "trace": trace + [f"human_approval: re-optimising with budget ${float(answer['budget']):,.0f}"]})
    if answer["decision"] == "approve":
        return Command(goto="place_order", update={"approval": answer, "trace": trace})
    return Command(goto="report", update={"approval": answer, "trace": trace})


def make_place_order(client: Any = None):
    async def place_order(state: ReplenishState) -> dict[str, Any]:
        lines = [{"sku": l["sku"], "quantity": l["order_qty"]} for l in state["order_plan"]["lines"] if l["order_qty"] > 0]
        if not lines:
            return {"trace": ["place_order: nothing to order"]}
        async with mcp_session(client) as tools:
            po = await tools.call("create_purchase_order", store=state["store"], lines=lines,
                                  approved_by=state["approval"]["approver"],
                                  note=state["approval"].get("comment", ""))
        return {"purchase_order": po,
                "trace": [f"place_order: MCP create_purchase_order -> {po['po_id']} (${po['total_cost']:,.2f})"]}

    return place_order
