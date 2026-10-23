"""Report node: deterministic table + a short LLM narrative over a compact plan summary.

The LLM only sees a small JSON summary (not raw tool output) and is told to use
only those numbers, which keeps the prompt tiny and the explanation grounded.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents import rules
from agents.llm import llm_name
from agents.state import ReplenishState

SYSTEM = """You are a retail replenishment analyst writing for a store manager.
Explain the order plan in at most 120 words. Use ONLY numbers that appear in the JSON;
never compute or invent numbers. Mention budget use, the biggest orders, and any SKU
still below target with its binding reason. Plain prose, no tables."""


def compact_plan(state: ReplenishState) -> dict[str, Any]:
    plan = state["order_plan"]
    return {
        "store": state["store"],
        "totals": plan["totals"],
        "lines": [
            {k: l[k] for k in ("sku", "name", "order_qty", "order_cost", "shortfall_vs_target", "binding")}
            for l in plan["lines"] if l["order_qty"] > 0 or l["shortfall_vs_target"] > 0
        ],
        "promotions": state["policy"]["promo_uplift"],
        "decision": state.get("approval", {}).get("decision"),
    }


def render_table(plan: dict[str, Any]) -> str:
    rows = ["| SKU | Item | Position | Target | Order qty | Cases | Cost | Short | Binding |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for l in plan["lines"]:
        rows.append(f"| {l['sku']} | {l['name']} | {l['position']} | {l['target']:.0f} | {l['order_qty']} | "
                    f"{l['cases']} x {l['case_pack']} | ${l['order_cost']:,.2f} | {l['shortfall_vs_target']} | "
                    f"{', '.join(l['binding']) or '-'} |")
    t = plan["totals"]
    rows.append(f"| **Total** | | | | **{t['units_ordered']}** | | **${t['total_cost']:,.2f}** | | "
                f"{t['budget_utilization_pct']}% of ${t['budget']:,.0f} |")
    return "\n".join(rows)


def make_report(llm: Any):
    def report(state: ReplenishState) -> dict[str, Any]:
        if not state.get("order_plan"):
            return {"report": "No plan was produced.", "trace": ["report: no plan"]}
        payload = json.dumps(compact_plan(state), separators=(",", ":"))
        prompt = f"<plan_json>{payload}</plan_json>"
        warnings: list[str] = []
        source = llm_name(llm)
        try:
            narrative = str(llm.invoke([SystemMessage(SYSTEM), HumanMessage(prompt)]).content).strip()
        except Exception as exc:
            warnings.append(f"Report LLM failed ({type(exc).__name__}); used template summary.")
            narrative = rules.explain_plan(json.loads(payload))
            source = "template fallback"
        decision = state.get("approval", {}).get("decision", "pending")
        po = state.get("purchase_order")
        status = f"Submitted as {po['po_id']}" if po else ("Rejected by reviewer - no order placed" if decision == "reject" else "Not submitted")
        cites = "\n".join(f"- {c}" for c in state["policy"].get("citations", []))
        text = (f"## Replenishment plan - store {state['store']}\n\n**Status:** {status}\n\n{narrative}\n\n"
                f"{render_table(state['order_plan'])}\n\n**Policy sources:**\n{cites}")
        if state.get("warnings"):
            text += "\n\n**Warnings:**\n" + "\n".join(f"- {w}" for w in state["warnings"])
        return {"report": text, "warnings": warnings, "trace": [f"report [{source}]: narrative written"]}

    return report
