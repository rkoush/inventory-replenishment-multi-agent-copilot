"""Report node: every number is computed in code; the LLM only explains *why*.

1. `plan_facts()` computes the facts: counts, units, $ amounts, %, top orders, shortfalls.
2. `facts_paragraph()` renders those facts as fixed template sentences.
3. The LLM gets the facts JSON and writes a 2-4 sentence "why" paragraph.
   `check_numbers()` rejects that paragraph if it has any number that isn't in the facts,
   a $ amount without "$", a share written as a fraction, or "all SKUs ordered" when some
   SKUs got qty 0. A rejected or failed reply falls back to `rules.explain_why()`.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents import rules
from agents.llm import llm_name, strip_reasoning
from agents.state import ReplenishState

SYSTEM = """You are a retail replenishment analyst writing for a store manager.
The reader already sees the key numbers (counts, totals, top orders, shortfalls).
Write 2-4 plain sentences explaining WHY the optimizer chose this plan: which constraint
limited each short SKU (budget, shelf capacity, concentration limit), how promotions and
safety stock shaped the targets, and why some SKUs got no order.
Rules: refer to SKUs by ID. Prefer no numbers. If you use a number, copy it EXACTLY from
the JSON including its $ or % sign. Never compute, round or invent numbers. No tables."""

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def money(x: float) -> str:
    return f"${x:,.2f}"


def units(n: int) -> str:
    return f"{n} unit{'s' if n != 1 else ''}"


def pct(x: float) -> str:
    return f"{x:.1f}%"


def plan_facts(state: ReplenishState) -> dict[str, Any]:
    """All numbers that the report states, computed from the solver output."""
    plan, policy = state["order_plan"], state.get("policy", {})
    t, lines = plan["totals"], plan["lines"]
    budget = t["budget"]
    ordered = sorted((l for l in lines if l["order_qty"] > 0), key=lambda l: -l["order_cost"])
    zero = [l for l in lines if l["order_qty"] <= 0]
    return {
        "store": state["store"],
        "n_skus_total": len(lines),
        "n_skus_ordered": len(ordered),
        "n_skus_zero": len(zero),
        "zero_qty_skus": [{"sku": l["sku"], "reason": "already at target" if l["shortfall_vs_target"] <= 0
                           else ", ".join(l["binding"]) or "solver choice"} for l in zero],
        "total_units": sum(l["order_qty"] for l in lines),
        "total_cost": money(t["total_cost"]),
        "budget": money(budget),
        "budget_utilization": pct(100 * t["total_cost"] / budget if budget else 0.0),
        "max_sku_share_limit": pct(100 * t.get("max_sku_share", policy.get("max_sku_budget_share", 0))),
        "top_orders": [{"sku": l["sku"], "name": l["name"], "units": l["order_qty"], "cost": money(l["order_cost"]),
                        "share_of_budget": pct(100 * l["order_cost"] / budget if budget else 0.0)}
                       for l in ordered[:3]],
        "shortfalls": [{"sku": l["sku"], "short_units": l["shortfall_vs_target"], "binding": l["binding"]}
                       for l in lines if l["shortfall_vs_target"] > 0],
        "promotions": {k: pct(100 * v) for k, v in policy.get("promo_uplift", {}).items()},
        "safety_stock_days": policy.get("safety_stock_days", {}),
        "decision": state.get("approval", {}).get("decision", "pending"),
    }


def facts_paragraph(f: dict[str, Any]) -> str:
    s = (f"The plan orders {f['n_skus_ordered']} of {f['n_skus_total']} SKUs ({f['total_units']} units) "
         f"for {f['total_cost']} of a {f['budget']} budget ({f['budget_utilization']} used).")
    if f["n_skus_zero"]:
        s += f" {f['n_skus_zero']} SKU{'s' if f['n_skus_zero'] != 1 else ''} get no order: " + ", ".join(
            f"{z['sku']} ({z['reason']})" for z in f["zero_qty_skus"]) + "."
    if f["top_orders"]:
        s += " Largest orders by cost: " + ", ".join(
            f"{o['sku']} ({units(o['units'])}, {o['cost']}, {o['share_of_budget']} of budget)"
            for o in f["top_orders"]) + f"; the per-SKU limit is {f['max_sku_share_limit']} of budget."
    if f["shortfalls"]:
        s += " Still below target: " + ", ".join(
            f"{x['sku']} (short {units(x['short_units'])}, {', '.join(x['binding']) or 'n/a'})"
            for x in f["shortfalls"]) + "."
    else:
        s += " Every SKU reaches its order-up-to target."
    return s


# ---------------------------------------------------------------- number check
_ID = re.compile(r"\b[A-Z]{1,4}-?\d+\b")  # SKU / store / PO ids like HH-002, S001
_NUM = re.compile(r"(\$\s?)?(\d[\d,]*(?:\.\d+)?)\s*(%|percent\b)?", re.I)


def _values(f: dict[str, Any]) -> tuple[set[float], set[float], set[float]]:
    """(money, percent, plain) values that appear in the facts."""
    m, p, n = set(), set(), set()

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, bool):
            return
        elif isinstance(v, (int, float)):
            n.add(float(v))
        elif isinstance(v, str):
            for mm in _NUM.finditer(_ID.sub(" ", v)):
                val = float(mm.group(2).replace(",", ""))
                (m if mm.group(1) else p if mm.group(3) else n).add(val)

    walk(f)
    return m, p, n


def check_numbers(text: str, f: dict[str, Any]) -> list[str]:
    """Problems with numbers in LLM text vs the facts. An empty list means OK."""
    m, p, n = _values(f)
    problems = []
    for mm in _NUM.finditer(_ID.sub(" ", text)):
        raw, val = mm.group(0).strip(), float(mm.group(2).replace(",", ""))
        if mm.group(1):
            ok = val in m
        elif mm.group(3):
            ok = val in p
        else:
            ok = val in n and val not in m  # a bare money amount means a dropped "$"
        if not ok:
            problems.append(f"number not in facts: {raw!r}")
    for w in re.findall(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", text, flags=re.I):
        if float(NUMBER_WORDS[w.lower()]) not in n:
            problems.append(f"count word not in facts: {w!r}")
    if f["n_skus_zero"] and re.search(r"\b(all|every)\b[^.]*\bSKUs?\b[^.]*\b(ordered|replenished)\b", text, re.I):
        problems.append("claims all SKUs are ordered, but some have qty 0")
    return problems


# ---------------------------------------------------------------- rendering
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


def escape_dollars(text: str) -> str:
    """Escape `$` so Jupyter/Colab Markdown doesn't render "$4,000 ... $5,000" as LaTeX.

    Use only for notebook display; the CLI prints the raw report.
    """
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def make_report(llm: Any):
    def report(state: ReplenishState) -> dict[str, Any]:
        if not state.get("order_plan"):
            return {"report": "No plan was produced.", "trace": ["report: no plan"]}
        facts = plan_facts(state)
        payload = json.dumps(facts, separators=(",", ":"))
        warnings: list[str] = []
        source = llm_name(llm)
        try:
            why = strip_reasoning(str(llm.invoke(
                [SystemMessage(SYSTEM), HumanMessage(f"<facts_json>{payload}</facts_json>")]).content))
            if not why:  # e.g. reasoning used up the whole token budget
                raise ValueError("empty LLM reply")
            if problems := check_numbers(why, facts):
                raise ValueError("; ".join(problems[:3]))
        except Exception as exc:
            warnings.append(f"Report LLM text rejected ({type(exc).__name__}: {exc}); used template explanation.")
            why = rules.explain_why(facts)
            source = "template fallback"
        decision = facts["decision"]
        po = state.get("purchase_order")
        status = f"Submitted as {po['po_id']}" if po else ("Rejected by reviewer - no order placed" if decision == "reject" else "Not submitted")
        cites = "\n".join(f"- {c}" for c in state["policy"].get("citations", []))
        text = (f"## Replenishment plan - store {state['store']}\n\n**Status:** {status}\n\n"
                f"{facts_paragraph(facts)}\n\n**Why:** {why}\n\n"
                f"{render_table(state['order_plan'])}\n\n**Policy sources:**\n{cites}")
        all_warnings = list(state.get("warnings", [])) + warnings
        if all_warnings:
            text += "\n\n**Warnings:**\n" + "\n".join(f"- {w}" for w in all_warnings)
        return {"report": text, "report_facts": facts, "warnings": warnings,
                "trace": [f"report [{source}]: facts computed in code; explanation written"]}

    return report
