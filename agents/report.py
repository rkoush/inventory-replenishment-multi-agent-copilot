"""Report node: every number is computed in code; the LLM only explains *why*.

1. `plan_facts()` computes the facts: counts, units, $ amounts, %, top orders, shortfalls.
2. `facts_paragraph()` renders those facts as fixed template sentences.
3. The LLM gets the facts JSON and writes a 2-4 sentence "why" paragraph.
   `check_numbers()` parses each number as one token ($4,000 / 3,999.90 / 23.1%), ignores
   SKU/store/PO codes, and rejects the paragraph if a number doesn't match a fact
   numerically, a count word is wrong, or it says "all SKUs ordered" when some got qty 0. A rejected or failed reply falls back to `rules.explain_why()`.
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
# LLMs (gpt-oss in particular) often write U+2011 non-breaking hyphens ("GRO\u2011004"),
# dashes, and narrow/no-break spaces ("$4\u202f000"); normalize before tokenizing.
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe63\uff0d"), "-")
_SPACES = dict.fromkeys(map(ord, "\u00a0\u2007\u2009\u202f"), " ")
# Ids are removed before extracting numbers: PO ids, SKU / store codes like GRO-001, HH-002, S001.
_ID = re.compile(r"\bPO-\d{8}-[A-Z0-9]+\b|\b[A-Z]{1,4}-?\d+\b")
# A number is one whole token: 1,234,567.89 | 3999.90 | 23.1 | 8 (never "004," or "4" + "000").
_NUM = re.compile(r"(?<!\d)(?<!\d[.,])(\$\s?)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\d])(\s*(?:%|percent\b|per cent\b))?",
                  re.I)
# Count words that must match a count in the facts. "one" is exempt: it is mostly not a count
# ("one reason", "the one SKU"). Digits are never exempt, even small ones (see check_numbers).
NUMBER_WORDS_CHECKED = {w: v for w, v in NUMBER_WORDS.items() if w != "one"}


def _normalize(text: str) -> str:
    text = re.sub("(?<=\\d)[\u00a0\u2009\u202f](?=\\d{3}(?!\\d))", ",", text)  # "4\u202f000" -> "4,000"
    return text.translate(_DASHES).translate(_SPACES)


def _tokens(text: str):
    """(raw, value, kind, decimals) for each number in text; kind is 'money' | 'pct' | 'plain'."""
    for mm in _NUM.finditer(_ID.sub(" ", _normalize(text))):
        digits = mm.group(2)
        decimals = len(digits.partition(".")[2])
        kind = "money" if mm.group(1) else "pct" if mm.group(3) else "plain"
        yield mm.group(0).strip(), float(digits.replace(",", "")), kind, decimals


def _values(f: dict[str, Any]) -> tuple[set[float], set[float], set[float]]:
    """(money, percent, plain) values in the facts. Plain includes counts and the length of
    every list/dict in the facts (e.g. "3 largest orders", "2 SKUs short")."""
    m, p, n = set(), set(), set()

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            n.add(float(len(v)))
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            n.add(float(len(v)))
            for x in v:
                walk(x)
        elif isinstance(v, bool):
            return
        elif isinstance(v, (int, float)):
            n.add(float(v))
        elif isinstance(v, str):
            for _, val, kind, _ in _tokens(v):
                {"money": m, "pct": p, "plain": n}[kind].add(val)

    walk(f)
    return m, p, n


def _close(x: float, values: set[float], decimals: int | None = None) -> bool:
    """x equals a fact; with `decimals`, a fact rounded to that many places also counts (23 for 23.1%)."""
    return any(abs(x - v) < 1e-6 or (decimals is not None and abs(x - round(v, decimals)) < 1e-6) for v in values)


def check_numbers(text: str, f: dict[str, Any]) -> list[str]:
    """Problems with numbers in LLM text vs the facts. An empty list means OK.

    Every number is parsed as one token ("$3,999.90", "23.1%", "8") and compared numerically:
    - "$X" must equal a $ fact to the cent ("$4,000" == "$4,000.00").
    - "X%" must equal a % fact, or that fact rounded to the digits written ("23%" for 23.1%).
    - a bare X may be any fact value: a count, units, a $ amount without "$", a % as 30 or 0.3.
    Small integers are NOT exempt: miscounts ("9 of 10 SKUs") are a typical LLM error.
    """
    m, p, n = _values(f)
    fractions = {v / 100 for v in p}
    problems = []
    for raw, val, kind, decimals in _tokens(text):
        if kind == "money":
            ok = _close(val, m)
        elif kind == "pct":
            ok = _close(val, p, decimals)
        else:
            ok = _close(val, n | m | p | fractions)
        if not ok:
            problems.append(f"number not in facts: {raw!r}")
    norm = _normalize(text)
    for w in re.findall(r"\b(" + "|".join(NUMBER_WORDS_CHECKED) + r")\b", norm, flags=re.I):
        if float(NUMBER_WORDS_CHECKED[w.lower()]) not in n:
            problems.append(f"count word not in facts: {w!r}")
    if f["n_skus_zero"] and re.search(r"\b(all|every)\b[^.]*\bSKUs?\b[^.]*\b(ordered|replenished)\b", norm, re.I):
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
