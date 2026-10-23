"""Deterministic, rule-based stand-ins for the LLM steps.

Used (1) as the offline "LLM" so the whole graph runs without an API key, and
(2) as a fallback/validator when the real LLM fails or returns ungrounded values.
"""
from __future__ import annotations

import json
import re
from typing import Any

WORKERS = ["inventory_agent", "demand_agent", "policy_agent", "optimizer_agent"]


def parse_request(text: str) -> dict[str, Any]:
    store = re.search(r"\b(S\d{3})\b", text, flags=re.I)
    budget = re.search(r"\$\s?([\d,]+(?:\.\d+)?)\s*([kK])?\b", text)
    horizon = 7
    if m := re.search(r"(\d+)\s*(day|week)s?", text, flags=re.I):
        horizon = int(m.group(1)) * (7 if m.group(2).lower() == "week" else 1)
    elif re.search(r"two weeks|fortnight", text, flags=re.I):
        horizon = 14
    value = None
    if budget:
        value = float(budget.group(1).replace(",", "")) * (1000 if budget.group(2) else 1)
    return {
        "store": store.group(1).upper() if store else "",
        "budget": value,
        "horizon_days": horizon,
        "plan": list(WORKERS),
        "reasoning": "Rule-based parse: fetch stock, forecast demand, load policy, then optimize.",
    }


def extract_policy(context: str) -> dict[str, Any]:
    safety = [
        {"category": c.strip(), "days": int(d)}
        for c, d in re.findall(r"^-\s*([A-Za-z ]+):\s*(\d+)\s+days? of safety stock", context, flags=re.M)
    ]
    share = re.search(r"more than (\d+(?:\.\d+)?)% of the weekly", context)
    cap = re.search(r"budget cap of \$([\d,]+(?:\.\d+)?)", context)
    promos = [
        {"sku": s, "uplift_pct": float(p)}
        for s, p in re.findall(r"^-\s*([A-Z]{2,4}-\d{3})[^\n]*?uplift of (\d+(?:\.\d+)?)%", context, flags=re.M)
    ]
    return {
        "safety_stock": safety,
        "max_sku_budget_share_pct": float(share.group(1)) if share else None,
        "weekly_budget_cap": float(cap.group(1).replace(",", "")) if cap else None,
        "promos": promos,
        "summary": "Constraints extracted with deterministic rules from the retrieved policy sections.",
    }


def explain_why(facts: dict[str, Any]) -> str:
    """Number-free 'why' paragraph built from the report facts (offline LLM + fallback)."""
    by_reason: dict[str, list[str]] = {}
    for x in facts["shortfalls"]:
        for r in x["binding"] or ["solver choice"]:
            by_reason.setdefault(r, []).append(x["sku"])
    parts = [f"{r} limited {', '.join(skus)}" for r, skus in by_reason.items()]
    msg = ("The solver covers lead-time demand first and weights Grocery higher. "
           + (("Below target, " + "; ".join(parts) + ". ") if parts else ""))
    if facts["promotions"]:
        msg += f"Promotions on {', '.join(facts['promotions'])} raised their targets. "
    at_target = [z["sku"] for z in facts["zero_qty_skus"] if z["reason"] == "already at target"]
    idle = [z["sku"] for z in facts["zero_qty_skus"] if z["reason"] != "already at target"]
    if at_target:
        msg += f"{', '.join(at_target)} already had enough stock, so {'they' if len(at_target) > 1 else 'it'} got no order. "
    if idle:
        msg += f"{', '.join(idle)} got no order because spend went to SKUs with less cover. "
    return msg + "All quantities respect case packs, MOQs and shelf capacity."


def extract_json_block(text: str, tag: str = "facts_json") -> dict[str, Any]:
    """Pull the JSON payload out of a prompt built by the report node."""
    start = text.index(f"<{tag}>") + len(f"<{tag}>")
    end = text.index(f"</{tag}>")
    return json.loads(text[start:end])
