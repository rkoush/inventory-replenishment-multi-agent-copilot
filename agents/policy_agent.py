"""Policy agent: RAG over policy docs -> LLM extracts constraints -> code checks grounding.

Every number the LLM extracts must literally appear in the retrieved text and be
in a sane range; otherwise it is replaced by the deterministic parser's value
(and a warning is logged). Retrieved chunks are short sections, so the prompt
stays small (free-tier friendly).
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

import config
from agents import rules
from agents.llm import PolicyExtraction, llm_name, structured
from agents.state import ReplenishState
from rag.retriever import retrieve

QUERIES = [
    "safety stock days by product category",
    "weekly replenishment budget cap and per-SKU concentration limit",
    "active promotions and expected demand uplift next week",
]

SYSTEM = """You extract replenishment planning constraints from policy excerpts.
Only use values that are explicitly written in the excerpts. Do not guess or compute.
Percentages are returned as numbers (30% -> 30)."""


def _grounded(value: float | int | None, context: str) -> bool:
    if value is None:
        return False
    forms = {f"{value:g}", f"{value:,.0f}" if float(value).is_integer() else f"{value}"}
    return any(re.search(rf"(?<![\d.]){re.escape(f)}(?![\d])", context) for f in forms)


def _validate(ext: PolicyExtraction, context: str, categories: set[str], skus: set[str]) -> tuple[dict, list[str]]:
    ref = rules.extract_policy(context)  # deterministic reference / fallback
    issues: list[str] = []

    safety = {s.category: s.days for s in ext.safety_stock
              if s.category in categories and 0 <= s.days <= 60 and _grounded(s.days, context)}
    if len(safety) < len(ext.safety_stock) or not safety:
        issues.append("safety stock values not grounded in policy text; used rule-based values")
        safety = {s["category"]: s["days"] for s in ref["safety_stock"]}

    share = ext.max_sku_budget_share_pct
    if not (share and 1 <= share <= 100 and _grounded(share, context)):
        issues.append(f"concentration limit {share!r} not grounded; used {ref['max_sku_budget_share_pct']}")
        share = ref["max_sku_budget_share_pct"]

    cap = ext.weekly_budget_cap
    if not (cap and cap > 0 and _grounded(cap, context)):
        issues.append(f"budget cap {cap!r} not grounded; used {ref['weekly_budget_cap']}")
        cap = ref["weekly_budget_cap"]

    promos = {p.sku: p.uplift_pct for p in ext.promos
              if p.sku in skus and 0 < p.uplift_pct <= 200 and _grounded(p.uplift_pct, context)}
    if len(promos) != len(ext.promos):
        issues.append("some promo uplifts not grounded; used rule-based values")
        promos = {p["sku"]: p["uplift_pct"] for p in ref["promos"]}

    return {
        "safety_stock_days": safety,
        "max_sku_budget_share": (share or config.DEFAULT_MAX_SKU_BUDGET_SHARE * 100) / 100,
        "weekly_budget_cap": cap,
        "promo_uplift": {k: v / 100 for k, v in promos.items()},
        "summary": ext.summary,
    }, issues


def make_policy_agent(llm: Any):
    def policy_agent(state: ReplenishState) -> dict[str, Any]:
        chunks: dict[str, dict] = {}
        for q in QUERIES:
            for hit in retrieve(q, k=2):
                chunks.setdefault(hit["id"], hit)
        context = "\n\n".join(f"[{c['source']} | {c['section']}]\n{c['text']}" for c in chunks.values())
        categories = {i["category"] for i in state.get("inventory", [])} or {
            "Grocery", "Household", "Personal Care", "Electronics"}
        skus = {i["sku"] for i in state.get("inventory", [])}

        warnings: list[str] = []
        source = llm_name(llm)
        try:
            ext = structured(llm, PolicyExtraction).invoke(
                [SystemMessage(SYSTEM), HumanMessage(f"Policy excerpts:\n\n{context}")]
            )
        except Exception as exc:
            warnings.append(f"Policy LLM failed ({type(exc).__name__}); used rule-based extraction.")
            ext = PolicyExtraction(**rules.extract_policy(context))
            source = "rules fallback"
        policy, issues = _validate(ext, context, categories, skus)
        warnings += [f"policy_agent: {i}" for i in issues]
        policy["citations"] = [f"{c['source']} > {c['section']}" for c in chunks.values()]
        return {
            "policy": policy,
            "warnings": warnings,
            "trace": [
                f"policy_agent [{source}]: retrieved {len(chunks)} policy sections; "
                f"safety stock days={policy['safety_stock_days']}, cap=${policy['weekly_budget_cap']}, "
                f"max SKU share={policy['max_sku_budget_share']:.0%}, promos={policy['promo_uplift']}, "
                f"grounding issues={len(issues)}"
            ],
        }

    return policy_agent
