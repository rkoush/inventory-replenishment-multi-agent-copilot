"""Full graph runs with no API key: RuleBasedLLM + a 'bad' fake LLM to exercise guardrails."""
import asyncio
import uuid

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from agents.graph import build_graph
from agents.llm import PolicyExtraction, RequestPlan, RuleBasedLLM
from agents.mcp_client import make_client

REQUEST = "Plan next week's replenishment for store S001 with a $5,000 budget"


def run_graph(llm, mcp_url, decisions):
    graph = build_graph(llm=llm, mcp_client=make_client(mcp_url, token=""))
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}

    async def go():
        await graph.ainvoke({"request": REQUEST}, cfg)
        interrupts = []
        for d in decisions:
            snap = await graph.aget_state(cfg)
            if not snap.interrupts:
                break
            interrupts.append(snap.interrupts[0].value)
            await graph.ainvoke(Command(resume=d), cfg)
        return (await graph.aget_state(cfg)).values, interrupts

    return asyncio.run(go())


def test_offline_approve_places_order(mcp_url):
    state, interrupts = run_graph(RuleBasedLLM(), mcp_url, ["approve"])
    assert len(interrupts) == 1 and interrupts[0]["store"] == "S001"
    plan = state["order_plan"]
    assert plan["status"] == "Optimal" and plan["totals"]["total_cost"] <= 5000
    assert state["purchase_order"]["po_id"].startswith("PO-")
    assert state["policy"]["promo_uplift"] == {"HH-002": 0.4, "ELE-001": 0.25}
    assert "Submitted as PO-" in state["report"]


def test_reject_places_no_order(mcp_url):
    state, _ = run_graph(RuleBasedLLM(), mcp_url, ["reject"])
    assert "purchase_order" not in state and "Rejected" in state["report"]


def test_revise_reoptimises_with_new_budget(mcp_url):
    state, interrupts = run_graph(RuleBasedLLM(), mcp_url, [{"decision": "revise", "budget": 2500}, "approve"])
    assert len(interrupts) == 2
    assert interrupts[1]["totals"]["budget"] == 2500
    assert state["order_plan"]["totals"]["total_cost"] <= 2500


class SloppyLLM:
    """Fake LLM that skips workers and hallucinates policy numbers."""

    name = "sloppy-fake"

    def with_structured_output(self, schema, **_):
        if schema is RequestPlan:
            out = RequestPlan(store="s001", budget=5000, horizon_days=7, plan=["optimizer_agent"], reasoning="go fast")
        else:
            out = PolicyExtraction(safety_stock=[{"category": "Grocery", "days": 99}], max_sku_budget_share_pct=55,
                                   weekly_budget_cap=9999, promos=[{"sku": "GRO-001", "uplift_pct": 300}],
                                   summary="made up")
        return RunnableLambda(lambda _: out)

    def invoke(self, *_a, **_k):
        return AIMessage(content="Narrative.")


def test_guardrails_fix_bad_plan_and_ungrounded_policy(mcp_url):
    state, _ = run_graph(SloppyLLM(), mcp_url, ["reject"])
    assert state["plan"] == ["inventory_agent", "demand_agent", "policy_agent", "optimizer_agent"]
    p = state["policy"]
    assert p["safety_stock_days"]["Grocery"] == 3 and p["weekly_budget_cap"] == 5000
    assert p["max_sku_budget_share"] == 0.30 and "GRO-001" not in p["promo_uplift"]
    assert any("not grounded" in w for w in state["warnings"])
