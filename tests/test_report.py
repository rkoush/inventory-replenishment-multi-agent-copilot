"""The report states only numbers computed in code; LLM text with other numbers is rejected."""
from langchain_core.messages import AIMessage

from agents.llm import RuleBasedLLM
from agents.report import check_numbers, facts_paragraph, plan_facts
from tests.test_graph_offline import run_graph


def _state():
    lines = [
        {"sku": "HH-002", "name": "Detergent", "order_qty": 104, "order_cost": 925.60, "shortfall_vs_target": 61, "binding": ["budget"]},
        {"sku": "GRO-001", "name": "Milk", "order_qty": 272, "order_cost": 761.60, "shortfall_vs_target": 0, "binding": []},
        {"sku": "ELE-001", "name": "Cable", "order_qty": 90, "order_cost": 2312.70, "shortfall_vs_target": 52, "binding": ["concentration limit"]},
        {"sku": "HH-003", "name": "Dish soap", "order_qty": 0, "order_cost": 0.0, "shortfall_vs_target": 41, "binding": ["budget"]},
        {"sku": "PC-002", "name": "Shampoo", "order_qty": 0, "order_cost": 0.0, "shortfall_vs_target": 0, "binding": []},
    ]
    totals = {"total_cost": 3999.90, "budget": 4000.0, "budget_utilization_pct": 100.0, "max_sku_share": 0.3,
              "units_ordered": 466, "skus_ordered": 3}
    return {"store": "S001", "order_plan": {"lines": lines, "totals": totals},
            "policy": {"promo_uplift": {"HH-002": 0.4}, "safety_stock_days": {"Grocery": 3}}}


def test_facts_are_computed_in_code():
    f = plan_facts(_state())
    assert (f["n_skus_total"], f["n_skus_ordered"], f["n_skus_zero"], f["total_units"]) == (5, 3, 2, 466)
    assert (f["total_cost"], f["budget"], f["budget_utilization"]) == ("$3,999.90", "$4,000.00", "100.0%")
    assert [o["sku"] for o in f["top_orders"]] == ["ELE-001", "HH-002", "GRO-001"]
    assert f["top_orders"][0]["cost"] == "$2,312.70" and f["top_orders"][0]["share_of_budget"] == "57.8%"
    assert f["zero_qty_skus"] == [{"sku": "HH-003", "reason": "budget"}, {"sku": "PC-002", "reason": "already at target"}]
    text = facts_paragraph(f)
    assert "orders 3 of 5 SKUs (466 units) for $3,999.90 of a $4,000.00 budget (100.0% used)" in text
    assert "2 SKUs get no order: HH-003 (budget), PC-002 (already at target)" in text


def test_check_numbers_rejects_the_live_run_mistakes():
    f = plan_facts(_state())
    assert check_numbers("All eight SKUs are ordered.", f)  # wrong count word
    assert check_numbers("All SKUs are ordered within budget.", f)  # HH-003 has qty 0
    assert check_numbers("The plan spends 3999.9 of the budget.", f)  # dropped $
    assert check_numbers("ELE-001 has the highest share at 0.3 of the budget.", f)  # fraction, not a fact
    assert check_numbers("HH-002 costs $925.00.", f)  # rounded / invented amount
    assert check_numbers("Utilization is 99%.", f)


def test_check_numbers_accepts_grounded_text():
    f = plan_facts(_state())
    ok = ("Budget limited HH-002 and HH-003, while ELE-001 hit the 30.0% concentration limit at $2,312.70. "
          "The promotion on HH-002 raised its target by 40.0%, and PC-002 was already at target, "
          "so three SKUs were ordered and two were not.")
    assert check_numbers(ok, f) == []
    assert check_numbers("Spend was $3,999.90 of $4,000.00 (100.0% used).", f) == []


class _SloppyLLM(RuleBasedLLM):
    name = "sloppy"

    def invoke(self, messages, **_):
        return AIMessage(content="All eight SKUs are ordered, using 3999.9 with the highest share at 0.3 of the budget.")


def test_report_falls_back_when_llm_invents_numbers(mcp_url):
    state, _ = run_graph(_SloppyLLM(), mcp_url, ["approve"])
    f = state["report_facts"]
    assert any("Report LLM text rejected" in w for w in state["warnings"])
    assert "All eight SKUs" not in state["report"] and "3999.9 " not in state["report"]
    assert f"orders {f['n_skus_ordered']} of {f['n_skus_total']} SKUs" in state["report"]
    assert f["total_cost"] in state["report"] and "**Why:**" in state["report"]


def test_offline_report_passes_its_own_check(mcp_url):
    state, _ = run_graph(RuleBasedLLM(), mcp_url, ["approve"])
    assert not any("rejected" in w for w in state["warnings"])
    assert "report [rule-based (offline)]" in " ".join(state["trace"])
