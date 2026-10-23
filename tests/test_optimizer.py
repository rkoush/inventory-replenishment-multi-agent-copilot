from agents.forecasting import forecast_sku
from agents.milp import SkuInput, solve_replenishment


def item(**kw):
    base = dict(sku="GRO-001", name="Milk", category="Grocery", unit_cost=2.0, on_hand=10, on_order=0,
                shelf_capacity=500, case_pack=4, moq=24, lead_time_days=2, demand_lead=20.0,
                demand_cover=100.0, safety_stock=20.0, penalty_per_unit=8.0)
    base.update(kw)
    return SkuInput(**base)


def test_orders_case_pack_multiples_up_to_target():
    plan = solve_replenishment([item()], budget=1000)
    line = plan["lines"][0]
    assert plan["status"] == "Optimal"
    assert line["order_qty"] % 4 == 0 and line["order_qty"] >= 24
    assert line["shortfall_vs_target"] == 0  # target 120 - position 10 = 110 -> 112 units


def test_budget_is_respected_and_reported():
    items = [item(sku=f"GRO-00{i}") for i in range(1, 5)]
    plan = solve_replenishment(items, budget=300)
    assert plan["totals"]["total_cost"] <= 300
    assert any("budget" in l["binding"] for l in plan["lines"])


def test_moq_blocks_tiny_orders():
    plan = solve_replenishment([item(demand_cover=12.0, safety_stock=0.0, on_hand=5, penalty_per_unit=2.5)], budget=1000)
    qty = plan["lines"][0]["order_qty"]
    assert qty == 0 or qty >= 24


def test_shelf_capacity_caps_order():
    plan = solve_replenishment([item(shelf_capacity=60, demand_cover=400.0)], budget=10_000)
    line = plan["lines"][0]
    assert 10 + line["order_qty"] - 20 <= 60
    assert "shelf capacity" in line["binding"]


def test_concentration_limit():
    plan = solve_replenishment([item(demand_cover=1000.0, shelf_capacity=5000)], budget=1000, max_sku_share=0.3)
    assert plan["lines"][0]["order_cost"] <= 300


def test_forecast_is_deterministic_and_positive():
    hist = [{"date": f"2026-08-{d:02d}", "units_sold": 10 + (d % 7)} for d in range(1, 32)]
    a, b = forecast_sku(hist, lead_time_days=3), forecast_sku(hist, lead_time_days=3)
    assert a == b and a["avg_daily"] > 0 and a["demand_lead"] > 0
