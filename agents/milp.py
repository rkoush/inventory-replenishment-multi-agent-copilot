"""Replenishment MILP (PuLP). The solver, not the LLM, decides every quantity.

For each SKU i:
  cases_i  integer >= 0          order quantity q_i = case_pack_i * cases_i
  y_i      binary                1 if SKU i is ordered (needed for MOQ)
  gapD_i   >= 0                  units short of forecast demand (lead + horizon)
  gapT_i   >= 0                  units short of target (= demand + safety stock)

minimise   sum_i cost_i*q_i + penalty_i*(gapD_i + 0.5*gapT_i)
s.t.       q_i >= moq_i * y_i,  q_i <= M_i * y_i                     (MOQ)
           position_i + q_i - demand_lead_i <= shelf_capacity_i      (fits on shelf on arrival)
           gapD_i >= demand_cover_i - position_i - q_i
           gapT_i >= target_i - position_i - q_i
           cost_i * q_i <= max_sku_share * budget                     (concentration limit)
           sum_i cost_i * q_i <= budget                               (budget)

Missing forecast demand is penalised 1.5x as heavily as missing safety stock,
so under a tight budget the solver covers real demand first.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import pulp

# PuLP 3.x warns about its upcoming 4.0 API on every variable; the 3.x API is what we pin.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pulp(\..*)?")


@dataclass
class SkuInput:
    sku: str
    name: str
    category: str
    unit_cost: float
    on_hand: int
    on_order: int
    shelf_capacity: int
    case_pack: int
    moq: int
    lead_time_days: int
    demand_lead: float  # forecast units sold before the order arrives
    demand_cover: float  # forecast units over lead time + planning horizon (incl. promo uplift)
    safety_stock: float  # units
    penalty_per_unit: float  # cost of one unit of unmet demand


def _solver() -> pulp.LpSolver:
    available = pulp.listSolvers(onlyAvailable=True)
    for name in ("PULP_CBC_CMD", "HiGHS", "COIN_CMD", "GLPK_CMD"):
        if name in available:
            return pulp.getSolver(name, msg=False, timeLimit=20)
    raise RuntimeError(f"No MILP solver available to PuLP (found {available}). Install `pulp>=3.3,<4` (bundles CBC).")


def solve_replenishment(items: list[SkuInput], budget: float, max_sku_share: float = 1.0) -> dict[str, Any]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    prob = pulp.LpProblem("replenishment", pulp.LpMinimize)
    cases, y, gap_d, gap_t, qty = {}, {}, {}, {}, {}
    for it in items:
        position = it.on_hand + it.on_order
        room = max(0.0, it.shelf_capacity + it.demand_lead - position)  # units that fit on arrival
        max_cases = int(room // it.case_pack)
        cases[it.sku] = pulp.LpVariable(f"cases_{it.sku}", lowBound=0, upBound=max_cases, cat="Integer")
        y[it.sku] = pulp.LpVariable(f"order_{it.sku}", cat="Binary")
        gap_d[it.sku] = pulp.LpVariable(f"gapD_{it.sku}", lowBound=0)
        gap_t[it.sku] = pulp.LpVariable(f"gapT_{it.sku}", lowBound=0)
        q = it.case_pack * cases[it.sku]
        qty[it.sku] = q
        prob += q >= it.moq * y[it.sku], f"moq_{it.sku}"
        prob += q <= max(max_cases * it.case_pack, 0) * y[it.sku], f"link_{it.sku}"
        prob += gap_d[it.sku] >= it.demand_cover - position - q, f"demand_{it.sku}"
        prob += gap_t[it.sku] >= it.demand_cover + it.safety_stock - position - q, f"target_{it.sku}"
        prob += it.unit_cost * q <= max_sku_share * budget, f"share_{it.sku}"

    spend = pulp.lpSum(it.unit_cost * qty[it.sku] for it in items)
    prob += spend <= budget, "budget"
    prob += spend + pulp.lpSum(it.penalty_per_unit * (gap_d[it.sku] + 0.5 * gap_t[it.sku]) for it in items)

    solver = _solver()
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal",):
        return {"status": status, "solver": solver.name, "lines": [], "totals": {"budget": budget}}

    lines = []
    for it in items:
        n = int(round(cases[it.sku].value() or 0))
        q = n * it.case_pack
        position = it.on_hand + it.on_order
        target = it.demand_cover + it.safety_stock
        shortfall = max(0, int(round(target - position - q)))
        room = max(0.0, it.shelf_capacity + it.demand_lead - position)
        reasons = []
        if shortfall > 0:  # explain why the solver stopped short of target
            if q + it.case_pack > room:
                reasons.append("shelf capacity")
            elif it.unit_cost * (q + it.case_pack) > max_sku_share * budget + 1e-6:
                reasons.append("concentration limit")
            else:
                reasons.append("budget")
        lines.append({
            "sku": it.sku, "name": it.name, "category": it.category,
            "position": position, "demand_cover": round(it.demand_cover, 1),
            "safety_stock": round(it.safety_stock, 1), "target": round(target, 1),
            "cases": n, "case_pack": it.case_pack, "moq": it.moq, "order_qty": q,
            "unit_cost": it.unit_cost, "order_cost": round(q * it.unit_cost, 2),
            "shortfall_vs_target": shortfall,
            "binding": reasons,
        })
    total = round(sum(l["order_cost"] for l in lines), 2)
    return {
        "status": status,
        "solver": solver.name,
        "objective": round(pulp.value(prob.objective), 2),
        "lines": lines,
        "totals": {
            "total_cost": total,
            "budget": round(budget, 2),
            "budget_utilization_pct": round(100 * total / budget, 1),
            "skus_ordered": sum(1 for l in lines if l["order_qty"] > 0),
            "units_ordered": sum(l["order_qty"] for l in lines),
            "max_sku_share": max_sku_share,
        },
        "inputs": [asdict(i) for i in items],
    }
