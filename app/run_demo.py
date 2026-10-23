"""ReplenishIQ end-to-end demo (CLI).

    python -m app.run_demo                         # Groq LLM (needs GROQ_API_KEY), interactive approval
    python -m app.run_demo --offline               # no API key: rule-based stand-in for the LLM steps
    python -m app.run_demo --decision approve      # non-interactive (also: reject, revise:4000)

Requires the MCP server to be running:  python -m mcp_servers.inventory_server
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langgraph.types import Command  # noqa: E402

import config  # noqa: E402
from agents.graph import build_graph  # noqa: E402
from agents.llm import get_llm  # noqa: E402

DEFAULT_REQUEST = "Plan next week's replenishment for store S001 with a $5,000 budget"


def print_interrupt(payload: dict[str, Any]) -> None:
    t = payload["totals"]
    print("\n" + "=" * 78)
    print(f"APPROVAL NEEDED - store {payload['store']}: ${t['total_cost']:,.2f} of ${t['budget']:,.2f} "
          f"({t['budget_utilization_pct']}%), {t['skus_ordered']} SKUs, {t['units_ordered']} units")
    print(f"{'SKU':<9}{'Item':<26}{'Qty':>6}{'Cost':>11}{'Short':>7}")
    for l in payload["lines"]:
        print(f"{l['sku']:<9}{l['name'][:25]:<26}{l['order_qty']:>6}{l['order_cost']:>11,.2f}{l['shortfall_vs_target']:>7}")
    for w in payload.get("warnings", []):
        print(f"  ! {w}")
    print("=" * 78)


def parse_decision(text: str) -> Any:
    text = text.strip()
    if text.lower().startswith("revise"):
        _, _, amount = text.partition(":")
        return {"decision": "revise", "budget": float(amount.replace("$", "").replace(",", "") or 0), "approver": "cli-user"}
    return {"decision": "approve" if text.lower() in {"a", "y", "yes", "approve"} else "reject", "approver": "cli-user"}


async def stream(graph, inp: Any, cfg: dict) -> dict | None:
    """Run until END or an interrupt; print trace lines as nodes finish. Returns interrupt payload."""
    async for update in graph.astream(inp, cfg, stream_mode="updates"):
        for node, delta in update.items():
            if node == "__interrupt__":
                return delta[0].value
            for line in (delta or {}).get("trace", []):
                print(f"  [{node}] {line}")
    return None


async def run(request: str, offline: bool, decision: str | None) -> dict:
    llm = get_llm(offline=offline)
    router = get_llm(offline=offline, model=config.GROQ_ROUTER_MODEL)
    graph = build_graph(llm=llm, router_llm=router)
    cfg = {"configurable": {"thread_id": f"demo-{uuid.uuid4().hex[:8]}"}}
    print(f"Request: {request}\nLLM: {getattr(llm, 'name', llm)} | MCP: {config.MCP_SERVER_URL}\n")

    scripted = [d.strip() for d in decision.split(",") if d.strip()] if decision else []
    payload = await stream(graph, {"request": request}, cfg)
    while payload is not None:
        print_interrupt(payload)
        if scripted:
            answer = scripted.pop(0)
            print(f"Decision (scripted): {answer}")
        elif decision:  # scripted decisions used up (e.g. after a revise): stop safely
            answer = "reject"
            print("Decision (scripted): no more decisions -> reject")
        else:
            answer = input("Decision [approve / reject / revise:<budget>]: ")
        payload = await stream(graph, Command(resume=parse_decision(answer)), cfg)

    final = (await graph.aget_state(cfg)).values
    print("\n" + final.get("report", ""))
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", default=DEFAULT_REQUEST)
    ap.add_argument("--offline", action="store_true", help="use the rule-based stand-in instead of Groq")
    ap.add_argument("--decision", help="non-interactive decisions, comma-separated, e.g. approve | reject | revise:4000,approve")
    args = ap.parse_args()
    asyncio.run(run(args.request, args.offline, args.decision))


if __name__ == "__main__":
    main()
