"""Chainlit UI logic without a browser: helpers + the approve/reject/revise loop against a live MCP server."""
import asyncio
import uuid

from agents.llm import RuleBasedLLM
from agents.mcp_client import make_client
from ui import chainlit_app as ui

REQUEST = "Plan next week's replenishment for store S001 with a $5,000 budget"


def test_helpers():
    assert ui.parse_budget("$4,000") == 4000 and ui.parse_budget("3.5k") == 3500 and ui.parse_budget("no") is None
    assert ui.resume_value("revise", 4000)["budget"] == 4000
    assert ui.resume_value("revise", None)["decision"] == "reject"
    assert ui.resume_value("approve")["decision"] == "approve"
    assert ui.health_url("http://localhost:8123/mcp") == "http://localhost:8123/health"
    trace = ["inventory_agent: MCP as role=planner scopes=inventory:read; tools visible ['get_stock']",
             "place_order: MCP create_purchase_order as role=approver scopes=inventory:read po:write -> PO-1 ($1.00)"]
    assert ui.credentials_used(trace) == {"inventory_agent": "role=planner scopes=inventory:read",
                                          "place_order": "role=approver scopes=inventory:read po:write"}


def _drive(graph, decisions, budget="4000"):
    nodes, asked = [], []

    async def on_node(node, lines):
        nodes.append(node)

    async def ask_decision(payload):
        asked.append(ui.plan_markdown(payload))
        return decisions.pop(0)

    async def ask_budget():
        return budget

    async def notify(text):
        pass

    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}
    return asyncio.run(ui.drive(graph, REQUEST, cfg, on_node, ask_decision, ask_budget, notify)), nodes, asked


def test_ui_revise_then_approve_with_jwt_roles(mcp_url_jwt, jwt_tokens):
    clients = {r: make_client(mcp_url_jwt, token=jwt_tokens[r], role=r) for r in ("planner", "approver")}
    graph, _ = ui.make_graph(offline=True, mcp_clients=clients)
    values, nodes, asked = _drive(graph, ["revise", "approve"])
    assert len(asked) == 2 and "| SKU |" in asked[0] and "$5,000.00 budget" in asked[0] and "$4,000.00 budget" in asked[1]
    for n in ("supervisor", "inventory_agent", "demand_agent", "policy_agent", "optimizer_agent", "place_order", "report"):
        assert n in nodes
    md = ui.final_markdown(values)
    assert values["purchase_order"]["po_id"] in md
    assert "place_order → `role=approver scopes=inventory:read po:write`" in md
    assert "inventory_agent → `role=planner scopes=inventory:read`" in md


def test_ui_reject_places_nothing(mcp_url):
    graph, _ = ui.make_graph(offline=True, mcp_client=make_client(mcp_url, token=""))
    values, nodes, _ = _drive(graph, ["reject"])
    assert "place_order" not in nodes and not values.get("purchase_order")
    assert "none placed" in ui.final_markdown(values)


def test_ensure_mcp_server_detects_running(mcp_url):
    assert "already running" in ui.ensure_mcp_server(mcp_url)
