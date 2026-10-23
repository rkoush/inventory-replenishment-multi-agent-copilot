"""JWT role-based access: planner / approver / auditor scopes enforced per MCP tool."""
import asyncio
import traceback
import uuid

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from langgraph.types import Command

from agents.graph import build_graph
from agents.llm import RuleBasedLLM
from agents.mcp_client import make_client, mcp_session
from scripts.mint_tokens import RSAKeyPair

READ_TOOLS = {"get_stock", "list_store_inventory", "list_low_stock", "get_sales_history", "get_supplier_terms"}
PO_LINE = [{"sku": "GRO-001", "quantity": 48}]


def run(coro):
    return asyncio.run(coro)


async def _visible(url, token):
    async with mcp_session(make_client(url, token=token)) as tools:
        return set(tools.names)


async def _call(url, token, name, **args):
    async with Client(url, auth=token) as c:  # raw fastmcp client: see the server's own error text
        return (await c.call_tool(name, args)).data


def test_tool_listing_is_filtered_by_role(mcp_url_jwt, jwt_tokens):
    assert run(_visible(mcp_url_jwt, jwt_tokens["planner"])) == READ_TOOLS
    assert run(_visible(mcp_url_jwt, jwt_tokens["approver"])) == READ_TOOLS | {"create_purchase_order"}
    assert run(_visible(mcp_url_jwt, jwt_tokens["auditor"])) == {"list_purchase_orders"}


def test_planner_cannot_create_po(mcp_url_jwt, jwt_tokens):
    with pytest.raises(ToolError, match="forbidden: missing scope po:write"):
        run(_call(mcp_url_jwt, jwt_tokens["planner"], "create_purchase_order",
                  store="S001", lines=PO_LINE, approved_by="x"))


def test_auditor_cannot_read_stock(mcp_url_jwt, jwt_tokens):
    with pytest.raises(ToolError, match="forbidden: missing scope inventory:read"):
        run(_call(mcp_url_jwt, jwt_tokens["auditor"], "get_stock", sku="GRO-001", store="S001"))


def test_approver_creates_and_auditor_reads_po(mcp_url_jwt, jwt_tokens):
    po = run(_call(mcp_url_jwt, jwt_tokens["approver"], "create_purchase_order",
                   store="S001", lines=PO_LINE, approved_by="rishi"))
    assert po["status"] == "submitted"
    orders = run(_call(mcp_url_jwt, jwt_tokens["auditor"], "list_purchase_orders", store="S001"))["orders"]
    assert po["po_id"] in {o["po_id"] for o in orders}


def test_missing_bad_or_foreign_token_is_rejected(mcp_url_jwt):
    r = httpx.post(mcp_url_jwt, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                   headers={"Accept": "application/json, text/event-stream"})
    assert r.status_code == 401
    other_idp = RSAKeyPair.generate().create_token(issuer="replenishiq-local-idp", audience="replenishiq-mcp",
                                                   scopes=["inventory:read", "po:write"])
    for bad in ("not-a-jwt", other_idp):  # wrong signing key -> rejected even with the right scopes
        with pytest.raises(BaseException):
            run(_visible(mcp_url_jwt, bad))


def test_graph_uses_planner_for_reads_and_approver_for_po(mcp_url_jwt, jwt_tokens):
    graph = build_graph(llm=RuleBasedLLM(), mcp_clients={
        "planner": make_client(mcp_url_jwt, token=jwt_tokens["planner"], role="planner"),
        "approver": make_client(mcp_url_jwt, token=jwt_tokens["approver"], role="approver")})
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}

    async def go():
        await graph.ainvoke({"request": "Plan next week's replenishment for store S001 with a $5,000 budget"}, cfg)
        await graph.ainvoke(Command(resume="approve"), cfg)
        return (await graph.aget_state(cfg)).values

    state = run(go())
    trace = "\n".join(state["trace"])
    assert "inventory_agent: MCP as role=planner scopes=inventory:read" in trace
    assert "create_purchase_order" not in trace.split("place_order")[0]  # planner never saw the write tool
    assert "place_order: MCP create_purchase_order as role=approver scopes=inventory:read po:write" in trace
    assert state["purchase_order"]["po_id"].startswith("PO-")


def test_planner_token_cannot_place_order_from_graph(mcp_url_jwt, jwt_tokens):
    planner = make_client(mcp_url_jwt, token=jwt_tokens["planner"], role="planner")
    graph = build_graph(llm=RuleBasedLLM(), mcp_client=planner)  # misconfigured: planner for everything
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}}

    async def go():
        await graph.ainvoke({"request": "Plan next week's replenishment for store S001 with a $5,000 budget"}, cfg)
        await graph.ainvoke(Command(resume="approve"), cfg)

    with pytest.raises(BaseException) as ei:  # PermissionError, wrapped in anyio's ExceptionGroup
        run(go())
    assert "'create_purchase_order' is not available to this caller (role=planner" in "".join(
        traceback.format_exception(ei.value))
