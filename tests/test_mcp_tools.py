import asyncio

import pytest

from agents.mcp_client import make_client, mcp_session


def run(coro):
    return asyncio.run(coro)


def test_tools_discovered_and_callable(mcp_url):
    async def go():
        async with mcp_session(make_client(mcp_url, token="")) as tools:
            assert {"get_stock", "list_low_stock", "get_sales_history", "get_supplier_terms"} <= set(tools.names)
            stock = await tools.call("get_stock", sku="GRO-001", store="S001")
            low = await tools.call("list_low_stock", store="S001")
            hist = await tools.call("get_sales_history", sku="GRO-001", store="S001", days=30)
            terms = await tools.call("get_supplier_terms", sku="GRO-001")
            return stock, low, hist, terms

    stock, low, hist, terms = run(go())
    assert stock["on_hand"] >= 0 and stock["unit_cost"] > 0
    assert any(i["sku"] == "GRO-003" for i in low["items"])
    assert hist["days"] == 30
    assert terms["case_pack"] == 4 and terms["moq"] == 24


def test_invalid_input_raises(mcp_url):
    async def go():
        async with mcp_session(make_client(mcp_url, token="")) as tools:
            await tools.call("get_stock", sku="DROP TABLE", store="S001")

    with pytest.raises(Exception):
        run(go())


def test_purchase_order_requires_valid_case_pack(mcp_url):
    async def go(qty):
        async with mcp_session(make_client(mcp_url, token="")) as tools:
            return await tools.call("create_purchase_order", store="S001",
                                    lines=[{"sku": "GRO-001", "quantity": qty}], approved_by="tester")

    assert run(go(48))["status"] == "submitted"
    with pytest.raises(Exception):
        run(go(25))  # not a multiple of the case pack


def test_bearer_token_auth(mcp_url_auth):
    async def go(token):
        async with mcp_session(make_client(mcp_url_auth, token=token)) as tools:
            return tools.names

    assert "get_stock" in run(go("test-secret"))
    with pytest.raises(BaseException):
        run(go("wrong-token"))
