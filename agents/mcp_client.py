"""Thin wrapper around langchain-mcp-adapters' MultiServerMCPClient.

Agents discover tools from the MCP server at runtime (no hard-coded HTTP calls)
and call them through LangChain tool objects, so the same tools could also be
handed to an LLM tool-calling agent.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

import config


def make_client(url: str | None = None, token: str | None = None) -> MultiServerMCPClient:
    url = url or config.MCP_SERVER_URL
    token = config.MCP_AUTH_TOKEN if token is None else token
    conn: dict[str, Any] = {"transport": "streamable_http", "url": url}
    if token:
        conn["headers"] = {"Authorization": f"Bearer {token}"}
    # handle_tool_errors=False -> tool errors raise instead of being returned as text,
    # so deterministic agents fail loudly rather than parse an error string as data.
    return MultiServerMCPClient({"inventory": conn}, handle_tool_errors=False)


class MCPTools:
    """Name -> tool lookup with JSON decoding of tool results."""

    def __init__(self, tools: list[BaseTool]):
        self.tools = {t.name: t for t in tools}
        self.calls: list[str] = []  # audit log of tool calls made in this session

    @property
    def names(self) -> list[str]:
        return sorted(self.tools)

    async def call(self, name: str, **args: Any) -> Any:
        if name not in self.tools:
            raise KeyError(f"MCP tool '{name}' not exposed by server (have: {self.names})")
        self.calls.append(f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items() if k != 'lines')})")
        result = await self.tools[name].ainvoke(args)
        return _decode(result)


def _decode(result: Any) -> Any:
    if isinstance(result, tuple):  # (content, artifact) response format
        result = result[0]
    if isinstance(result, list):
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in result)
    else:
        text = str(result)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MCP tool returned non-JSON content: {text[:200]}") from exc


@asynccontextmanager
async def mcp_session(client: MultiServerMCPClient | None = None) -> AsyncIterator[MCPTools]:
    """One MCP session per agent step (cheap; the server is stateless)."""
    client = client or make_client()
    async with client.session("inventory") as session:
        yield MCPTools(
            await load_mcp_tools(session, server_name="inventory", handle_tool_errors=client.handle_tool_errors)
        )
