"""Thin wrapper around langchain-mcp-adapters' MultiServerMCPClient.

Agents discover tools from the MCP server at runtime (no hard-coded HTTP calls)
and call them through LangChain tool objects, so the same tools could also be
handed to an LLM tool-calling agent.
"""
from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

import config


def _jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT payload WITHOUT verifying it (only for trace labels; the server verifies)."""
    try:
        payload = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception:
        return {}


def auth_label(token: str, role: str | None = None) -> str:
    """Short description of the credential, for trace lines, e.g. 'role=planner scopes=inventory:read'."""
    if not token:
        return "no auth"
    claims = _jwt_claims(token)
    if claims:
        return f"role={claims.get('role', role or '?')} scopes={claims.get('scope', '-')}"
    return "static token"


def make_client(url: str | None = None, token: str | None = None, role: str | None = None) -> MultiServerMCPClient:
    """MCP client for the inventory server.

    token=None picks the credential from config: with MCP_AUTH=jwt, the JWT for `role`
    (planner/approver/auditor); with MCP_AUTH=static, MCP_AUTH_TOKEN; otherwise none.
    The returned client carries `.role` and `.auth_label` for traces.
    """
    url = url or config.MCP_SERVER_URL
    if token is None:
        if config.MCP_AUTH == "jwt" and role:
            token = config.role_token(role)
            if not token:
                raise RuntimeError(f"MCP_AUTH=jwt but no token for role '{role}'. Run: python -m scripts.mint_tokens")
        else:
            token = config.MCP_AUTH_TOKEN
    conn: dict[str, Any] = {"transport": "streamable_http", "url": url}
    if token:
        conn["headers"] = {"Authorization": f"Bearer {token}"}
    # handle_tool_errors=False -> tool errors raise instead of being returned as text,
    # so deterministic agents fail loudly rather than parse an error string as data.
    client = MultiServerMCPClient({"inventory": conn}, handle_tool_errors=False)
    client.role = role or (_jwt_claims(token).get("role") if token else None)
    client.auth_label = auth_label(token, role)
    return client


def clients_for_roles(mcp_client: Any = None, mcp_clients: dict[str, Any] | None = None) -> dict[str, Any]:
    """{'planner': client, 'approver': client}: explicit per-role clients, one shared client, or None (lazy)."""
    out = {"planner": mcp_client, "approver": mcp_client}
    out.update(mcp_clients or {})
    return out


class MCPTools:
    """Name -> tool lookup with JSON decoding of tool results."""

    def __init__(self, tools: list[BaseTool], auth: str = "no auth"):
        self.tools = {t.name: t for t in tools}
        self.auth = auth  # e.g. "role=planner scopes=inventory:read"
        self.calls: list[str] = []  # audit log of tool calls made in this session

    @property
    def names(self) -> list[str]:
        return sorted(self.tools)

    async def call(self, name: str, **args: Any) -> Any:
        if name not in self.tools:
            raise PermissionError(f"MCP tool '{name}' is not available to this caller ({self.auth}); "
                                  f"visible tools: {self.names}")
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
async def mcp_session(client: MultiServerMCPClient | None = None, role: str | None = None) -> AsyncIterator[MCPTools]:
    """One MCP session per agent step (cheap; the server is stateless).

    The tool list is what the server shows this caller: tools it lacks scopes for are hidden.
    """
    client = client or make_client(role=role)
    async with client.session("inventory") as session:
        yield MCPTools(
            await load_mcp_tools(session, server_name="inventory", handle_tool_errors=client.handle_tool_errors),
            auth=getattr(client, "auth_label", "no auth"),
        )
