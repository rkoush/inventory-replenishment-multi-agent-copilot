"""Role-based access control for the inventory MCP server.

Roles map to OAuth-style scopes, and each tool requires one scope:

    planner   inventory:read            read stock, sales, and supplier terms
    approver  inventory:read po:write   also create purchase orders (after human approval)
    auditor   po:read                   read purchase orders only

The MCP server acts as an OAuth *resource server*: it only validates tokens
(signature, issuer, audience, expiry) and checks scopes. It never issues them.
Locally, `scripts/mint_tokens.py` plays the identity provider (IdP).
"""
from __future__ import annotations

from typing import Any

from fastmcp.exceptions import AuthorizationError
from fastmcp.server.auth import AuthContext

import config

ROLES: dict[str, list[str]] = {
    "planner": ["inventory:read"],
    "approver": ["inventory:read", "po:write"],
    "auditor": ["po:read"],
}
ALL_SCOPES = sorted({s for scopes in ROLES.values() for s in scopes})

TOOL_SCOPES: dict[str, str] = {
    "get_stock": "inventory:read",
    "list_store_inventory": "inventory:read",
    "list_low_stock": "inventory:read",
    "get_sales_history": "inventory:read",
    "get_supplier_terms": "inventory:read",
    "create_purchase_order": "po:write",
    "list_purchase_orders": "po:read",
}


def scope_check(ctx: AuthContext) -> bool:
    """AuthMiddleware check: the caller's token must carry the tool's scope (deny by default).

    Raising AuthorizationError (rather than returning False) gives a clear
    'forbidden: missing scope ...' error on tools/call. tools/list hides the tool.
    """
    name = ctx.component.name
    need = TOOL_SCOPES.get(name)
    if need is None:
        raise AuthorizationError(f"forbidden: no scope policy for '{name}'")
    if ctx.token is None:
        raise AuthorizationError("forbidden: not authenticated")
    if need not in ctx.token.scopes:
        who = (ctx.token.claims or {}).get("role") or ctx.token.client_id
        raise AuthorizationError(f"forbidden: missing scope {need} for tool '{name}' "
                                 f"(caller={who}, scopes={' '.join(ctx.token.scopes) or '-'})")
    return True


def build_auth() -> tuple[Any, list[Any]]:
    """(auth provider, middleware) for config.MCP_AUTH = none | static | jwt."""
    mode = config.MCP_AUTH
    if mode == "none":
        return None, []
    from fastmcp.server.middleware import AuthMiddleware

    if mode == "static":  # legacy single shared token -> all scopes (demo only)
        from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

        if not config.MCP_AUTH_TOKEN:
            raise SystemExit("MCP_AUTH=static needs MCP_AUTH_TOKEN")
        verifier = StaticTokenVerifier(tokens={config.MCP_AUTH_TOKEN: {
            "client_id": "replenishiq-agents", "scopes": ALL_SCOPES}})
    elif mode == "jwt":
        from fastmcp.server.auth.providers.jwt import JWTVerifier

        if not config.MCP_JWT_PUBLIC_KEY_FILE.exists():
            raise SystemExit(f"MCP_AUTH=jwt: no public key at {config.MCP_JWT_PUBLIC_KEY_FILE}. "
                             "Run: python -m scripts.mint_tokens")
        verifier = JWTVerifier(
            public_key=config.MCP_JWT_PUBLIC_KEY_FILE.read_text(),
            issuer=config.MCP_JWT_ISSUER,
            audience=config.MCP_JWT_AUDIENCE,
            algorithm="RS256",
        )
    else:
        raise SystemExit(f"Unknown MCP_AUTH={mode!r} (use none | static | jwt)")
    return verifier, [AuthMiddleware(auth=scope_check)]
