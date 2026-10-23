"""Local stand-in for an identity provider: an RS256 keypair plus one JWT per role.

    python -m scripts.mint_tokens               # writes .secrets/ (gitignored), prints a summary
    python -m scripts.mint_tokens --hours 2     # token lifetime (default 8h)
    python -m scripts.mint_tokens --print-env   # also print export lines (contains tokens!)

Writes:
  .secrets/jwt_private.pem   signing key (only this script uses it)
  .secrets/jwt_public.pem    verification key (the MCP server loads it when MCP_AUTH=jwt)
  .secrets/tokens.env        MCP_PLANNER_TOKEN / MCP_APPROVER_TOKEN / MCP_AUDITOR_TOKEN

In production you'd delete this script: tokens come from the enterprise IdP (OAuth 2.1),
and the MCP server only validates them.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastmcp.server.auth.providers.jwt import RSAKeyPair  # noqa: E402
from pydantic import SecretStr  # noqa: E402

import config  # noqa: E402
from mcp_servers.auth import ROLES  # noqa: E402


def _write_private(path: Path, text: str) -> None:
    path.write_text(text)
    os.chmod(path, 0o600)


def load_or_create_keypair(secrets: Path, new_key: bool = False) -> RSAKeyPair:
    priv, pub = secrets / "jwt_private.pem", secrets / "jwt_public.pem"
    if priv.exists() and pub.exists() and not new_key:
        return RSAKeyPair(private_key=SecretStr(priv.read_text()), public_key=pub.read_text())
    kp = RSAKeyPair.generate()
    _write_private(priv, kp.private_key.get_secret_value())
    pub.write_text(kp.public_key)
    return kp


def mint(secrets: Path | None = None, hours: float = 8, new_key: bool = False) -> dict[str, str]:
    """Create/reuse the keypair and write one token per role. Returns {role: token}."""
    secrets = secrets or config.SECRETS_DIR
    secrets.mkdir(parents=True, exist_ok=True)
    kp = load_or_create_keypair(secrets, new_key)
    tokens = {
        role: kp.create_token(
            subject=f"replenishiq-{role}",
            issuer=config.MCP_JWT_ISSUER,
            audience=config.MCP_JWT_AUDIENCE,
            scopes=scopes,
            expires_in_seconds=int(hours * 3600),
            additional_claims={"role": role, "client_id": f"replenishiq-{role}"},
        )
        for role, scopes in ROLES.items()
    }
    _write_private(secrets / "tokens.env",
                   "".join(f"MCP_{r.upper()}_TOKEN={t}\n" for r, t in tokens.items()))
    return tokens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8, help="token lifetime in hours (default 8)")
    ap.add_argument("--new-key", action="store_true", help="rotate the signing keypair (restart the server)")
    ap.add_argument("--print-env", action="store_true", help="print export lines (they contain the tokens)")
    args = ap.parse_args()
    tokens = mint(hours=args.hours, new_key=args.new_key)
    print(f"Wrote {config.SECRETS_DIR}/ (jwt_private.pem, jwt_public.pem, tokens.env); "
          f"iss={config.MCP_JWT_ISSUER} aud={config.MCP_JWT_AUDIENCE} ttl={args.hours:g}h")
    for role, scopes in ROLES.items():
        print(f"  {role:<9} scopes: {' '.join(scopes)}")
    if args.print_env:
        for role, tok in tokens.items():
            print(f"export MCP_{role.upper()}_TOKEN={tok}")


if __name__ == "__main__":
    main()
