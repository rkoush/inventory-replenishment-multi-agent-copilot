"""Central configuration for ReplenishIQ.

All settings come from environment variables (optionally loaded from a local
`.env` file) so the same code runs locally, in Colab, and in containers.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"

DB_PATH = Path(os.getenv("REPLENISHIQ_DB", DATA_DIR / "replenishiq.db"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
CHROMA_COLLECTION = "replenishment_policies"

# LLM (Groq)
GROQ_MODEL = os.getenv("GROQ_MODEL", "") or "openai/gpt-oss-120b"
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0"))
# Cheaper/faster model for the supervisor's routing step
GROQ_ROUTER_MODEL = os.getenv("GROQ_ROUTER_MODEL", "") or "openai/gpt-oss-20b"
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "") or "1024")  # cap output tokens incl. reasoning tokens
# Reasoning models (gpt-oss, qwen3): keep thinking short so it doesn't eat the output budget
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "") or "low"
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "4"))  # SDK retries 429/5xx with exponential backoff

# MCP inventory server (Streamable HTTP)
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://localhost:{MCP_PORT}/mcp")
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "")  # legacy single static token (all scopes)
# Auth mode: none | static (MCP_AUTH_TOKEN) | jwt (RS256 role tokens, see scripts/mint_tokens.py)
MCP_AUTH = (os.getenv("MCP_AUTH", "") or ("static" if MCP_AUTH_TOKEN else "none")).lower()
SECRETS_DIR = Path(os.getenv("REPLENISHIQ_SECRETS", ROOT / ".secrets"))  # gitignored
MCP_JWT_ISSUER = os.getenv("MCP_JWT_ISSUER", "replenishiq-local-idp")
MCP_JWT_AUDIENCE = os.getenv("MCP_JWT_AUDIENCE", "replenishiq-mcp")
MCP_JWT_PUBLIC_KEY_FILE = Path(os.getenv("MCP_JWT_PUBLIC_KEY_FILE", "") or SECRETS_DIR / "jwt_public.pem")


def role_token(role: str) -> str:
    """JWT for a role: env MCP_<ROLE>_TOKEN, else SECRETS_DIR/tokens.env (written by mint_tokens.py)."""
    key = f"MCP_{role.upper()}_TOKEN"
    if os.getenv(key):
        return os.environ[key]
    path = SECRETS_DIR / "tokens.env"
    if path.exists():
        for line in path.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    return ""

# Planning defaults (used when the request or the policy docs don't say otherwise)
DEFAULT_HORIZON_DAYS = 7
DEFAULT_SAFETY_STOCK_DAYS = 7
DEFAULT_MAX_SKU_BUDGET_SHARE = 0.30
SHORTAGE_PENALTY_MULTIPLIER = 3.0  # cost of 1 unit short = 3x unit cost (lost sale + goodwill)
MAX_SUPERVISOR_STEPS = 12  # hard stop against routing loops
