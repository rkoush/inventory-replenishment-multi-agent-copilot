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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0"))
# Optional cheaper/faster model for the supervisor's routing step (e.g. llama-3.1-8b-instant)
GROQ_ROUTER_MODEL = os.getenv("GROQ_ROUTER_MODEL", "") or GROQ_MODEL
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "768"))  # cap output tokens (free tier friendly)
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "4"))  # SDK retries 429/5xx with exponential backoff

# MCP inventory server (Streamable HTTP)
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://localhost:{MCP_PORT}/mcp")
MCP_AUTH_TOKEN = os.getenv("MCP_AUTH_TOKEN", "")  # empty = auth disabled

# Planning defaults (used when the request or the policy docs don't say otherwise)
DEFAULT_HORIZON_DAYS = 7
DEFAULT_SAFETY_STOCK_DAYS = 7
DEFAULT_MAX_SKU_BUDGET_SHARE = 0.30
SHORTAGE_PENALTY_MULTIPLIER = 3.0  # cost of 1 unit short = 3x unit cost (lost sale + goodwill)
MAX_SUPERVISOR_STEPS = 12  # hard stop against routing loops
