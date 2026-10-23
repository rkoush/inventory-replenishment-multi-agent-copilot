"""Shared fixtures: a temp copy of the synthetic DB and an MCP server subprocess on it."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> Path:
    from scripts.generate_data import build_frames, write_db

    return write_db(build_frames(), tmp_path_factory.mktemp("data") / "test.db")


def _start_server(db_path: Path, token: str = "", **extra_env: str):
    port = _free_port()
    env = {**os.environ, "REPLENISHIQ_DB": str(db_path), "MCP_PORT": str(port), "MCP_AUTH_TOKEN": token,
           "MCP_AUTH": "", **extra_env}
    proc = subprocess.Popen([sys.executable, "-m", "mcp_servers.inventory_server"], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5).status_code == 200:
                return proc, f"http://127.0.0.1:{port}/mcp"
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("MCP server did not start")


@pytest.fixture(scope="session")
def mcp_url(db_path):
    proc, url = _start_server(db_path)
    yield url
    proc.terminate()


@pytest.fixture(scope="session")
def mcp_url_auth(db_path):
    proc, url = _start_server(db_path, token="test-secret")
    yield url
    proc.terminate()


@pytest.fixture(scope="session")
def jwt_tokens(tmp_path_factory) -> dict[str, str]:
    """RS256 keypair + planner/approver/auditor tokens in a temp secrets dir."""
    from scripts.mint_tokens import mint

    secrets = tmp_path_factory.mktemp("secrets")
    tokens = mint(secrets=secrets, hours=1)
    tokens["_secrets_dir"] = str(secrets)
    return tokens


@pytest.fixture(scope="session")
def mcp_url_jwt(db_path, jwt_tokens):
    proc, url = _start_server(db_path, MCP_AUTH="jwt", REPLENISHIQ_SECRETS=jwt_tokens["_secrets_dir"])
    yield url
    proc.terminate()


@pytest.fixture(scope="session", autouse=True)
def rag_index(tmp_path_factory):
    """Build the policy index in a temp dir so tests don't depend on local state."""
    import config

    config.CHROMA_DIR = tmp_path_factory.mktemp("chroma")
    from rag.build_index import build_index

    build_index()
    yield
