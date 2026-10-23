"""ReplenishIQ chat UI (Chainlit).

    chainlit run ui/chainlit_app.py --port 8501        # http://localhost:8501 (8000 is the MCP server)
    chainlit run ui/chainlit_app.py --port 8501 -h     # -h/--headless: do not open a browser

Each graph node shows up as a collapsible step with its trace. At the approval interrupt you get
Approve / Reject / Revise buttons; Revise asks for a new budget. The buttons map to the same
Command(resume=...) payloads as the CLI. If no MCP server answers at MCP_SERVER_URL, the app starts
one (python -m mcp_servers.inventory_server), building the demo data and index first if needed.

Env: GROQ_API_KEY (unset -> offline rule-based stand-ins), RIQ_OFFLINE=1 (force offline),
MCP_AUTH=jwt (role tokens from `python -m scripts.mint_tokens`), CHAINLIT_TRANSPORTS=polling
(force HTTP long-polling when a proxy breaks WebSockets, e.g. some Colab setups).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from langgraph.types import Command  # noqa: E402

import config  # noqa: E402
from agents.graph import build_graph  # noqa: E402
from agents.llm import get_llm  # noqa: E402

DEFAULT_REQUEST = "Plan next week's replenishment for store S001 with a $5,000 budget"
STEP_LABELS = {"supervisor": "Supervisor", "inventory_agent": "Inventory agent (MCP)",
               "demand_agent": "Demand agent (forecast)", "policy_agent": "Policy agent (RAG)",
               "optimizer_agent": "Optimizer (MILP)", "human_approval": "Human approval",
               "place_order": "Place order (MCP)", "report": "Report"}
_AUTH_RE = re.compile(r"\bas (role=\S+ scopes=[^;]+?|static token|no auth)(?=;| ->|$)")


# ---------------------------------------------------------------- plain helpers (unit-tested)
def health_url(mcp_url: str | None = None) -> str:
    u = urlparse(mcp_url or config.MCP_SERVER_URL)
    return f"{u.scheme}://{u.netloc}/health"


def mcp_healthy(mcp_url: str | None = None) -> dict | None:
    try:
        r = httpx.get(health_url(mcp_url), timeout=1)
        return r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None


def ensure_mcp_server(mcp_url: str | None = None, wait_s: float = 30) -> str:
    """Start the MCP server in the background unless one already answers /health."""
    if (h := mcp_healthy(mcp_url)) is not None:
        return f"MCP server already running at {mcp_url or config.MCP_SERVER_URL} (auth={h.get('auth', '?')})"
    if not config.DB_PATH.exists():
        subprocess.run([sys.executable, "-m", "scripts.generate_data"], cwd=ROOT, check=True)
    if not config.CHROMA_DIR.exists():
        subprocess.run([sys.executable, "-m", "rag.build_index"], cwd=ROOT, check=True)
    port = urlparse(mcp_url or config.MCP_SERVER_URL).port or config.MCP_PORT
    log = open(ROOT / "mcp_server.log", "a")
    subprocess.Popen([sys.executable, "-m", "mcp_servers.inventory_server"], cwd=ROOT,
                     env={**os.environ, "MCP_PORT": str(port)}, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if (h := mcp_healthy(mcp_url)) is not None:
            return f"Started MCP server on port {port} (auth={h.get('auth', '?')}); log: mcp_server.log"
        time.sleep(0.5)
    raise RuntimeError(f"MCP server did not start within {wait_s:.0f}s; see {ROOT / 'mcp_server.log'}")


def plan_markdown(payload: dict[str, Any]) -> str:
    t = payload["totals"]
    out = [f"### Approval needed: store {payload['store']}",
           f"**${t['total_cost']:,.2f}** of ${t['budget']:,.2f} budget ({t['budget_utilization_pct']}%), "
           f"{t['skus_ordered']} SKUs, {t['units_ordered']} units", "",
           "| SKU | Item | Order qty | Cost | Short vs target |", "|---|---|---:|---:|---:|"]
    out += [f"| {l['sku']} | {l['name']} | {l['order_qty']} | ${l['order_cost']:,.2f} | {l['shortfall_vs_target']} |"
            for l in payload["lines"]]
    out += [f"\n> ⚠️ {w}" for w in payload.get("warnings", [])]
    return "\n".join(out)


def parse_budget(text: str | None) -> float | None:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", (text or "").replace("$", ""))
    if not m:
        return None
    value = float(m.group().replace(",", ""))
    if (text or "").strip().lower().endswith("k"):
        value *= 1000
    return value if value > 0 else None


def resume_value(action: str, budget: float | None = None, approver: str = "chainlit-user") -> dict[str, Any]:
    """Map a button to the human_approval resume payload."""
    if action == "revise" and budget:
        return {"decision": "revise", "budget": budget, "approver": approver}
    if action == "approve":
        return {"decision": "approve", "approver": approver, "comment": "approved in Chainlit UI"}
    return {"decision": "reject", "approver": approver}


def credentials_used(trace: list[str]) -> dict[str, str]:
    """{'inventory_agent': 'role=planner scopes=inventory:read', 'place_order': 'role=approver ...'}"""
    out: dict[str, str] = {}
    for line in trace:
        node = line.split(":", 1)[0].strip()
        if node in {"inventory_agent", "demand_agent", "place_order"} and (m := _AUTH_RE.search(line)):
            out[node] = m.group(1)
    return out


def final_markdown(values: dict[str, Any]) -> str:
    po = values.get("purchase_order") or {}
    creds = credentials_used(values.get("trace", []))
    head = [f"**Purchase order:** `{po['po_id']}` ({po.get('status', 'submitted')}, ${po['total_cost']:,.2f})"
            if po else "**Purchase order:** none placed"]
    if creds:
        head.append("**MCP credentials used:** " + "; ".join(f"{n} → `{c}`" for n, c in creds.items()))
    return "\n\n".join(head + ["---", values.get("report", "")])


NodeCallback = Callable[[str, list[str]], Awaitable[None]]


async def run_until_pause(graph, inp: Any, cfg: dict, on_node: NodeCallback) -> dict | None:
    """Stream the graph; call on_node(node, trace_lines) per finished node. Returns the interrupt payload or None."""
    async for update in graph.astream(inp, cfg, stream_mode="updates"):
        for node, delta in update.items():
            if node == "__interrupt__":
                return delta[0].value
            await on_node(node, list((delta or {}).get("trace", [])))
    return None


async def drive(graph, request: str, cfg: dict, on_node: NodeCallback,
                ask_decision: Callable[[dict], Awaitable[str]], ask_budget: Callable[[], Awaitable[str | None]],
                notify: Callable[[str], Awaitable[None]]) -> dict[str, Any]:
    """The UI's conversation loop: run, then at each interrupt ask approve/reject/revise (and a budget), resume."""
    payload = await run_until_pause(graph, {"request": request.strip() or DEFAULT_REQUEST}, cfg, on_node)
    while payload is not None:
        action, budget = await ask_decision(payload), None
        if action == "revise":
            budget = parse_budget(await ask_budget())
            if budget is None:
                await notify("Couldn't read a budget; rejecting the plan instead.")
                action = "reject"
        payload = await run_until_pause(graph, Command(resume=resume_value(action, budget)), cfg, on_node)
    return (await graph.aget_state(cfg)).values


def make_graph(offline: bool | None = None, **graph_kwargs):
    offline = offline if offline is not None else (os.getenv("RIQ_OFFLINE") == "1" or not os.getenv("GROQ_API_KEY"))
    llm = get_llm(offline=offline)
    graph = build_graph(llm=llm, router_llm=get_llm(offline=offline, model=config.GROQ_ROUTER_MODEL), **graph_kwargs)
    return graph, getattr(llm, "name", type(llm).__name__)


# ---------------------------------------------------------------- Chainlit handlers
try:
    import chainlit as cl
    from chainlit.config import config as cl_config
except ImportError:  # helpers above stay importable (tests, notebooks) without chainlit
    cl = None

if cl is not None:
    if os.getenv("CHAINLIT_TRANSPORTS"):
        cl_config.project.transports = [t.strip() for t in os.environ["CHAINLIT_TRANSPORTS"].split(",") if t.strip()]

    async def _show_node(node: str, lines: list[str]) -> None:
        async with cl.Step(name=STEP_LABELS.get(node, node), type="tool", show_input=False) as step:
            step.output = "\n".join(f"- {l}" for l in lines) or "(no trace)"

    @cl.on_chat_start
    async def start() -> None:
        try:
            status = ensure_mcp_server()
        except Exception as e:  # noqa: BLE001 - show the problem in the chat
            status = f"⚠️ {e}"
        graph, llm_name = make_graph()
        cl.user_session.set("graph", graph)
        await cl.Message(content=(
            f"**ReplenishIQ** multi-agent replenishment copilot\n\n"
            f"- LLM: `{llm_name}`\n- MCP: `{config.MCP_SERVER_URL}` (auth mode `{config.MCP_AUTH}`)\n- {status}\n\n"
            f"Try: _{DEFAULT_REQUEST}_")).send()

    async def _ask_decision(payload: dict) -> str:
        res = await cl.AskActionMessage(content=plan_markdown(payload), timeout=3600, actions=[
            cl.Action(name="approve", payload={"action": "approve"}, label="✅ Approve"),
            cl.Action(name="reject", payload={"action": "reject"}, label="❌ Reject"),
            cl.Action(name="revise", payload={"action": "revise"}, label="✏️ Revise budget"),
        ]).send()
        return (res or {}).get("payload", {}).get("action", "reject")  # timeout -> reject (nothing ordered)

    async def _ask_budget() -> str | None:
        ans = await cl.AskUserMessage(content="New budget in $ (e.g. 4000)?", timeout=600).send()
        return (ans or {}).get("output")

    async def _notify(text: str) -> None:
        await cl.Message(content=text).send()

    @cl.on_message
    async def on_message(message: cl.Message) -> None:
        cfg = {"configurable": {"thread_id": f"ui-{uuid.uuid4().hex[:8]}"}}
        try:
            values = await drive(cl.user_session.get("graph"), message.content, cfg, _show_node,
                                 _ask_decision, _ask_budget, _notify)
            await _notify(final_markdown(values))
        except Exception as e:  # noqa: BLE001
            await _notify(f"⚠️ Run failed: `{type(e).__name__}: {e}`")
