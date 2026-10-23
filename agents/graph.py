"""LangGraph wiring: supervisor + 4 workers + human approval + order placement + report.

    START -> supervisor -> {inventory, demand, policy, optimizer}_agent -> supervisor ...
          -> human_approval --approve--> place_order -> report -> END
                            --revise---> optimizer_agent (new budget) -> supervisor -> human_approval
                            --reject---> report -> END

LLM calls per run (Groq free-tier friendly): supervisor plan (1), policy extraction (1),
report narrative (1) = 3 calls. Everything else is MCP tools, pandas, or the MILP solver.
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.demand_agent import make_demand_agent
from agents.human_approval import human_approval, make_place_order
from agents.inventory_agent import make_inventory_agent
from agents.llm import get_llm
from agents.mcp_client import clients_for_roles
from agents.optimizer_agent import optimizer_agent
from agents.policy_agent import make_policy_agent
from agents.report import make_report
from agents.state import ReplenishState
from agents.supervisor import WORKERS, make_supervisor


def build_graph(llm: Any = None, router_llm: Any = None, mcp_client: Any = None, checkpointer: Any = None,
                mcp_clients: dict[str, Any] | None = None):
    """mcp_client: one client for every MCP step. mcp_clients: per role, e.g.
    {"planner": ..., "approver": ...}. With neither, each step builds a client for its role
    (planner for inventory/demand reads, approver for place_order) from config."""
    clients = clients_for_roles(mcp_client, mcp_clients)
    llm = llm or get_llm()
    router_llm = router_llm or llm
    g = StateGraph(ReplenishState)
    g.add_node("supervisor", make_supervisor(router_llm))
    g.add_node("inventory_agent", make_inventory_agent(clients["planner"]))
    g.add_node("demand_agent", make_demand_agent(clients["planner"]))
    g.add_node("policy_agent", make_policy_agent(llm))
    g.add_node("optimizer_agent", optimizer_agent)
    g.add_node("human_approval", human_approval)
    g.add_node("place_order", make_place_order(clients["approver"]))
    g.add_node("report", make_report(llm))

    g.add_edge(START, "supervisor")
    for w in WORKERS:  # every worker reports back to the supervisor
        g.add_edge(w, "supervisor")
    g.add_edge("place_order", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
