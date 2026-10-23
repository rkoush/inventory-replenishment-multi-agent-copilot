"""LLM setup (Groq) plus an offline rule-based stand-in with the same interface.

Nodes only use two LangChain calls:  `llm.with_structured_output(Schema).invoke(msgs)`
and `llm.invoke(msgs).content`. `RuleBasedLLM` implements both, so the graph can be
run and tested without an API key (`--offline`).
"""
from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

import config
from agents import rules


# ------------------------------------------------------------- output schemas
class RequestPlan(BaseModel):
    """Structured interpretation of a planner's request."""

    store: str = Field(description="Store id exactly as written in the request, e.g. 'S001'. Empty if not given.")
    budget: Optional[float] = Field(default=None, description="Budget in USD if stated, else null.")
    horizon_days: int = Field(default=7, description="Planning horizon in days ('next week' = 7).")
    plan: list[str] = Field(description="Ordered list of workers to run, chosen from the allowed worker names.")
    reasoning: str = Field(description="One sentence explaining the plan.")


class CategorySafetyStock(BaseModel):
    category: str
    days: int


class PromoUplift(BaseModel):
    sku: str
    uplift_pct: float = Field(description="Expected demand uplift in percent, e.g. 40 for 40%.")


class PolicyExtraction(BaseModel):
    """Planning constraints stated in the retrieved policy text."""

    safety_stock: list[CategorySafetyStock] = Field(description="Safety stock days per product category.")
    max_sku_budget_share_pct: Optional[float] = Field(default=None, description="Max % of budget one SKU may use.")
    weekly_budget_cap: Optional[float] = Field(default=None, description="Weekly store budget cap in USD.")
    promos: list[PromoUplift] = Field(default_factory=list, description="Promoted SKUs and their uplift.")
    summary: str = Field(description="2-3 sentence summary of the constraints that matter for this order.")


# ------------------------------------------------------------- offline LLM
class RuleBasedLLM:
    """Deterministic drop-in for a chat model (no network, no key)."""

    name = "rule-based (offline)"

    def with_structured_output(self, schema: type[BaseModel], **_: Any) -> RunnableLambda:
        def run(messages: list[BaseMessage] | str) -> BaseModel:
            text = _last_text(messages)
            if schema is RequestPlan:
                return RequestPlan(**rules.parse_request(text))
            if schema is PolicyExtraction:
                return PolicyExtraction(**rules.extract_policy(text))
            raise NotImplementedError(schema)

        return RunnableLambda(run)

    def invoke(self, messages: list[BaseMessage] | str, **_: Any) -> AIMessage:
        return AIMessage(content=rules.explain_plan(rules.extract_json_block(_last_text(messages))))


def _last_text(messages: list[BaseMessage] | str) -> str:
    if isinstance(messages, str):
        return messages
    return str(messages[-1].content)


# ------------------------------------------------------------- factory
def get_llm(offline: bool = False, model: str | None = None):
    """Return a Groq chat model, or the offline stand-in if requested / no key is set.

    Free-tier friendly: small max_tokens, temperature 0, and the Groq SDK's built-in
    exponential backoff on 429 rate-limit responses (max_retries).
    """
    if offline or not os.getenv("GROQ_API_KEY"):
        return RuleBasedLLM()
    from langchain_groq import ChatGroq

    model = model or config.GROQ_MODEL
    llm = ChatGroq(
        model=model,
        temperature=config.GROQ_TEMPERATURE,
        max_tokens=config.GROQ_MAX_TOKENS,
        max_retries=config.GROQ_MAX_RETRIES,
        timeout=60,
    )
    llm.name = f"groq:{model}"
    return llm


def llm_name(llm: Any) -> str:
    return getattr(llm, "name", None) or type(llm).__name__
