"""LLM setup (Groq) plus an offline rule-based stand-in with the same interface.

Nodes only use two LangChain calls:  `llm.with_structured_output(Schema).invoke(msgs)`
and `llm.invoke(msgs).content`. `RuleBasedLLM` implements both, so the graph can be
run and tested without an API key (`--offline`).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
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
        return AIMessage(content=rules.explain_why(rules.extract_json_block(_last_text(messages))))


def _last_text(messages: list[BaseMessage] | str) -> str:
    if isinstance(messages, str):
        return messages
    return str(messages[-1].content)


# ------------------------------------------------------------- factory
def is_gpt_oss(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def is_qwen_reasoning(model: str) -> bool:
    return model.startswith("qwen/qwen3")


def reasoning_kwargs(model: str) -> dict[str, Any]:
    """Groq reasoning controls, only for models that accept them (others would 400).

    - gpt-oss: `reasoning_effort` is supported, `reasoning_format` is NOT; reasoning is
      dropped from the response with `include_reasoning=False` (passed via model_kwargs,
      since ChatGroq has no field for it; the groq SDK accepts it).
    - qwen3: `reasoning_effort` + `reasoning_format="hidden"` (final answer only).
    Reasoning tokens still count towards max_tokens, hence the low effort.
    """
    effort = config.GROQ_REASONING_EFFORT
    if is_gpt_oss(model):
        return {"reasoning_effort": effort, "model_kwargs": {"include_reasoning": False}}
    if is_qwen_reasoning(model):
        return {"reasoning_effort": effort, "reasoning_format": "hidden"}
    return {}


@lru_cache(maxsize=1)
def _chat_groq_cls():
    from langchain_groq import ChatGroq

    class _ChatGroq(ChatGroq):
        """ChatGroq that omits unset (None) params instead of sending JSON nulls.

        langchain-groq always sends `reasoning_format: null`; Groq treats
        `reasoning_format` and `include_reasoning` as mutually exclusive, so we don't
        send keys we never set.
        """

        @property
        def _default_params(self) -> dict[str, Any]:
            return {k: v for k, v in super()._default_params.items() if v is not None}

    return _ChatGroq


def get_llm(offline: bool = False, model: str | None = None):
    """Return a Groq chat model, or the offline stand-in if requested / no key is set.

    Free-tier friendly: capped max_tokens, temperature 0, low reasoning effort, and the
    Groq SDK's built-in exponential backoff on 429 rate-limit responses (max_retries).
    """
    if offline or not os.getenv("GROQ_API_KEY"):
        return RuleBasedLLM()
    model = model or config.GROQ_MODEL
    llm = _chat_groq_cls()(
        model=model,
        temperature=config.GROQ_TEMPERATURE,
        max_tokens=config.GROQ_MAX_TOKENS,
        max_retries=config.GROQ_MAX_RETRIES,
        timeout=60,
        **reasoning_kwargs(model),
    )
    llm.name = f"groq:{model}"
    return llm


def _strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for Groq strict mode: all fields required, closed objects, no defaults."""
    from langchain_core.utils.function_calling import convert_to_json_schema

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if k != "default"}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(convert_to_json_schema(schema, strict=True))


def structured(llm: Any, schema: type[BaseModel]):
    """Runnable returning a `schema` instance.

    gpt-oss / qwen3 on Groq: JSON-schema structured outputs (strict, constrained decoding
    where Groq supports it). Other chat models: tool calling. Offline stand-ins: their own.
    Callers wrap this in try/except and fall back to rules on any failure.
    """
    model = getattr(llm, "model_name", "") or ""
    if is_gpt_oss(model) or is_qwen_reasoning(model):
        runnable = llm.with_structured_output(_strict_schema(schema), method="json_schema", strict=True)
        return runnable | RunnableLambda(lambda d: schema.model_validate(d))
    return llm.with_structured_output(schema)


_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_reasoning(text: str) -> str:
    """Remove any inline <think>...</think> block (raw reasoning format)."""
    return _THINK.sub("", text or "").strip()


def smoke_test(model: str | None = None) -> tuple[bool, str]:
    """One tiny Groq call to check the key + model before a run. Returns (ok, message)."""
    if not os.getenv("GROQ_API_KEY"):
        return False, "GROQ_API_KEY not set -> offline (rule-based) mode"
    model = model or config.GROQ_MODEL
    try:
        reply = get_llm(model=model).invoke("Reply with OK", max_tokens=256)
        text = strip_reasoning(str(reply.content))
        return True, f"Groq reachable, model={model}, reply={text[:40]!r}"
    except Exception as exc:  # noqa: BLE001 - surface any failure (NotFound, auth, 429, ...)
        return False, f"Groq call FAILED for model={model}: {type(exc).__name__}: {str(exc)[:200]}"


def llm_name(llm: Any) -> str:
    return getattr(llm, "name", None) or type(llm).__name__
