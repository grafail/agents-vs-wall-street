"""LLM access. OpenAI-compatible (OpenAI or OpenRouter per settings).

Rules (see CLAUDE.md): never hardcode model IDs; keep prompts cache-friendly
(stable system prompt first, volatile content last, append-only history);
callers log usage to the run log.
"""
from typing import Any, Type

from openai import OpenAI
from pydantic import BaseModel

from pipeline.config import settings


def client() -> OpenAI:
    s = settings()
    kwargs: dict[str, Any] = {"api_key": s.api_key}
    if s.base_url:
        kwargs["base_url"] = s.base_url
    return OpenAI(**kwargs)


def model_id(size: str) -> str:
    s = settings()
    if size == "big":
        return s.model_big
    if size == "small":
        return s.model_small
    raise ValueError(f"unknown model size: {size}")


def _extra_body(size: str) -> dict:
    """OpenRouter extras: pin ONE backend provider (routing to different hosts
    destroys prompt-cache continuity — observed live: identical prompts, different
    tokenizers, cached_tokens always 0), request per-call billed cost, and dial
    reasoning to low for the small role."""
    s = settings()
    if s.llm_provider != "openrouter":
        return {}
    extra: dict = {"usage": {"include": True}}
    if s.openrouter_provider:
        extra["provider"] = {"order": [s.openrouter_provider], "allow_fallbacks": False}
    if size == "small" and s.small_reasoning_effort:
        extra["reasoning"] = {"effort": s.small_reasoning_effort}
    return extra


def complete_structured(
    size: str,
    messages: list[dict],
    schema: Type[BaseModel],
    **kwargs: Any,
) -> tuple[BaseModel, dict]:
    """Structured-output call. Returns (parsed pydantic object, usage dict incl cached tokens)."""
    c = client()
    extra = _extra_body(size)
    if extra:
        kwargs.setdefault("extra_body", {}).update(extra)
    resp = c.beta.chat.completions.parse(
        model=model_id(size), messages=messages, response_format=schema, **kwargs
    )
    usage = _usage_dict(resp)
    return resp.choices[0].message.parsed, usage


def complete_text(size: str, messages: list[dict], **kwargs: Any) -> tuple[str, dict]:
    c = client()
    extra = _extra_body(size)
    if extra:
        kwargs.setdefault("extra_body", {}).update(extra)
    resp = c.chat.completions.create(model=model_id(size), messages=messages, **kwargs)
    return resp.choices[0].message.content or "", _usage_dict(resp)


def chat_model(size: str, **kwargs: Any):
    """Provider-specific LangChain chat model for LangGraph nodes, selected by
    LLM_PROVIDER: ChatOpenRouter for openrouter, ChatOpenAI for openai.
    `size` is "big" | "small" (env-configured model IDs)."""
    s = settings()
    if s.llm_provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter
        return ChatOpenRouter(model=model_id(size), api_key=s.openrouter_api_key, **kwargs)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_id(size), api_key=s.openai_api_key, **kwargs)


def _usage_dict(resp: Any) -> dict:
    u = resp.usage
    cached = 0
    details = getattr(u, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    out = {
        "model": resp.model,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "cached_prompt_tokens": cached,
    }
    cost = getattr(u, "cost", None)  # OpenRouter returns billed USD with usage.include
    if cost is not None:
        out["cost_usd"] = cost
    return out
