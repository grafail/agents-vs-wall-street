"""Cross-provider estimator panel with deterministic aggregation.

Panel membership comes from settings().estimator_panel — comma-separated
"provider:model" entries (provider = openrouter|openai). An EMPTY setting means
a single-entry panel using the run's llm_provider + MODEL_BIG through the exact
same code path as before (llm.complete_structured("big", ...) — provider pin,
cache extras and all), so default behavior is unchanged.

All aggregation is deterministic quant code (no LLM), per metric label:

- growth quantiles: p10/p50/p90 = element-wise MEDIAN across surviving members.
- cross-model disagreement widens uncertainty:
      d   = max(p50_i) - min(p50_i)
      p10 = min(median(p10_i), p50_agg - d/2)
      p90 = max(median(p90_i), p50_agg + d/2)
  so models that disagree on the central call force a wider band (which the
  nudge layer then shrinks toward zero — disagreement lowers panel influence).
- categorical labels: majority vote; ties resolve to the MORE NEUTRAL label
  (momentum: neutral > cooling > warming > cold > hot;
   surprise_skew: balanced > downside > upside;
   guidance_style: accurate > sandbagger > promotional).
  Aggregated Grounded.explanation = "panel N/M: " + the first winning voter's
  explanation (N = votes for winner, M = members voting on the metric);
  citations = order-preserving union of the winning voters' citations.
- confidence: majority vote; tie -> "low".
- method: majority vote; tie -> earliest tied method in member input order.

The blind rule applies unchanged: every panelist gets the SAME consensus-free
messages. Failures (including a missing API key for a member's provider) are
logged as panel_member_failed and skipped; zero survivors => None (the graph's
failsafe cascade covers the metrics).
"""
from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from typing import Callable

from pipeline import llm
from pipeline.config import settings
from pipeline.types import CompanyEstimates, Estimate, MetricEstimate, MetricSpec

# Tie-break priority per categorical field: more neutral labels first.
_TIE_ORDER: dict[str, list[str]] = {
    "momentum": ["neutral", "cooling", "warming", "cold", "hot"],
    "guidance_style": ["accurate", "sandbagger", "promotional"],
    "surprise_skew": ["balanced", "downside", "upside"],
}


def panel_members() -> list[tuple[str, str] | None]:
    """Parse ESTIMATOR_PANEL. None entry = the default single member (current
    provider + MODEL_BIG via llm.complete_structured — exact legacy behavior)."""
    raw = (settings().estimator_panel or "").strip()
    if not raw:
        return [None]
    out: list[tuple[str, str] | None] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        provider, _, model = entry.partition(":")
        out.append((provider.strip(), model.strip()))
    return out


def _member_id(member: tuple[str, str] | None) -> str:
    if member is None:
        s = settings()
        return f"{s.llm_provider}:{s.model_big or 'MODEL_BIG'}"
    return f"{member[0]}:{member[1]}"


def _missing_key_reason(member: tuple[str, str] | None) -> str | None:
    """Pre-flight check: an explicit panel member whose provider key is absent
    is skipped without burning a call (or a retry sleep). The default member
    (empty panel) is NOT pre-checked — exact legacy single-call behavior."""
    if member is None:
        return None
    provider = member[0]
    if provider == "openrouter":
        return None if settings().openrouter_api_key else "missing API key for openrouter"
    if provider == "openai":
        return None if settings().openai_api_key else "missing API key for openai"
    return f"unknown provider {provider!r} (openrouter|openai)"


def _call_member(member: tuple[str, str] | None, messages: list[dict]):
    if member is None:
        return llm.complete_structured("big", messages, CompanyEstimates)
    return llm.complete_structured_at(member[0], member[1], messages, CompanyEstimates)


def _call_with_retry(member: tuple[str, str] | None, messages: list[dict], log):
    """ONE retry with 2s backoff (same pattern as graph._structured_retry):
    a transient blip during the final-run window must not drop a panelist."""
    try:
        return _call_member(member, messages)
    except Exception as e:  # noqa: BLE001
        if log:
            log.event("llm_retry", stage="estimate_panel", member=_member_id(member),
                      error=f"{type(e).__name__}: {e}")
        time.sleep(2)
        return _call_member(member, messages)


# ---------------------------------------------------------------- aggregation

def _vote(grounded: list, field: str) -> dict:
    """Majority vote over one Grounded categorical across members. Returns the
    aggregated Grounded as a plain dict (pydantic validates on Estimate build)."""
    counts = Counter(g.label for g in grounded)
    top = max(counts.values())
    winners = [lab for lab in counts if counts[lab] == top]
    order = _TIE_ORDER[field]
    win = min(winners, key=lambda lab: order.index(lab) if lab in order else len(order))
    voters = [g for g in grounded if g.label == win]
    citations: list[str] = []
    for g in voters:  # order-preserving union
        for c in g.citations:
            if c not in citations:
                citations.append(c)
    return {
        "explanation": f"panel {len(voters)}/{len(grounded)}: {voters[0].explanation}",
        "label": win,
        "citations": citations,
    }


def _aggregate_metric(ests: list[MetricEstimate]) -> Estimate:
    p50s = [e.growth_p50 for e in ests]
    p50 = median(p50s)
    d = max(p50s) - min(p50s)
    p10 = min(median(e.growth_p10 for e in ests), p50 - d / 2.0)
    p90 = max(median(e.growth_p90 for e in ests), p50 + d / 2.0)

    conf_counts = Counter(e.confidence for e in ests)
    conf_top = max(conf_counts.values())
    conf_winners = [c for c in conf_counts if conf_counts[c] == conf_top]
    confidence = conf_winners[0] if len(conf_winners) == 1 else "low"

    method_counts = Counter(e.method for e in ests)
    m_top = max(method_counts.values())
    method = next(e.method for e in ests if method_counts[e.method] == m_top)

    return Estimate(
        method=method,
        growth_p10=p10, growth_p50=p50, growth_p90=p90,
        momentum=_vote([e.momentum for e in ests], "momentum"),
        guidance_style=_vote([e.guidance_style for e in ests], "guidance_style"),
        surprise_skew=_vote([e.surprise_skew for e in ests], "surprise_skew"),
        confidence=confidence,
        rationale=f"panel {len(ests)} member(s): {ests[0].rationale}",
    )


def aggregate(per_model: list[CompanyEstimates]) -> dict[str, Estimate]:
    """Deterministic per-label aggregation across panelists (formulas in the
    module docstring). Labels group case-insensitively; the first-seen spelling
    is the returned key. A single member aggregates to itself (identity, given
    ordered quantiles)."""
    groups: dict[str, tuple[str, list[MetricEstimate]]] = {}
    for ce in per_model:
        for me in ce.estimates:
            key = me.metric_label.strip().casefold()
            groups.setdefault(key, (me.metric_label, []))[1].append(me)
    return {display: _aggregate_metric(ests) for display, ests in groups.values()}


def _p50_spread(per_model: list[CompanyEstimates]) -> dict[str, float]:
    spread: dict[str, list[float]] = {}
    for ce in per_model:
        for me in ce.estimates:
            spread.setdefault(me.metric_label.strip().casefold(), []).append(me.growth_p50)
    return {k: (max(v) - min(v)) for k, v in spread.items()}


# ---------------------------------------------------------------- entry point

def panel_estimate(
    specs: list[MetricSpec],
    messages_builder: Callable[[], list[dict]],
    log=None,
) -> tuple[dict[str, Estimate] | None, dict]:
    """Run the estimator panel for one company.

    Every panelist gets the SAME blind messages (messages_builder()); members
    run in parallel; each gets one retry; failures (incl. missing API key) are
    logged as panel_member_failed and skipped. Returns
    (label -> aggregated Estimate, audit summary). 0 survivors => (None, summary).
    The summary is a plain JSON-able dict for blob["panel"] (per-member p50s and
    categorical labels, failures, p50 spreads)."""
    ticker = specs[0].ticker if specs else "?"
    messages = messages_builder()
    members = panel_members()

    ok: list[tuple[str, CompanyEstimates]] = []
    failed: list[dict] = []

    def _run(member: tuple[str, str] | None):
        mid = _member_id(member)
        reason = _missing_key_reason(member)
        if reason is not None:
            raise RuntimeError(reason)
        parsed, usage = _call_with_retry(member, messages, log)
        if log:
            log.event("llm_call", stage="estimate_panel", ticker=ticker,
                      member=mid, **usage)
        return parsed

    with ThreadPoolExecutor(max_workers=max(1, len(members))) as pool:
        futures = [(m, pool.submit(_run, m)) for m in members]
        for member, fut in futures:  # input order => deterministic aggregation
            mid = _member_id(member)
            try:
                ok.append((mid, fut.result()))
            except Exception as e:  # noqa: BLE001 — member failure tolerated
                failed.append({"member": mid, "error": f"{type(e).__name__}: {e}"})
                if log:
                    log.event("panel_member_failed", ticker=ticker, member=mid,
                              error=f"{type(e).__name__}: {e}")

    spreads = _p50_spread([ce for _, ce in ok])
    summary = {
        "members_ok": [mid for mid, _ in ok],
        "members_failed": failed,
        "p50_spread": spreads,
        "per_member": {
            mid: {
                me.metric_label: {
                    "p50": me.growth_p50,
                    "momentum": me.momentum.label,
                    "guidance_style": me.guidance_style.label,
                    "surprise_skew": me.surprise_skew.label,
                    "confidence": me.confidence,
                }
                for me in ce.estimates
            }
            for mid, ce in ok
        },
    }
    if log:
        log.event("panel_aggregated", ticker=ticker,
                  members_ok=len(ok), members_failed=len(failed),
                  p50_spread=spreads)
    if not ok:
        return None, summary
    return aggregate([ce for _, ce in ok]), summary
