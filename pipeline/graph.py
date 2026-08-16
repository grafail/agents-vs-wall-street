"""LangGraph per-company pipeline: load → research → estimate → consensus →
reconcile → finalize → write.

Every node degrades gracefully: a failed stage records the error and lets the
finalize cascade guarantee a number (never-blank rule). Every node logs to the
RunLog. Model access only via pipeline.llm (chat_model factory / complete_*).
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from pipeline import ensemble, llm, tools
from pipeline.backtest import beat_factor, best_method
from pipeline.baselines import Series, interim_map, run_all, trend_sheet, yoy_sigma
from pipeline.baselines import prior_year_period
from pipeline.config import settings, yf_ticker
from pipeline.extract import ExtractedFact, extract_company, load_facts
from pipeline.nudges import apply_nudges, reconcile_blend
from pipeline.prompts import (
    ESTIMATOR_SYSTEM,
    RECONCILER_SYSTEM,
    RESEARCH_SYSTEM,
    build_company_context,
    build_metric_block,
    build_reconciler_context,
)
from pipeline.runlog import RunLog
from pipeline.types import Estimate, MetricSpec, Reconciliation, metric_specs
from pipeline.validate import resolve_final_value, run_all_gates, sibling_coherence
from pipeline.writer import verify_workbook, write_workbook


RESEARCH_RECURSION_LIMIT = 20  # ~8 tool iterations for the react scout


def _structured_retry(size: str, messages: list, schema, log=None, **ctx):
    """complete_structured with ONE retry (2s backoff) — a transient API blip
    during the final-run window must not silently drop a metric to baseline."""
    import time as _time
    try:
        return llm.complete_structured(size, messages, schema)
    except Exception as e:  # noqa: BLE001
        if log:
            log.event("llm_retry", error=f"{type(e).__name__}: {e}", **ctx)
        _time.sleep(2)
        return llm.complete_structured(size, messages, schema)


class CompanyState(TypedDict, total=False):
    ticker: str
    log: Any                    # RunLog (opaque to langgraph)
    facts: list[Any]            # list[ExtractedFact]
    digest: list[str]           # fresh-evidence bullets from the research scout
    metrics: dict[str, dict]    # label -> audit blob (built up node by node)
    workbook: str
    verify_issues: list[str]
    errors: list[str]
    out_dir: Any                # Path override for tests (default: SUBMISSION_DIR)


def _specs(ticker: str) -> list[MetricSpec]:
    return [s for s in metric_specs() if s.ticker == ticker]


def _series_facts(facts: list[ExtractedFact], label: str,
                  period_type: str = "quarter") -> list[ExtractedFact]:
    """Actuals for one metric, GRANULARITY-MATCHED to the target: quarterly
    targets get only quarterly facts, annual targets only annual facts.
    Mixing them poisoned YoY stats (HD: annual/quarter pairs gave +6488%
    growth => sigma 2400%). H1 half-years are excluded either way."""
    from pipeline.baselines import period_key
    out = []
    for f in facts:
        if f.metric_label != label or f.fact_type != "actual":
            continue
        try:
            _, q = period_key(f.period)
        except ValueError:
            continue
        if (period_type == "quarter") != (q > 0):
            continue
        out.append(f)
    return out


def _guidance_facts(facts: list[ExtractedFact], label: str) -> list[ExtractedFact]:
    return [f for f in facts if f.metric_label == label and f.fact_type.startswith("guidance")]


def _consensus_facts(facts: list[ExtractedFact], label: str) -> list[ExtractedFact]:
    return [f for f in facts if f.metric_label == label and f.fact_type.startswith("consensus")]


_CONSENSUS_QUOTE = ("consensus", "analyst estimate", "analysts' estimate", "market expectation")


def _reclassify_consensus(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Blind-integrity guard for artifacts extracted before consensus fact types
    existed: guidance facts whose quote attributes the figure to consensus are
    reclassified consensus_* so they never reach the blind estimator. A company
    expectation stated RELATIVE to consensus ('at the top of the consensus
    range') stays guidance only when the quote also carries expectation wording."""
    out = []
    for f in facts:
        if f.fact_type.startswith("guidance"):
            q = f.quote.lower()
            if any(k in q for k in _CONSENSUS_QUOTE) and not any(
                    w in q for w in ("we expect", "we currently expect", "company expects",
                                     "expected to be", "guidance")):
                f = f.model_copy(update={
                    "fact_type": f.fact_type.replace("guidance", "consensus")})
        out.append(f)
    return out


# ================================================================ nodes

def node_load(state: CompanyState) -> dict:
    ticker, log = state["ticker"], state.get("log")
    try:
        facts = load_facts(ticker)
    except FileNotFoundError:
        if log:
            log.event("load_facts_missing", ticker=ticker, action="extracting now")
        extract_company(ticker, log=log)
        facts = load_facts(ticker)
    facts = _reclassify_consensus(facts)
    # Unit auto-correction ALSO at load time: artifacts extracted before a scale
    # factor was known (e.g. DE's x1e6 micro-mislabels) get repaired without
    # re-extraction. Corrections carry audit flags like any extraction-time fix.
    from pipeline.extract import auto_correct_units
    auto_correct_units(facts)
    n_fixed = sum(1 for f in facts
                  if any(fl.startswith("auto_corrected_scale") for fl in f.flags))
    if log:
        n_cons = sum(1 for f in facts if f.fact_type.startswith("consensus"))
        log.event("load_facts", ticker=ticker, n=len(facts), consensus_facts=n_cons,
                  units_auto_corrected=n_fixed)
    return {"facts": facts, "metrics": {}, "errors": state.get("errors", [])}


def _research_digest(ticker: str, log: RunLog | None) -> list[str]:
    """Live react scout on the small model. Any failure => empty digest."""
    from langchain_core.tools import tool as lc_tool
    from langgraph.prebuilt import create_react_agent

    @lc_tool
    def web_search(query: str) -> str:
        """Search the web. Returns JSON results with url/title/snippet/trust tier."""
        data, _ = tools.search_web(query, log=log)
        return json.dumps(data, default=str)[:4000]

    @lc_tool
    def news_search(query: str, days: int = 7) -> str:
        """Search recent news (LEADS only — verify via fetch_page). days: lookback window."""
        data, _ = tools.search_news(query, days=days, log=log)
        return json.dumps(data, default=str)[:4000]

    @lc_tool
    def fetch_page(url: str) -> str:
        """Fetch a URL and return clean extracted text (capped)."""
        data, _ = tools.fetch_url(url, log=log)
        return json.dumps(data, default=str)[:6000]

    @lc_tool
    def market_data(ticker: str, what: str) -> str:
        """yfinance market data. what: info | quarterly_financials | news."""
        if what == "estimates":  # blind rule: the research path must never fetch consensus
            return json.dumps({"error": "estimates are not available to research"})
        data, _ = tools.get_market_data(ticker, what, log=log)
        return json.dumps(data, default=str)[:4000]

    spec0 = _specs(ticker)[0]
    agent = create_react_agent(
        llm.chat_model("small"),
        tools=[web_search, news_search, fetch_page, market_data],
        prompt=RESEARCH_SYSTEM,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content":
            f"Research {spec0.company} ({ticker}) ahead of its {spec0.period} report. "
            f"Metrics of interest: " + ", ".join(s.label for s in _specs(ticker)) + "."}]},
        config={"recursion_limit": RESEARCH_RECURSION_LIMIT},
    )
    # Log the scout's own LLM usage — these calls otherwise bypass llm.py's
    # usage accounting (LangChain path) and would be invisible in the trace.
    if log is not None:
        n_calls, p_tok, c_tok = 0, 0, 0
        for msg in result["messages"]:
            um = getattr(msg, "usage_metadata", None)
            if um:
                n_calls += 1
                p_tok += um.get("input_tokens", 0)
                c_tok += um.get("output_tokens", 0)
        if n_calls:
            log.event("llm_call", stage="research", ticker=ticker, model="small",
                      calls=n_calls, prompt_tokens=p_tok, completion_tokens=c_tok)

    content = result["messages"][-1].content
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    bullets = [ln.strip().lstrip("-• ").strip() for ln in text.splitlines()
               if ln.strip().lstrip("-• ").strip()]
    return bullets[:20]


def node_research(state: CompanyState) -> dict:
    ticker, log = state["ticker"], state.get("log")
    if not settings().enable_research:
        if log:
            log.event("research_skipped", ticker=ticker, reason="ENABLE_RESEARCH=false")
        return {"digest": []}
    try:
        # Digest-level cache: the scout's tools were already cached, but its LLM
        # reasoning re-ran on every forecast. One digest per (ticker, model) per
        # cache lifetime; prepare --refresh bypasses via state["refresh"].
        from pipeline.cache import read_through
        digest, was_hit = read_through(
            "research_digest",
            {"ticker": ticker, "model": settings().model_small},
            lambda: _research_digest(ticker, log),
            refresh=bool(state.get("refresh")),
        )
        if log:
            log.event("research_done", ticker=ticker, bullets=len(digest), cache_hit=was_hit)
        return {"digest": digest}
    except Exception as e:  # noqa: BLE001 — research is optional enrichment
        if log:
            log.event("research_failed", ticker=ticker, error=f"{type(e).__name__}: {e}")
        return {"digest": [], "errors": state.get("errors", []) + [f"research: {e}"]}


def node_estimate(state: CompanyState) -> dict:
    """Company-level estimation: per-metric deterministic prep exactly as
    before, then ONE blind panel call for the whole company (the model sees all
    sibling metrics and keeps them coherent — profit vs EPS in one story).
    Per-metric nudging/finalize is unchanged; a metric missing from the panel
    response falls to the failsafe cascade."""
    ticker, log = state["ticker"], state.get("log")
    facts = state.get("facts", [])
    digest = state.get("digest", [])
    metrics = dict(state.get("metrics", {}))

    # 1. deterministic per-metric prep (baselines/backtests/trend/anchor/sigma)
    prepped: list[tuple[MetricSpec, str]] = []   # specs with baselines + context block
    for spec in _specs(ticker):
        try:
            block = _prep_one(spec, facts, metrics)
            if block is not None:
                prepped.append((spec, block))
        except Exception as e:  # noqa: BLE001 — one metric must never kill the company
            if log:
                log.event("metric_failed", ticker=ticker, metric=spec.label,
                          error=f"{type(e).__name__}: {e}")
            blob = metrics.get(spec.label, {})
            blob.setdefault("spec", spec)
            blob["error"] = f"{type(e).__name__}: {e}"
            metrics[spec.label] = blob
    if not prepped:
        return {"metrics": metrics}

    # 2. ONE blind panel call per company (single-entry panel when
    # ESTIMATOR_PANEL is empty — exact previous single-model behavior)
    est_specs = [s for s, _ in prepped]

    def messages_builder() -> list[dict]:
        return [{"role": "system", "content": ESTIMATOR_SYSTEM},
                {"role": "user", "content": build_company_context(
                    est_specs, [b for _, b in prepped], list(digest))}]

    try:
        est_map, panel_summary = ensemble.panel_estimate(est_specs, messages_builder, log)
    except Exception as e:  # noqa: BLE001 — cascade will cover every metric
        if log:
            log.event("estimate_failed", ticker=ticker, metric="*company*",
                      error=f"{type(e).__name__}: {e}")
        est_map, panel_summary = None, {"members_ok": [], "members_failed": [
            {"member": "panel", "error": f"{type(e).__name__}: {e}"}]}

    # 3. map returned estimates back by metric_label (exact, else case-insensitive)
    by_fold = {k.strip().casefold(): v for k, v in (est_map or {}).items()}
    for spec in est_specs:
        blob = metrics[spec.label]
        blob["panel"] = panel_summary
        est = (est_map or {}).get(spec.label) or by_fold.get(spec.label.strip().casefold())
        if est is None:
            reason = ("panel produced no estimates" if not est_map else
                      f"metric missing from company estimate response: {spec.label!r}")
            if log:
                log.event("estimate_failed", ticker=ticker, metric=spec.label,
                          error=reason)
            blob["estimate"] = None
            blob["estimate_error"] = reason
            metrics[spec.label] = blob
            continue
        try:
            blob["estimate"] = est
            chosen = blob["baselines"][0]
            mae = (chosen["backtest"] or {}).get("mae")
            blob["nudges"] = apply_nudges(
                chosen["value"], est, mae, blob.get("sigma"), spec,
                backtest_n=(chosen["backtest"] or {}).get("n"))
        except Exception as e:  # noqa: BLE001 — cascade will cover this metric
            if log:
                log.event("estimate_failed", ticker=ticker, metric=spec.label,
                          error=f"{type(e).__name__}: {e}")
            blob["estimate"] = None
            blob["estimate_error"] = f"{type(e).__name__}: {e}"
        metrics[spec.label] = blob
    return {"metrics": metrics}


def _prep_one(spec: MetricSpec, facts: list, metrics: dict) -> str | None:
    """Deterministic per-metric prep (NO LLM): baselines + backtests, trend
    sheet, anchor, sigma, interim actuals — stored on the metric's audit blob.
    Returns the metric's estimator context block, or None when no baseline
    candidates exist (the metric skips estimation; cascade covers it)."""
    blob: dict[str, Any] = metrics.get(spec.label, {})
    sfacts = _series_facts(facts, spec.label, spec.period_type)
    gfacts = _guidance_facts(facts, spec.label)
    series = Series(sfacts)
    beat = beat_factor(gfacts, sfacts)
    all_metric_facts = [f for f in facts if f.metric_label == spec.label]
    interims = interim_map(all_metric_facts)
    candidates = run_all(series, spec, gfacts, beat, interims=interims)
    ranked = best_method(series, spec, gfacts, beat, interims=interims)
    bt_by_method = {r["method"]: r for r in ranked}
    cands_bt = sorted(
        [{"method": c["method"], "value": c["value"], "inputs_used": c["inputs_used"],
          "backtest": bt_by_method.get(c["method"])} for c in candidates],
        key=lambda c: (c["backtest"] is None, (c["backtest"] or {}).get("mae", 0.0)),
    )
    trend = trend_sheet(series, spec, gfacts)
    sigma = yoy_sigma(series, spec)

    # anchor: same period last year, with its quote for the report
    anchor_period = prior_year_period(spec.period)
    anchor_value = series.value(anchor_period)
    anchor_fact = next((f for f in sfacts if f.period == anchor_period), None)

    blob.update({
        "spec": spec,
        "series_values": [v for _, v in series.points],
        "guidance_facts": gfacts,
        "beat": beat,
        "baselines": cands_bt,
        "trend": trend,
        "sigma": sigma,
        "anchor": {"value": anchor_value, "period": anchor_period,
                   "quote": anchor_fact.quote if anchor_fact else None,
                   "source_doc": anchor_fact.source.doc_id if anchor_fact else None},
        "fact_flags": sorted({fl for f in sfacts + gfacts for fl in f.flags}),
    })
    metrics[spec.label] = blob

    if not cands_bt:
        blob["estimate"] = None
        blob["estimate_error"] = "no baseline candidates (insufficient history)"
        return None
    blob["baseline_chosen"] = cands_bt[0]["method"]

    # Interim (half-year) actuals can't enter Series math but are prime
    # evidence for annual metrics (e.g. Hays H1-FY2026 net fees) — feed
    # them to the estimator as cited evidence lines.
    interim = [
        f"INTERIM ACTUAL {f.period}: {f.metric_label} = {f.value} "
        f"[{f.source.doc_id}] \"{f.quote[:160]}\""
        for f in facts
        if f.metric_label == spec.label and f.fact_type == "actual"
        and f not in sfacts and "H" in f.period
    ]
    return build_metric_block(spec, sfacts, gfacts, trend, cands_bt, interim)


def _consensus_for(spec: MetricSpec, est_data: dict) -> tuple[float | None, str]:
    """Map yfinance estimate tables to one metric's consensus, defensively.
    Returns (value in canonical units | None, note). HAS is skipped entirely:
    'Net fees' is not revenue, and LSE currency/units are ambiguous on yahoo."""
    if spec.ticker == "HAS" or spec.kind == "ratio_pct" or spec.scope == "segment":
        return None, "no comparable consensus series"
    if spec.ticker == "DE" and spec.kind == "flow_absolute":
        # Basis mismatch: Yahoo tracks Deere's equipment-operations net sales
        # (~10.7B); the contest metric is WORLDWIDE net sales and revenues incl.
        # financial services (~12B+). Blending toward the wrong basis moved the
        # submitted number ~$370M in the first full run — skip instead.
        return None, "no comparable consensus series (yahoo tracks equipment-ops, not worldwide)"
    horizon = "0y" if spec.period_type == "fiscal_year" else "0q"
    try:
        if spec.kind == "flow_absolute":
            raw = est_data["revenue_estimate"]["avg"][horizon]
            if raw is None:
                return None, "consensus table empty"
            return float(raw) / 1e6, f"yfinance revenue_estimate avg [{horizon}]"
        if spec.kind == "per_share":
            raw = est_data["earnings_estimate"]["avg"][horizon]
            if raw is None:
                return None, "consensus table empty"
            return float(raw), f"yfinance earnings_estimate avg [{horizon}]"
    except (KeyError, TypeError, ValueError) as e:
        return None, f"consensus parse failed: {type(e).__name__}"
    return None, "unmapped metric kind"


def node_consensus(state: CompanyState) -> dict:
    ticker, log = state["ticker"], state.get("log")
    metrics = dict(state.get("metrics", {}))
    if not settings().enable_reconcile:
        if log:
            log.event("consensus_skipped", ticker=ticker, reason="ENABLE_RECONCILE=false")
        return {}
    facts = state.get("facts", [])
    data, hit = tools.get_market_data(yf_ticker(ticker), "estimates", log=log)
    est_data = (data or {}).get("data") if isinstance(data, dict) else None
    for spec in _specs(ticker):
        blob = metrics.get(spec.label, {})

        # Corpus-published compiled consensus first (tier 1, exact metric basis —
        # e.g. Hays RNS: "company compiled consensus ... is £43.5m"). Beats any
        # aggregator mapping when present for the target period.
        cfacts = [f for f in _consensus_facts(facts, spec.label) if f.period == spec.period]
        mids = [f for f in cfacts if f.fact_type == "consensus_mid"]
        lows = [f for f in cfacts if f.fact_type == "consensus_low"]
        highs = [f for f in cfacts if f.fact_type == "consensus_high"]
        corpus_val, corpus_src = None, None
        rng = (f" range {lows[-1].value}-{highs[-1].value}" if lows and highs else "")
        if mids:
            corpus_val = mids[-1].value
            corpus_src = (f"corpus compiled consensus{rng} [{mids[-1].source.doc_id}] "
                          f"— company statement: \"{mids[-1].quote[:160]}\"")
        elif lows and highs:
            corpus_val = (lows[-1].value + highs[-1].value) / 2.0
            corpus_src = (f"corpus consensus range midpoint{rng} [{lows[-1].source.doc_id}] "
                          f"— company statement: \"{lows[-1].quote[:160]}\"")
        if corpus_val is not None:
            blob["consensus"] = {"value": corpus_val, "source": corpus_src}
            metrics[spec.label] = blob
            if log:
                log.event("consensus_from_corpus", ticker=ticker, metric=spec.label,
                          value=corpus_val, source=corpus_src)
            continue

        if not isinstance(est_data, dict):
            blob["consensus"] = {"value": None, "source": "yfinance estimates unavailable"}
            metrics[spec.label] = blob
            continue
        value, note = _consensus_for(spec, est_data)
        # defensive magnitude sanity vs our own history: a unit surprise on
        # yahoo's side must not poison the cascade
        if value is not None and blob.get("series_values"):
            from pipeline.validate import magnitude_gate
            gate = magnitude_gate(value, blob["series_values"], kind=spec.kind)
            if not gate["passed"]:
                if log:
                    log.event("consensus_rejected", ticker=ticker, metric=spec.label,
                              value=value, detail=gate["detail"])
                value, note = None, f"rejected by magnitude gate: {note}"
        blob["consensus"] = {"value": value, "source": note,
                             "fetched_at": (data or {}).get("fetched_at")}
        metrics[spec.label] = blob
    return {"metrics": metrics}


def node_reconcile(state: CompanyState) -> dict:
    ticker, log = state["ticker"], state.get("log")
    metrics = dict(state.get("metrics", {}))
    if not settings().enable_reconcile:
        return {}
    for spec in _specs(ticker):
        blob = metrics.get(spec.label, {})
        consensus = (blob.get("consensus") or {}).get("value")
        nudges = blob.get("nudges")
        ours = nudges["pre_reconcile"] if nudges else None
        if consensus is None or ours is None:
            continue
        est: Estimate | None = blob.get("estimate")
        chosen = next((c for c in blob.get("baselines", [])
                       if c["method"] == blob.get("baseline_chosen")), None)
        baseline_summary = (f"{chosen['method']}={chosen['value']:.2f} "
                            f"(backtest MAE {((chosen.get('backtest') or {}).get('mae') or 0):.2f})"
                            if chosen else "n/a")
        nudge_summary = (f"adjustment {nudges['adjustment']:+.3f} (cap {nudges['cap']:.3f}, "
                         f"{nudges['cap_reason']})")
        citations = []
        if est is not None:
            for g in (est.momentum, est.guidance_style, est.surprise_skew):
                citations += g.citations
        context = build_reconciler_context(
            spec, ours, est.rationale if est else "(estimator unavailable)",
            citations, nudge_summary, baseline_summary,
            consensus, (blob.get("consensus") or {}).get("source", "?"))
        try:
            rec, usage = _structured_retry(
                "big",
                [{"role": "system", "content": RECONCILER_SYSTEM},
                 {"role": "user", "content": context}],
                Reconciliation, log=log, stage="reconcile", metric=spec.label)
            if log:
                log.event("llm_call", stage="reconcile", ticker=ticker,
                          metric=spec.label, **usage)
            blob["reconciliation"] = rec
            blob["blend"] = reconcile_blend(ours, consensus, rec.weight_ours)
        except Exception as e:  # noqa: BLE001 — keep our blind value
            if log:
                log.event("reconcile_failed", ticker=ticker, metric=spec.label,
                          error=f"{type(e).__name__}: {e}")
            blob["reconcile_error"] = f"{type(e).__name__}: {e}"
        metrics[spec.label] = blob
    return {"metrics": metrics}


def node_finalize(state: CompanyState) -> dict:
    ticker, log = state["ticker"], state.get("log")
    metrics = dict(state.get("metrics", {}))
    for spec in _specs(ticker):
        blob = metrics.get(spec.label, {})
        nudges = blob.get("nudges")
        blend = blob.get("blend")
        chosen = next((c for c in blob.get("baselines", [])
                       if c["method"] == blob.get("baseline_chosen")), None)
        gfacts = blob.get("guidance_facts", [])
        target_g = [f for f in gfacts if f.period == spec.period]
        g_lo = next((f.value for f in target_g if f.fact_type == "guidance_low"), None)
        g_hi = next((f.value for f in target_g if f.fact_type == "guidance_high"), None)
        g_mid = next((f.value for f in target_g if f.fact_type == "guidance_mid"), None)
        if g_mid is None and g_lo is not None and g_hi is not None:
            g_mid = (g_lo + g_hi) / 2.0
        consensus = (blob.get("consensus") or {}).get("value")
        anchor_value = (blob.get("anchor") or {}).get("value")

        candidates: list[tuple[str, float | None]] = []
        if blend is not None:
            candidates.append(("reconciled", blend["final"]))
        elif nudges is not None:
            candidates.append(("estimator_nudged", nudges["pre_reconcile"]))
        if chosen is not None:
            candidates.append((f"baseline:{chosen['method']}", chosen["value"]))
        if g_mid is not None:
            candidates.append(("guidance_mid", g_mid))
        if consensus is not None:
            candidates.append(("consensus", consensus))
        if anchor_value is not None:
            candidates.append(("anchor_last_year", anchor_value))
        # Ultimate rung: the most recent reported actual of ANY period (the
        # series is period-ordered, so [-1] is newest). Covers metrics whose
        # history lacks the same-quarter anchor entirely (HD comparable sales
        # has no Q2 readings) — the never-blank rule must never hit the raise.
        sv = blob.get("series_values") or []
        if sv:
            candidates.append(("last_reported_actual", sv[-1]))

        # Additive audit field for the report's candidate ladder: EVERY canonical
        # rung in cascade order, absent ones included (value None). The cascade
        # itself still consumes `candidates` above — behavior unchanged.
        est_rung = (("reconciled", blend["final"]) if blend is not None else
                    ("estimator_nudged",
                     nudges["pre_reconcile"] if nudges is not None else None))
        blob["candidates"] = [
            {"name": n, "value": v} for n, v in (
                est_rung,
                (f"baseline:{chosen['method']}" if chosen else "baseline",
                 chosen["value"] if chosen else None),
                ("guidance_mid", g_mid),
                ("consensus", consensus),
                ("anchor_last_year", anchor_value),
                ("last_reported_actual", (blob.get("series_values") or [None])[-1]),
            )
        ]

        history = blob.get("series_values") or None
        guidance_bounds = (g_lo if g_lo is not None else g_mid,
                           g_hi if g_hi is not None else g_mid)
        try:
            final, source_used, reasons = resolve_final_value(
                candidates, spec, history=history, guidance=guidance_bounds)
        except ValueError as e:
            # zero numeric candidates: should be impossible (anchor rung), but the
            # never-blank rule gets a hard backstop anyway
            if log:
                log.event("finalize_no_candidates", ticker=ticker, metric=spec.label,
                          error=str(e))
            blob["final"] = None
            blob["final_source"] = "NONE"
            blob["cascade_reasons"] = [str(e)]
            metrics[spec.label] = blob
            continue

        gates = run_all_gates(final, spec, history=history, guidance=guidance_bounds)
        if blob.get("fact_flags"):
            gates.append({"check": "extraction_flags", "passed": True, "level": "warn",
                          "detail": "facts carried flags: " + ", ".join(blob["fact_flags"])})
        blob["final"] = final
        blob["final_source"] = source_used
        # a fallback happened when the cascade skipped a preferred candidate OR
        # the estimator itself errored (its rung never entered the cascade)
        if blob.get("estimate_error"):
            reasons = [f"estimator unavailable: {blob['estimate_error']}"] + reasons
        blob["cascade_reasons"] = reasons
        blob["gates"] = gates
        blob["fallback"] = bool(blob.get("estimate_error")) or (
            (source_used != candidates[0][0]) if candidates else False)
        if log:
            log.event("finalize", ticker=ticker, metric=spec.label, final=final,
                      source=source_used, fallback=blob["fallback"])
        metrics[spec.label] = blob

    # Implied-EPS coherence backstop (deterministic, warn-only): for companies
    # with both a per-share metric and a profit/income-like flow metric on the
    # SAME basis and period, check the final EPS against the EPS implied by the
    # final profit (validate.sibling_coherence). Appended to the EPS metric's
    # gate list so it surfaces in the report's validation chips.
    _PROFIT_WORDS = ("profit", "income", "earnings", "ebit")
    for eps_spec in [s for s in _specs(ticker) if s.kind == "per_share"]:
        mate = next(
            (s for s in _specs(ticker)
             if s.kind == "flow_absolute" and s.basis == eps_spec.basis
             and s.period == eps_spec.period
             and any(w in s.label.lower() for w in _PROFIT_WORDS)),
            None)
        if mate is None:
            continue
        eps_blob = metrics.get(eps_spec.label, {})
        mate_blob = metrics.get(mate.label, {})
        if eps_blob.get("final") is None or mate_blob.get("final") is None:
            continue  # runs only after BOTH finals exist
        check = sibling_coherence(
            eps_blob.get("final"), mate_blob.get("final"),
            (eps_blob.get("anchor") or {}).get("value"),
            (mate_blob.get("anchor") or {}).get("value"))
        eps_blob.setdefault("gates", []).append(check)
        metrics[eps_spec.label] = eps_blob
        if log:
            log.event("sibling_coherence", ticker=ticker, metric=eps_spec.label,
                      sibling=mate.label, level=check["level"], detail=check["detail"])
    return {"metrics": metrics}


def node_write(state: CompanyState) -> dict:
    ticker, log = state["ticker"], state.get("log")
    specs = _specs(ticker)
    metrics = state.get("metrics", {})
    # Kind-aware rounding: money 1dp, per-share 2dp, percent 1dp. Well inside
    # the scoring floors (0.5% of value / 0.5pp) — cosmetic only.
    def _round(v, kind):
        if v is None:
            return v
        return round(v, {"flow_absolute": 1, "per_share": 2, "ratio_pct": 1}[kind])

    values = {s.label: _round(metrics.get(s.label, {}).get("final"), s.kind) for s in specs}
    out_dir = state.get("out_dir")  # tests inject; default submission dir
    kwargs = {"out_dir": out_dir} if out_dir else {}
    path = write_workbook(specs, values, **kwargs)
    issues = verify_workbook(path, specs)
    if log:
        log.event("workbook_written", ticker=ticker, path=str(path), issues=issues)
    return {"workbook": str(path), "verify_issues": issues}


# ================================================================ graph

def build_graph():
    g = StateGraph(CompanyState)
    g.add_node("load", node_load)
    g.add_node("research", node_research)
    g.add_node("estimate", node_estimate)
    g.add_node("consensus", node_consensus)
    g.add_node("reconcile", node_reconcile)
    g.add_node("finalize", node_finalize)
    g.add_node("write", node_write)
    g.add_edge(START, "load")
    g.add_edge("load", "research")
    g.add_edge("research", "estimate")
    g.add_edge("estimate", "consensus")
    g.add_edge("consensus", "reconcile")
    g.add_edge("reconcile", "finalize")
    g.add_edge("finalize", "write")
    g.add_edge("write", END)
    return g.compile()


def run_company(ticker: str, log: RunLog | None = None, out_dir=None) -> CompanyState:
    """Run the full graph for one company. Returns the final state."""
    graph = build_graph()
    state: CompanyState = {"ticker": ticker.upper().removeprefix("LSE:"),
                           "log": log, "errors": []}
    if out_dir is not None:
        state["out_dir"] = out_dir  # type: ignore[typeddict-unknown-key]
    return graph.invoke(state)


def mermaid_diagram() -> str:
    """The architecture diagram, generated from the compiled graph itself
    (diagram accuracy by construction — used by the judged write-up)."""
    return build_graph().get_graph().draw_mermaid()
