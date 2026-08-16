"""Pipeline entrypoints.

Prepare (daytime, re-runnable — warms caches & commits facts artifacts):
    uv run python -m pipeline.run --prepare --all [--refresh]

Forecast (THE final command — produces the 4 workbooks + run report + log):
    uv run python -m pipeline.run --forecast --all
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline import report as report_mod
from pipeline.config import FACTS_DIR, ROOT, settings
from pipeline.extract import extract_company
from pipeline.graph import run_company
from pipeline.runlog import RunLog
from pipeline.types import Estimate, MetricSpec, Reconciliation, metric_specs
from pipeline.writer import run_submission_check

TICKERS = ["HD", "ADI", "HAS", "DE"]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- prepare

def cmd_prepare(tickers: list[str], refresh: bool) -> int:
    log = RunLog()
    log.event("prepare_start", tickers=tickers, refresh=refresh)
    rc = 0
    for t in tickers:
        path = FACTS_DIR / f"{t}.json"
        if path.exists() and not refresh:
            print(f"[prepare] {t}: facts exist ({path}) — skipping (use --refresh to redo)")
            continue
        try:
            out = extract_company(t, log=log)
            print(f"[prepare] {t}: facts -> {out}")
        except Exception as e:  # noqa: BLE001 — other companies must continue
            print(f"[prepare] {t}: FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            log.event("prepare_failed", ticker=t, error=str(e))
            rc = 1
    if settings().enable_research:
        # warm the research cache too so the final run is mostly cache hits
        from pipeline.graph import _research_digest
        for t in tickers:
            try:
                bullets = _research_digest(t, log)
                print(f"[prepare] {t}: research digest warmed ({len(bullets)} bullets)")
            except Exception as e:  # noqa: BLE001 — optional enrichment
                print(f"[prepare] {t}: research warm failed ({type(e).__name__}) — tolerated")
                log.event("prepare_research_failed", ticker=t, error=str(e))
    print(f"[prepare] log: {log.dir}")
    return rc


# ---------------------------------------------------------------- report assembly

def _nudge_audit_to_report(audit: dict | None) -> report_mod.NudgeAudit | None:
    if not audit:
        return None
    q = audit.get("quantiles", {})
    comps = [
        report_mod.NudgeComponent(
            name="quantile_p50", label=None, raw=q.get("p50"),
            applied=q.get("component"), capped=False,
            note=f"shrink {q.get('shrink'):.3f} (spread {q.get('spread')})"
            if q.get("shrink") is not None else None),
    ]
    for name in ("momentum", "surprise_skew"):
        c = audit.get(name) or {}
        comps.append(report_mod.NudgeComponent(
            name=name, label=c.get("label"), raw=c.get("value"),
            applied=c.get("value"), capped=False, note=c.get("source")))
    capped = audit.get("cap_reason") == "capped_at_k_x_mae"
    caps = [audit["cap_reason"]] if audit.get("cap_reason") not in (None, "within_cap") else []
    return report_mod.NudgeAudit(
        components=comps, cap=audit.get("cap"), caps_applied=caps,
        pre_reconcile_value=audit.get("pre_reconcile"))


def _fmt(v: float | None) -> str:
    return report_mod._num(v)


def _baseline_substituted(method: str, iu: dict, value: float | None) -> str:
    """Method-specific substituted formula from the baseline's inputs_used."""
    if method in ("seasonal_yoy", "growth_drift") and "growth_applied" in iu:
        g = iu["growth_applied"] * 100.0
        return f"B = {_fmt(iu.get('anchor'))} × (1 {g:+.2f}%) = {_fmt(value)}"
    if method == "seasonal_yoy" and "level_readings" in iu:
        lv = ", ".join(iu.get("level_readings", []))
        return f"B = mean({lv}) + trend {iu.get('trend_points', 0):+.2f} = {_fmt(value)}"
    if method == "guidance_mid":
        return f"B = guidance mid = {_fmt(value)}"
    if method == "guidance_x_beat":
        if "avg_beat_pct" in iu:
            return f"B = G_mid × (1 {iu['avg_beat_pct']:+.2f}%) = {_fmt(value)}"
        return f"B = G_mid {iu.get('avg_beat_points', 0):+.2f} pts = {_fmt(value)}"
    return f"B = {_fmt(value)}"


_BASELINE_FORMULA = {
    "seasonal_yoy": "B = anchor × (1 + mean(recent YoY))",
    "growth_drift": "B = anchor × (1 + drift-adjusted YoY)",
    "guidance_mid": "B = (guidance low + high) / 2",
    "guidance_x_beat": "B = guidance mid × (1 + historical beat)",
}


def _derivation_steps(spec: MetricSpec, blob: dict) -> list[report_mod.DerivationStep] | None:
    """Deterministic derivation-equation assembly from the audits the graph
    already produced. Provenance: data = extracted/fetched, math = deterministic
    computation, llm = model judgment (always entering through a computed cap)."""
    D = report_mod.DerivationStep
    steps: list[report_mod.DerivationStep] = []
    u = "pts" if spec.kind == "ratio_pct" else "%"

    anchor = blob.get("anchor") or {}
    if anchor.get("value") is not None:
        refs = []
        if anchor.get("source_doc"):
            refs.append(f"{anchor['source_doc']} (tier 1)")
        steps.append(D(name="anchor", formula=f"A = actual[{anchor.get('period')}]",
                       substituted=f"A = {_fmt(anchor['value'])} {spec.unit_str}",
                       result=anchor["value"], provenance="data", refs=refs,
                       note="same period last fiscal year, exact reported figure"))

    chosen = next((c for c in blob.get("baselines", [])
                   if c["method"] == blob.get("baseline_chosen")), None)
    if chosen is not None:
        bt = chosen.get("backtest") or {}
        refs = []
        if bt:
            refs.append(f"chosen by walk-forward backtest: MAE {_fmt(bt.get('mae'))} "
                        f"over n={bt.get('n')} periods")
        steps.append(D(name="baseline", provenance="math",
                       formula=_BASELINE_FORMULA.get(chosen["method"], f"B = {chosen['method']}"),
                       substituted=_baseline_substituted(
                           chosen["method"], chosen.get("inputs_used") or {}, chosen.get("value")),
                       result=chosen.get("value"), refs=refs))

    nudges = blob.get("nudges")
    est = blob.get("estimate")
    if nudges is not None:
        q = nudges.get("quantiles", {})
        steps.append(D(name="judgment: quantile", provenance="llm",
                       formula="Δq = p50 × shrink",
                       substituted=f"Δq = {q.get('p50', 0):+.2f}{u}(LLM) × {q.get('shrink', 1):.3f} "
                                   f"= {q.get('component', 0):+.3f}{u}",
                       result=q.get("component"),
                       note=f"shrink = 1/(1+(p90−p10)/σ), σ={_fmt(nudges.get('sigma'))} — "
                            "wide model uncertainty shrinks its own influence"))
        for comp_name, key in (("judgment: momentum", "momentum"),
                               ("judgment: surprise skew", "surprise_skew")):
            c = nudges.get(key) or {}
            cites = []
            if est is not None:
                g = getattr(est, key, None)
                if g is not None:
                    cites = list(g.citations)
            steps.append(D(name=comp_name, provenance="llm",
                           formula=f"Δ = map({key} label) × σ",
                           substituted=f"Δ({c.get('label', '—')}(LLM)) = {c.get('value', 0):+.3f}{u}",
                           result=c.get("value"), refs=cites,
                           note={"empty_citations_zeroed": "no citations → zeroed",
                                 "calibration_table": "empirically calibrated mapping",
                                 "fixed_fallback": "fixed σ-multiple mapping"}.get(c.get("source"))))
        adj_formula = ("adj = clamp(Δq+Δm+Δs, ±0.75×MAE)" if spec.kind == "ratio_pct"
                       else "adj = clamp(B × (Δq+Δm+Δs)/100, ±0.75×MAE)")
        steps.append(D(name="cap & apply", provenance="math", formula=adj_formula,
                       substituted=f"adj = clamp({_fmt(nudges.get('raw_adjustment'))}, "
                                   f"±{_fmt(nudges.get('cap'))}) = {_fmt(nudges.get('adjustment'))}",
                       result=nudges.get("adjustment"), note=nudges.get("cap_reason")))
        steps.append(D(name="pre-reconcile", provenance="math", formula="V = B + adj",
                       substituted=f"V = {_fmt(nudges.get('baseline_value'))} "
                                   f"{(nudges.get('adjustment') or 0):+,.2f} "
                                   f"= {_fmt(nudges.get('pre_reconcile'))}",
                       result=nudges.get("pre_reconcile")))

    final_source = blob.get("final_source")
    blend = blob.get("blend")
    rec = blob.get("reconciliation")
    cons = blob.get("consensus") or {}
    if final_source == "reconciled" and blend is not None:
        w = blend.get("weight_ours", 1.0)
        refs = [f"consensus: {cons.get('source')}"] if cons.get("source") else []
        steps.append(D(name="reconcile blend", provenance="math",
                       formula="F = w·V + (1−w)·C",
                       substituted=f"F = {w:.2f}(LLM)·{_fmt(blend.get('ours'))} + "
                                   f"{1 - w:.2f}·{_fmt(blend.get('consensus'))} "
                                   f"= {_fmt(blend.get('final'))}",
                       result=blend.get("final"), refs=refs,
                       note=f"reconciler verdict: {rec.verdict} (LLM)" if rec else None))
    elif final_source == "estimator_nudged":
        steps.append(D(name="final", provenance="math", formula="F = V",
                       substituted=f"F = {_fmt(blob.get('final'))}",
                       result=blob.get("final"),
                       note="no consensus blend (pure-blind mode or no consensus)"))
    elif final_source is not None:
        prov = "math" if str(final_source).startswith("baseline") else "data"
        steps.append(D(name="fallback", provenance=prov,
                       formula=f"F = {final_source}",
                       substituted=f"F = {_fmt(blob.get('final'))}",
                       result=blob.get("final"),
                       refs=blob.get("cascade_reasons", [])[:4],
                       note="failsafe cascade selected this rung"))
    return steps or None


def _metric_report(spec: MetricSpec, blob: dict) -> report_mod.MetricReport:
    anchor = blob.get("anchor") or {}
    baselines = [
        report_mod.BaselineCandidate(
            method=c["method"], value=c.get("value"),
            backtest_mae=(c.get("backtest") or {}).get("mae"),
            backtest_bias=(c.get("backtest") or {}).get("bias"),
            backtest_n=(c.get("backtest") or {}).get("n"))
        for c in blob.get("baselines", [])
    ]
    gfacts = blob.get("guidance_facts", [])
    target_g = [f for f in gfacts if f.period == spec.period]
    beat = blob.get("beat") or {}
    g_quote = target_g[0].quote if target_g else None
    g_src = target_g[0].source.doc_id if target_g else None
    guidance = report_mod.GuidanceBlock(
        low=next((f.value for f in target_g if f.fact_type == "guidance_low"), None),
        mid=next((f.value for f in target_g if f.fact_type == "guidance_mid"), None),
        high=next((f.value for f in target_g if f.fact_type == "guidance_high"), None),
        beat_factor=beat.get("avg_beat_pct"),
        quote=g_quote, source_doc=g_src,
    ) if (target_g or beat) else None

    cons = blob.get("consensus") or {}
    consensus = report_mod.ConsensusBlock(
        value=cons.get("value"), source=cons.get("source"),
    ) if cons else None

    est: Estimate | None = blob.get("estimate")
    rec: Reconciliation | None = blob.get("reconciliation")

    validation = [
        report_mod.ValidationCheck(check=g["check"], passed=g["passed"], detail=g.get("detail"))
        for g in blob.get("gates", [])
    ]
    fallback = None
    if blob.get("fallback"):
        fallback = report_mod.FallbackUsed(
            source_used=blob.get("final_source", "?"),
            reasons=blob.get("cascade_reasons", []))

    return report_mod.MetricReport(
        company=spec.company, ticker=spec.ticker, label=spec.label,
        unit=spec.unit_str, period=spec.period,
        anchor=report_mod.Anchor(
            value=anchor.get("value"), period=anchor.get("period"),
            quote=anchor.get("quote"), source_doc=anchor.get("source_doc")),
        baseline_candidates=baselines,
        baseline_chosen=blob.get("baseline_chosen"),
        guidance=guidance,
        estimate=est,
        nudges=_nudge_audit_to_report(blob.get("nudges")),
        consensus=consensus,
        reconciliation=rec,
        validation=validation,
        fallback_used=fallback,
        derivation=_derivation_steps(spec, blob),
        final_value=blob.get("final"),
    )


def _totals_from_events(events_path: Path) -> report_mod.RunTotals:
    llm_calls = prompt = completion = cached = hits = live = 0
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") == "llm_call":
                llm_calls += 1
                prompt += ev.get("prompt_tokens") or 0
                completion += ev.get("completion_tokens") or 0
                cached += ev.get("cached_prompt_tokens") or 0
            elif ev.get("kind") == "tool_call":
                if ev.get("cache_hit"):
                    hits += 1
                else:
                    live += 1
    return report_mod.RunTotals(
        llm_calls=llm_calls, prompt_tokens=prompt, completion_tokens=completion,
        cached_prompt_tokens=cached, tool_cache_hits=hits, tool_cache_live=live,
        est_cost_usd=None)


# ---------------------------------------------------------------- forecast

def cmd_forecast(tickers: list[str], out_dir: Path | None = None,
                 log: RunLog | None = None) -> int:
    s = settings()
    log = log or RunLog()
    started = datetime.now(timezone.utc)
    log.event("forecast_start", tickers=tickers,
              enable_reconcile=s.enable_reconcile, enable_research=s.enable_research,
              model_big=s.model_big, model_small=s.model_small, provider=s.llm_provider)

    metric_reports: list[report_mod.MetricReport] = []
    written: dict[str, str] = {}
    issues_all: dict[str, list[str]] = {}
    rc = 0
    for t in tickers:
        try:
            state = run_company(t, log=log, out_dir=out_dir)
        except Exception as e:  # noqa: BLE001 — one company must not sink the rest
            print(f"[forecast] {t}: FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            log.event("company_failed", ticker=t, error=f"{type(e).__name__}: {e}")
            rc = 1
            continue
        specs = [sp for sp in metric_specs() if sp.ticker == t]
        for sp in specs:
            metric_reports.append(_metric_report(sp, state.get("metrics", {}).get(sp.label, {})))
        if state.get("workbook"):
            written[t] = state["workbook"]
        issues_all[t] = state.get("verify_issues", [])

    run_report = report_mod.RunReport(
        meta=report_mod.RunMeta(
            run_id=log.dir.name, started=started,
            finished=datetime.now(timezone.utc),
            enable_reconcile=s.enable_reconcile, llm_provider=s.llm_provider,
            model_big=s.model_big, model_small=s.model_small,
            git_commit=_git_commit(),
            totals=_totals_from_events(log.dir / "events.jsonl")),
        metrics=metric_reports)
    report_path = log.save_report(json.loads(run_report.model_dump_json()))
    html_path = report_mod.render_report(report_path)

    ok, output = run_submission_check()
    log.event("submission_check", ok=ok, output=output[-2000:])

    # ---------------- final checklist ----------------
    print("\n========== RUN CHECKLIST ==========")
    print(f"mode: reconcile={'ON' if s.enable_reconcile else 'OFF (pure blind)'} "
          f"research={'ON' if s.enable_research else 'OFF'} provider={s.llm_provider}")
    for t in tickers:
        mark = "✓" if t in written else "✗"
        print(f" {mark} {t}: {written.get(t, 'NO WORKBOOK WRITTEN')}")
        for issue in issues_all.get(t, []):
            print(f"    ! {issue}")
        for m in metric_reports:
            if m.ticker != t:
                continue
            fb = f"  [FALLBACK: {m.fallback_used.source_used}]" if m.fallback_used else ""
            fails = sum(1 for v in m.validation if not v.passed)
            warn = f"  ({fails} gate fail)" if fails else ""
            print(f"    - {m.label}: {m.final_value} {m.unit}{fb}{warn}")
    print(f" report: {report_path}")
    print(f" report html: {html_path}")
    print(f" events log: {log.dir / 'events.jsonl'}")
    print(f" commit: {_git_commit() or 'unknown'}")
    if ok is None:
        print(f" npm check:submission: SKIPPED ({output})")
    else:
        print(f" npm check:submission: {'PASS' if ok else 'FAIL — see log'}")
    print("===================================\n")
    return rc


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pipeline.run",
        description="Agents vs Wall Street forecasting pipeline")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true",
                      help="extract corpus facts + warm research caches")
    mode.add_argument("--forecast", action="store_true",
                      help="run the full graph: estimates -> workbooks + run report")
    ap.add_argument("--company", help="single ticker: HD | ADI | HAS | DE")
    ap.add_argument("--all", action="store_true", help="all four companies (default)")
    ap.add_argument("--refresh", action="store_true",
                    help="prepare: re-extract even if facts exist")
    args = ap.parse_args(argv)

    tickers = ([args.company.upper().removeprefix("LSE:")] if args.company else TICKERS)
    if args.prepare:
        return cmd_prepare(tickers, refresh=args.refresh)
    return cmd_forecast(tickers)


if __name__ == "__main__":
    raise SystemExit(main())
