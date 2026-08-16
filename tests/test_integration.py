"""Integration tests: full graph run with mocked LLM / research / market data.
No network, no API key."""
import json

import pytest

from pipeline import graph as graph_mod
from pipeline import report as report_mod
from pipeline import run as run_mod
from pipeline.config import Settings
from pipeline.extract import ExtractedFact
from pipeline.runlog import RunLog
from pipeline.types import Estimate, Grounded, Reconciliation, SourceRef, metric_specs

HD_SPECS = [s for s in metric_specs() if s.ticker == "HD"]
SRC = SourceRef(doc_id="home-depot/filings/x.md", trust_tier=1, kind="filing")


def _fact(label, period, value, ftype="actual", flags=()):
    return ExtractedFact(
        metric_label=label, period=period, value=value,
        raw_text=str(value), raw_unit="USD_millions",
        quote=f"{label} of {value} for {period}", source=SRC,
        fact_type=ftype, flags=list(flags))


def synthetic_hd_facts() -> list[ExtractedFact]:
    """~4.5 years of quarterly history for all three HD metrics, mildly noisy,
    plus FY-period guidance (like HD's real growth-derived FY guidance) and a
    same-period H1 fact that must be filtered out (unparseable period)."""
    facts: list[ExtractedFact] = []
    sales, eps, comp = 38000.0, 3.6, 1.4
    wiggle = [0.9, 1.05, 1.1, 0.95, 1.0, 1.02, 0.97, 1.03]
    i = 0
    for year in range(2022, 2027):
        for q in range(1, 5):
            if (year, q) >= (2026, 2):
                break
            w = wiggle[i % len(wiggle)]
            facts.append(_fact("Net sales", f"FY{year}Q{q}", round(sales * w, 1)))
            facts.append(_fact("Adjusted diluted EPS", f"FY{year}Q{q}", round(eps * w, 2)))
            facts.append(_fact("Comparable sales, total company", f"FY{year}Q{q}",
                               round(comp * w - 1.0, 2)))
            sales *= 1.008
            eps *= 1.009
            i += 1
    # FY guidance (never pairs with quarterly actuals — like-for-like rule)
    facts.append(_fact("Net sales", "FY2026", 163000.0, "guidance_mid",
                       flags=["derived_from_growth"]))
    # an H1 fact the series builder must skip, not crash on
    facts.append(_fact("Net sales", "FY2025H1", 80000.0))
    return facts


def make_estimate(p50=1.0) -> Estimate:
    g = lambda label: Grounded(explanation="peers point this way", label=label,  # noqa: E731
                               citations=["home-depot/filings/x.md"])
    return Estimate(
        method="seasonal_yoy", growth_p10=p50 - 1.0, growth_p50=p50, growth_p90=p50 + 1.0,
        momentum=g("warming"), guidance_style=g("accurate"), surprise_skew=g("balanced"),
        confidence="medium", rationale="synthetic rationale")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Wire the graph for offline runs: synthetic facts, mocked LLM, mocked
    research + consensus, tmp output dirs."""
    facts = synthetic_hd_facts()
    monkeypatch.setattr(graph_mod, "load_facts", lambda t: facts)
    monkeypatch.setattr(graph_mod, "_research_digest",
                        lambda t, log: ["peer read-through: TXN beat on margin (url)"])

    calls = {"estimate": 0, "reconcile": 0}

    def fake_structured(size, messages, schema, **kw):
        usage = {"model": "mock", "prompt_tokens": 100, "completion_tokens": 20,
                 "cached_prompt_tokens": 50}
        if schema is Estimate:
            calls["estimate"] += 1
            return make_estimate(), usage
        if schema is Reconciliation:
            calls["reconcile"] += 1
            return Reconciliation(verdict="partial", weight_ours=0.6,
                                  rationale="evidence partly supports the gap"), usage
        raise AssertionError(f"unexpected schema {schema}")

    monkeypatch.setattr(graph_mod.llm, "complete_structured", fake_structured)

    def fake_market(ticker, what, **kw):
        assert what == "estimates"
        return ({"ticker": ticker, "what": what, "fetched_at": "2026-08-16T15:00:00Z",
                 "data": {"revenue_estimate": {"avg": {"0q": 45_000_000_000.0, "0y": None}},
                          "earnings_estimate": {"avg": {"0q": 4.7, "0y": None}}}},
                False)

    monkeypatch.setattr(graph_mod.tools, "get_market_data", fake_market)
    log = RunLog(run_dir=tmp_path / "run")
    return {"facts": facts, "calls": calls, "log": log, "tmp": tmp_path}


def test_full_graph_run(wired, tmp_path):
    out_dir = tmp_path / "sub"
    state = graph_mod.run_company("HD", log=wired["log"], out_dir=out_dir)

    # workbook written and clean
    assert state["workbook"].endswith("HD-FY2026Q2.xlsx")
    assert (out_dir / "HD-FY2026Q2.xlsx").exists()
    assert state["verify_issues"] == []

    # all three metrics got numeric finals
    for spec in HD_SPECS:
        blob = state["metrics"][spec.label]
        assert isinstance(blob["final"], float)
        assert blob["gates"]

    # estimator ran for each metric; reconciler only where consensus mapped
    assert wired["calls"]["estimate"] == 3
    assert wired["calls"]["reconcile"] == 2  # net sales + EPS (ratio_pct has no consensus)

    # ratio_pct metric had no consensus by design
    comp = state["metrics"]["Comparable sales, total company"]
    assert comp["consensus"]["value"] is None

    # report assembly conforms to the report models end to end
    reports = [run_mod._metric_report(s, state["metrics"][s.label]) for s in HD_SPECS]
    rr = report_mod.RunReport(meta=report_mod.RunMeta(run_id="t"), metrics=reports)
    html = report_mod.render_html(rr)
    assert "Net sales" in html and "<script" not in html

    # derivation equations: present, provenance-tagged, reconciled path ends in a blend
    net = next(r for r in reports if r.label == "Net sales")
    assert net.derivation is not None
    names = [s.name for s in net.derivation]
    assert names[0] == "anchor" and names[1] == "baseline"
    assert net.derivation[0].provenance == "data"
    assert any(s.provenance == "llm" for s in net.derivation)
    assert net.derivation[-1].name == "reconcile blend"
    assert "(LLM)" in net.derivation[-1].substituted
    assert "Derivation" in html and "prov-llm" in html


def test_estimator_failure_falls_back(wired, tmp_path, monkeypatch):
    def broken(size, messages, schema, **kw):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(graph_mod.llm, "complete_structured", broken)
    state = graph_mod.run_company("HD", log=wired["log"], out_dir=tmp_path / "sub2")

    for spec in HD_SPECS:
        blob = state["metrics"][spec.label]
        assert isinstance(blob["final"], float)          # never blank
        assert blob["final_source"].startswith("baseline:")
        assert blob["fallback"] is True
        mr = run_mod._metric_report(spec, blob)
        assert mr.fallback_used is not None
        assert mr.fallback_used.source_used.startswith("baseline:")
        # derivation in the fallback case ends with a fallback step naming the rung
        assert mr.derivation is not None
        assert mr.derivation[-1].name == "fallback"
        assert "baseline" in mr.derivation[-1].formula
        assert mr.derivation[-1].provenance == "math"


def test_reconcile_disabled_skips_consensus(wired, tmp_path, monkeypatch):
    monkeypatch.setattr(graph_mod, "settings",
                        lambda: Settings(enable_reconcile=False, enable_research=False))
    state = graph_mod.run_company("HD", log=wired["log"], out_dir=tmp_path / "sub3")

    assert wired["calls"]["reconcile"] == 0
    for spec in HD_SPECS:
        blob = state["metrics"][spec.label]
        assert blob.get("consensus") is None or blob["consensus"].get("value") is None
        assert blob.get("reconciliation") is None
        assert blob["final_source"] in ("estimator_nudged",) or "baseline" in blob["final_source"]


def test_totals_from_events(wired, tmp_path):
    graph_mod.run_company("HD", log=wired["log"], out_dir=tmp_path / "sub4")
    totals = run_mod._totals_from_events(wired["log"].dir / "events.jsonl")
    assert totals.llm_calls >= 3
    assert totals.prompt_tokens > 0
    assert totals.cached_prompt_tokens > 0


def test_mermaid_diagram_generates():
    d = graph_mod.mermaid_diagram()
    for node in ("load", "research", "estimate", "consensus", "reconcile", "finalize", "write"):
        assert node in d
