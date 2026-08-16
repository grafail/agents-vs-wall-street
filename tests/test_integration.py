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
from pipeline.types import (
    CompanyEstimates,
    Estimate,
    Grounded,
    MetricEstimate,
    Reconciliation,
    SourceRef,
    metric_specs,
)

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


def make_company_estimates(labels, p50=1.0) -> CompanyEstimates:
    """Panel response covering `labels` (company-level blind estimator shape)."""
    return CompanyEstimates(
        coherence_rationale="sales, EPS and comps move together",
        estimates=[MetricEstimate(**make_estimate(p50).model_dump(), metric_label=lab)
                   for lab in labels])


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
        if schema is CompanyEstimates:
            calls["estimate"] += 1
            return make_company_estimates([s.label for s in HD_SPECS]), usage
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

    # ONE company-level estimator call; reconciler only where consensus mapped
    assert wired["calls"]["estimate"] == 1
    assert wired["calls"]["reconcile"] == 2  # net sales + EPS (ratio_pct has no consensus)

    # panel audit trail recorded on every estimated metric's blob
    for spec in HD_SPECS:
        panel = state["metrics"][spec.label]["panel"]
        assert len(panel["members_ok"]) == 1 and panel["members_failed"] == []
        member = panel["members_ok"][0]
        assert panel["per_member"][member]["Net sales"]["p50"] == 1.0
        assert panel["per_member"][member]["Net sales"]["momentum"] == "warming"
        assert panel["p50_spread"]["net sales"] == 0.0

    # ratio_pct metric had no consensus by design
    comp = state["metrics"]["Comparable sales, total company"]
    assert comp["consensus"]["value"] is None

    # report assembly conforms to the report models end to end
    reports = [run_mod._metric_report(s, state["metrics"][s.label]) for s in HD_SPECS]
    rr = report_mod.RunReport(meta=report_mod.RunMeta(run_id="t"), metrics=reports)
    html = report_mod.render_html(rr)
    assert "Net sales" in html
    # self-contained: inline JS allowed, external assets are not
    assert 'src="http' not in html and "@import" not in html and "<link" not in html

    # derivation equations: present, provenance-tagged, reconciled path ends in a blend
    net = next(r for r in reports if r.label == "Net sales")
    assert net.derivation is not None
    names = [s.name for s in net.derivation]
    assert names[0] == "anchor" and names[1] == "baseline"
    assert net.derivation[0].provenance == "data"
    assert any(s.provenance == "llm" for s in net.derivation)
    assert net.derivation[-1].name == "reconcile blend"
    assert "(LLM)" in net.derivation[-1].substituted
    assert "Show full formulas" in html

    # worksheet ledger: hero lines with running totals + always-visible consensus check
    assert net.worksheet is not None
    assert net.worksheet[0].label.startswith("anchor —")
    assert net.worksheet[-1].is_final
    assert any(w.provenance == "llm" and w.label.startswith("+ judgment") for w in net.worksheet)
    assert any(w.label.startswith("blend:") for w in net.worksheet)
    assert "← FINAL" in html and "Consensus check" in html
    # ratio_pct metric has the not-available consensus line
    assert "not available" in html

    # candidate ladder: finalize records the FULL rung list (absent rungs too)
    blob = state["metrics"]["Net sales"]
    assert blob["candidates"][0]["name"] == "reconciled"
    assert {c["name"] for c in blob["candidates"]} >= {"guidance_mid", "consensus",
                                                       "anchor_last_year"}
    assert net.candidates is not None
    assert net.candidates[0].status == "chosen"
    assert all(c.status in ("chosen", "viable", "skipped", "absent") for c in net.candidates)
    assert "Cascade" in html and "CHOSEN" in html
    # pre-upload rollup + cap meter render from live pipeline output
    assert "Pre-upload checklist" in html
    assert "ws-bar" in html


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
        # ledger renders for the fallback case too
        assert mr.worksheet is not None and mr.worksheet[-1].is_final
        assert any(w.label.startswith("fallback →") for w in mr.worksheet)
        # ladder: estimator rung is absent (it errored), a baseline rung is chosen
        assert mr.candidates is not None
        assert mr.candidates[0].name == "estimator_nudged"
        assert mr.candidates[0].status == "absent"
        chosen_rung = next(c for c in mr.candidates if c.status == "chosen")
        assert chosen_rung.name.startswith("baseline:")
        html = report_mod.render_html(report_mod.RunReport(
            meta=report_mod.RunMeta(), metrics=[mr]))
        assert "fallback →" in html and "← FINAL" in html


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


def test_panel_missing_metric_cascades_and_labels_match_case_insensitively(
        wired, tmp_path, monkeypatch):
    """A metric absent from the company response falls to the cascade; label
    mapping tolerates case differences."""
    def fake_structured(size, messages, schema, **kw):
        usage = {"model": "mock", "prompt_tokens": 100, "completion_tokens": 20,
                 "cached_prompt_tokens": 50}
        if schema is CompanyEstimates:
            # lowercase EPS label (tolerant match) + comps metric missing entirely
            return make_company_estimates(["Net sales", "adjusted diluted eps"]), usage
        if schema is Reconciliation:
            return Reconciliation(verdict="hold", weight_ours=0.9, rationale="ok"), usage
        raise AssertionError(f"unexpected schema {schema}")

    monkeypatch.setattr(graph_mod.llm, "complete_structured", fake_structured)
    state = graph_mod.run_company("HD", log=wired["log"], out_dir=tmp_path / "sub5")

    # case-insensitive label mapped fine
    eps = state["metrics"]["Adjusted diluted EPS"]
    assert eps["estimate"] is not None and eps["nudges"] is not None

    # missing metric -> estimate None + cascade to baseline, still numeric
    comp = state["metrics"]["Comparable sales, total company"]
    assert comp["estimate"] is None
    assert "missing from company estimate" in comp["estimate_error"]
    assert isinstance(comp["final"], float)
    assert comp["final_source"].startswith("baseline:")
    assert comp["fallback"] is True

    # panel audit trail present even for the missing metric
    assert comp["panel"]["members_ok"]


HAS_SPECS = {s.label: s for s in metric_specs() if s.ticker == "HAS"}


def _has_finalize_state(eps_pre_reconcile: float):
    """Minimal HAS metrics blobs for node_finalize: EPS + operating profit on
    the same (pre_exceptional) basis, plus Net fees so the loop stays happy."""
    return {
        "ticker": "HAS", "log": None,
        "metrics": {
            "Pre-exceptional basic EPS": {
                "nudges": {"pre_reconcile": eps_pre_reconcile},
                "anchor": {"value": 2.0, "period": "FY2025"},
            },
            "Pre-exceptional operating profit": {
                "nudges": {"pre_reconcile": 100.0},   # flat vs anchor
                "anchor": {"value": 100.0, "period": "FY2025"},
            },
            "Net fees": {"anchor": {"value": 500.0, "period": "FY2025"}},
        },
    }


def test_sibling_coherence_gate_fires_on_incoherent_pair():
    # flat profit (100 -> 100) but collapsing EPS (2.0 -> 1.0): implied EPS is
    # 2.0, gap 50% > 35% -> warn chip on the EPS metric
    out = graph_mod.node_finalize(_has_finalize_state(eps_pre_reconcile=1.0))
    eps = out["metrics"]["Pre-exceptional basic EPS"]
    chk = next(g for g in eps["gates"] if g["check"] == "sibling_coherence")
    assert chk["level"] == "warn" and chk["passed"] is True   # warn, never fail
    assert "implied 2" in chk["detail"]
    # visible in the report's validation chips
    mr = run_mod._metric_report(HAS_SPECS["Pre-exceptional basic EPS"], eps)
    vc = next(v for v in mr.validation if v.check == "sibling_coherence")
    assert "implied" in vc.detail
    # profit metric itself carries no coherence chip (it lives on the EPS side)
    profit = out["metrics"]["Pre-exceptional operating profit"]
    assert not any(g["check"] == "sibling_coherence" for g in profit["gates"])


def test_sibling_coherence_gate_quiet_on_coherent_pair():
    out = graph_mod.node_finalize(_has_finalize_state(eps_pre_reconcile=2.05))
    eps = out["metrics"]["Pre-exceptional basic EPS"]
    chk = next(g for g in eps["gates"] if g["check"] == "sibling_coherence")
    assert chk["level"] == "ok"


def test_sibling_coherence_skipped_when_no_same_basis_pair(wired, tmp_path):
    # HD: Net sales is as_reported, EPS is adjusted -> no qualifying sibling
    state = graph_mod.run_company("HD", log=wired["log"], out_dir=tmp_path / "sub6")
    for spec in HD_SPECS:
        gates = state["metrics"][spec.label]["gates"]
        assert not any(g["check"] == "sibling_coherence" for g in gates)


def test_mermaid_diagram_generates():
    d = graph_mod.mermaid_diagram()
    for node in ("load", "research", "estimate", "consensus", "reconcile", "finalize", "write"):
        assert node in d
