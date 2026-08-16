"""Tests for pipeline/report.py — schema validation + JS-free HTML rendering."""
import json
import shutil
from pathlib import Path

import pytest

from pipeline.report import (
    MetricReport,
    RunReport,
    render_html,
    render_report,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_report.json"


@pytest.fixture
def report() -> RunReport:
    return RunReport.model_validate_json(FIXTURE.read_text())


@pytest.fixture
def rendered(tmp_path: Path) -> str:
    """Render the fixture in a tmp dir (so no artifacts land in the repo)."""
    src = tmp_path / "report.json"
    shutil.copy(FIXTURE, src)
    out = render_report(src)
    assert out == tmp_path / "report.html"
    return out.read_text()


# ---------------------------------------------------------------- models

def test_fixture_validates(report: RunReport):
    assert len(report.metrics) == 5
    assert report.meta.enable_reconcile is True
    assert report.meta.totals.cached_prompt_tokens == 361200
    # reused shared types round-trip
    hd_sales = report.metrics[0]
    assert hd_sales.estimate.momentum.label == "warming"
    assert hd_sales.reconciliation.verdict == "hold"
    # fallback + failed-validation paths are represented in the fixture
    assert any(m.fallback_used is not None for m in report.metrics)
    assert any(not v.passed for m in report.metrics for v in m.validation)


def test_unknown_keys_are_tolerated():
    data = json.loads(FIXTURE.read_text())
    data["metrics"][0]["some_future_field"] = {"x": 1}
    data["meta"]["extra"] = "ok"
    RunReport.model_validate(data)  # must not raise


# ---------------------------------------------------------------- rendering

def test_render_contains_expected_content(rendered: str):
    # metric labels
    for label in ["Net sales", "Adjusted diluted EPS", "Comparable sales, total company",
                  "Net fees", "Pre-exceptional operating profit"]:
        assert label in rendered
    # final values (formatted)
    assert "46,833" in rendered
    assert "4.75" in rendered
    assert "911.2" in rendered
    # drill-down uses <details>
    assert "<details" in rendered and "<summary" in rendered
    # a citation string survives escaping
    assert "Pro backlog strengthened through the quarter" in rendered
    # anchor quote rendered as a quote
    assert "<blockquote>" in rendered
    # fallback + failed validation surfaced
    assert "guidance_mid" in rendered
    assert "comp_vs_sales_coherence" in rendered
    # run meta
    assert "reconcile ON" in rendered
    assert "a1b2c3d4e5f6" in rendered


def test_render_is_self_contained(rendered: str):
    """Inline JS is allowed; external assets are not (must work over file://)."""
    low = rendered.lower()
    assert "<link" not in low and "@import" not in low
    assert 'src="http' not in low and "src='http" not in low
    assert "fetch(" not in low and "xmlhttprequest" not in low
    assert "onclick" not in low  # listeners are attached in the inline script


def test_render_escapes_html(tmp_path: Path):
    m = MetricReport(company="X", ticker="X", label="<b>evil</b>", unit="USDm", period="Q1")
    html = render_html(RunReport(metrics=[m]))
    assert "<b>evil</b>" not in html
    assert "&lt;b&gt;evil&lt;/b&gt;" in html


def test_minimal_report_renders(tmp_path: Path):
    """Most Optionals None / stages skipped -> render must not crash."""
    minimal = {
        "meta": {},
        "metrics": [
            {"company": "Deere", "ticker": "DE", "label": "Diluted EPS (GAAP)",
             "unit": "USD / share", "period": "FY2026Q3"}
        ],
    }
    src = tmp_path / "report.json"
    src.write_text(json.dumps(minimal))
    out = render_report(src)
    text = out.read_text()
    assert "Diluted EPS (GAAP)" in text
    assert "—" in text                      # missing fields degrade to em-dash
    # per-company file written alongside the main report
    assert (tmp_path / "report-DE.html").exists()


def test_empty_report_renders(tmp_path: Path):
    src = tmp_path / "report.json"
    src.write_text("{}")
    out = render_report(src)
    assert "No metrics in report" in out.read_text()


def test_derivation_renders():
    """Derivation equations render with provenance badges; absence never crashes."""
    from pipeline.report import (DerivationStep, MetricReport, RunMeta, RunReport,
                                 render_html)
    fixture = Path(__file__).parent / "fixtures" / "sample_report.json"
    rr = RunReport.model_validate_json(fixture.read_text())
    net = next(m for m in rr.metrics if m.label == "Net sales")
    assert net.derivation and net.derivation[0].provenance == "data"
    html = render_html(rr)
    # worksheet ledger is the hero: plain-English lines, running totals, prefixes
    assert "anchor — FY2025Q2 actual" in html
    assert "× seasonal growth (+2.70%)" in html
    assert "+ judgment (raw +631, capped at +545)" in html
    assert "blend: 60% ours / 40% consensus" in html
    assert "← FINAL" in html
    assert "[D]" in html and "[M]" in html and "[L]" in html
    assert "D data · M math · L model judgment" in html
    # fallback metric's ledger renders too
    assert "fallback → guidance midpoint (estimator failed: timeout)" in html
    # symbolic detail is demoted into a collapsed details block but still present
    assert "Show full formulas" in html
    assert "B = anchor × (1 + mean(recent YoY))" in html
    assert "F = 0.60(LLM)·47,048" in html
    # always-visible consensus check line, including the not-available case
    assert "Consensus check" in html
    assert "not available" in html

    # a metric with no derivation/worksheet renders fine
    bare = MetricReport(company="X", ticker="X", label="Y", unit="USDm", period="FY1")
    html2 = render_html(RunReport(meta=RunMeta(), metrics=[bare]))
    assert "Show full formulas" not in html2
    assert "No worksheet recorded" in html2


def test_page_is_self_explanatory():
    """Layered readability: lay-reader primer + tooltips coexist with expert detail."""
    fixture = Path(__file__).parent / "fixtures" / "sample_report.json"
    rr = RunReport.model_validate_json(fixture.read_text())
    html = render_html(rr)
    # how-to primer with plain definitions
    assert "How to read this report" in html
    assert "<dt>baseline</dt>" in html and "<dt>backtest</dt>" in html
    assert "a rehearsal on the past" in html
    # status words carry tooltips + a legend under the summary table
    assert 'title="the final number passed every sanity check"' in html
    assert "sum-legend" in html
    # per-stage plain-English helpers and jargon paraphrases
    assert "stage-help" in html
    assert "average historical miss" in html          # MAE paraphrase
    assert "Wall Street analysts" in html             # consensus paraphrase
    # metric labels get plain-words tooltips without altering the verbatim label
    assert "Sales growth in stores open for at least a year" in html
    assert "Comparable sales, total company" in html
