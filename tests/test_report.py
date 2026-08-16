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
    assert "46,580" in rendered
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


def test_render_is_js_free(rendered: str):
    assert "<script" not in rendered.lower()
    assert "onclick" not in rendered.lower()


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
    assert "<script" not in text.lower()


def test_empty_report_renders(tmp_path: Path):
    src = tmp_path / "report.json"
    src.write_text("{}")
    out = render_report(src)
    assert "No metrics in report" in out.read_text()
