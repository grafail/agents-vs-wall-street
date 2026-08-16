"""h1_ratio baseline: annual forecast from interim (H1) actuals via stable H1/FY ratios."""
from pipeline.baselines import Series, h1_ratio, interim_map
from pipeline.types import Fact, SourceRef, metric_specs

SRC_OLD = SourceRef(doc_id="a.md", trust_tier=1, kind="filing", published="2024-08-01")
SRC_NEW = SourceRef(doc_id="b.md", trust_tier=1, kind="filing", published="2026-02-01")
SPEC = [s for s in metric_specs() if s.ticker == "HAS" and s.label == "Net fees"][0]


def _fact(period, value, src=SRC_OLD, ft="actual"):
    return Fact(metric_label="Net fees", period=period, value=value, raw_text=str(value),
                raw_unit="GBP_millions", quote="q", source=src, fact_type=ft)


def test_h1_ratio_projects_fy_from_h1():
    series = Series([_fact("FY2024", 1000.0), _fact("FY2025", 900.0)])
    interims = {"FY2024H1": 520.0, "FY2025H1": 468.0, "FY2026H1": 400.0}
    cand = h1_ratio(series, SPEC, None, interims)
    # ratios: FY2024 0.52, FY2025 0.52 -> mean 0.52; FY = 400/0.52
    assert cand["method"] == "h1_ratio"
    assert abs(cand["value"] - 400.0 / 0.52) < 1e-6


def test_h1_ratio_needs_target_h1_and_prior_ratio():
    series = Series([_fact("FY2024", 1000.0), _fact("FY2025", 900.0)])
    assert h1_ratio(series, SPEC, None, {"FY2024H1": 520.0}) is None  # no target H1
    assert h1_ratio(series, SPEC, None, {"FY2026H1": 400.0}) is None  # no prior ratio


def test_h1_ratio_skips_quarterly_and_ratio_pct():
    q_spec = [s for s in metric_specs() if s.ticker == "HD" and s.label == "Net sales"][0]
    series = Series([_fact("FY2025", 900.0)])
    assert h1_ratio(series, q_spec, None, {"FY2026H1": 1.0}) is None


def test_interim_map_prefers_restated():
    facts = [_fact("FY2025H1", 583.3, SRC_OLD), _fact("FY2025H1", 496.0, SRC_NEW)]
    assert interim_map(facts) == {"FY2025H1": 496.0}
