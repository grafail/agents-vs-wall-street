"""Estimator-panel aggregation + failure tolerance. No network, no API key."""
import pytest

from pipeline import ensemble
from pipeline.config import Settings
from pipeline.types import CompanyEstimates, MetricEstimate, metric_specs

HD_SPECS = [s for s in metric_specs() if s.ticker == "HD"]


def me(label="Net sales", p10=0.0, p50=1.0, p90=2.0,
       momentum="warming", style="accurate", skew="balanced",
       confidence="medium", cites=("doc-a",), explanation="expl",
       method="seasonal_yoy", rationale="r") -> MetricEstimate:
    g = lambda lab: {"explanation": explanation, "label": lab,  # noqa: E731
                     "citations": list(cites)}
    return MetricEstimate(
        metric_label=label, method=method,
        growth_p10=p10, growth_p50=p50, growth_p90=p90,
        momentum=g(momentum), guidance_style=g(style), surprise_skew=g(skew),
        confidence=confidence, rationale=rationale)


def ce(*ests) -> CompanyEstimates:
    return CompanyEstimates(coherence_rationale="coherent", estimates=list(ests))


class FakeLog:
    def __init__(self):
        self.events = []

    def event(self, kind, **data):
        self.events.append({"kind": kind, **data})

    def of(self, kind):
        return [e for e in self.events if e["kind"] == kind]


# ---------------------------------------------------------------- aggregation

def test_quantile_medians_and_disagreement_widening():
    agg = ensemble.aggregate([
        ce(me(p10=-0.5, p50=0.0, p90=0.5)),
        ce(me(p10=0.5, p50=1.0, p90=1.5)),
        ce(me(p10=4.5, p50=5.0, p90=5.5)),
    ])["Net sales"]
    # p50 = median(0,1,5) = 1; d = 5-0 = 5
    assert agg.growth_p50 == 1.0
    # p10 = min(median(p10)=0.5, p50 - d/2 = -1.5) = -1.5
    assert agg.growth_p10 == -1.5
    # p90 = max(median(p90)=1.5, p50 + d/2 = 3.5) = 3.5
    assert agg.growth_p90 == 3.5


def test_no_widening_when_members_agree():
    agg = ensemble.aggregate([
        ce(me(p10=0.0, p50=1.0, p90=2.0)),
        ce(me(p10=0.2, p50=1.0, p90=1.8)),
    ])["Net sales"]
    assert agg.growth_p50 == 1.0
    assert agg.growth_p10 == pytest.approx(0.1)   # median, no widening (d=0)
    assert agg.growth_p90 == pytest.approx(1.9)


def test_majority_vote_and_explanation_and_citation_union():
    agg = ensemble.aggregate([
        ce(me(momentum="hot", cites=("a", "b"), explanation="first hot")),
        ce(me(momentum="hot", cites=("b", "c"), explanation="second hot")),
        ce(me(momentum="cooling", cites=("z",))),
    ])["Net sales"]
    assert agg.momentum.label == "hot"
    assert agg.momentum.explanation == "panel 2/3: first hot"
    assert agg.momentum.citations == ["a", "b", "c"]   # ordered union of winners only


def test_tie_breaks_to_more_neutral_label():
    m = ensemble.aggregate([ce(me(momentum="neutral")), ce(me(momentum="hot"))])
    assert m["Net sales"].momentum.label == "neutral"
    m = ensemble.aggregate([ce(me(momentum="hot")), ce(me(momentum="cooling"))])
    assert m["Net sales"].momentum.label == "cooling"
    m = ensemble.aggregate([ce(me(style="promotional")), ce(me(style="accurate"))])
    assert m["Net sales"].guidance_style.label == "accurate"
    m = ensemble.aggregate([ce(me(skew="upside")), ce(me(skew="balanced"))])
    assert m["Net sales"].surprise_skew.label == "balanced"


def test_confidence_majority_tie_goes_low():
    m = ensemble.aggregate([ce(me(confidence="high")), ce(me(confidence="medium"))])
    assert m["Net sales"].confidence == "low"
    m = ensemble.aggregate([ce(me(confidence="high")), ce(me(confidence="high")),
                            ce(me(confidence="low"))])
    assert m["Net sales"].confidence == "high"


def test_single_member_identity():
    one = me(p10=-1.0, p50=0.5, p90=2.0, momentum="cold", confidence="high")
    agg = ensemble.aggregate([ce(one)])["Net sales"]
    assert (agg.growth_p10, agg.growth_p50, agg.growth_p90) == (-1.0, 0.5, 2.0)
    assert agg.momentum.label == "cold"
    assert agg.momentum.explanation == "panel 1/1: expl"
    assert agg.confidence == "high"
    assert agg.method == "seasonal_yoy"


def test_labels_group_case_insensitively():
    agg = ensemble.aggregate([ce(me(label="Net Sales", p50=1.0)),
                              ce(me(label="net sales", p50=3.0))])
    assert list(agg) == ["Net Sales"]           # first-seen spelling wins
    assert agg["Net Sales"].growth_p50 == 2.0


# ---------------------------------------------------------------- panel run

@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(ensemble.time, "sleep", lambda s: None)


def _wire_settings(monkeypatch, **kw):
    defaults = dict(llm_provider="openrouter", openrouter_api_key="or-key",
                    openai_api_key="oa-key", model_big="big-m", estimator_panel="")
    defaults.update(kw)
    monkeypatch.setattr(ensemble, "settings", lambda: Settings(**defaults))


def test_empty_panel_uses_single_default_member(monkeypatch):
    _wire_settings(monkeypatch, estimator_panel="")
    seen = {}

    def fake_default(size, messages, schema, **kw):
        seen["size"] = size
        return ce(me()), {"model": "big-m", "prompt_tokens": 10,
                          "completion_tokens": 2, "cached_prompt_tokens": 0}

    monkeypatch.setattr(ensemble.llm, "complete_structured", fake_default)
    log = FakeLog()
    est_map, summary = ensemble.panel_estimate(HD_SPECS, lambda: [], log)
    assert seen["size"] == "big"                        # exact legacy path
    assert est_map["Net sales"].growth_p50 == 1.0
    assert summary["members_ok"] == ["openrouter:big-m"]
    calls = log.of("llm_call")
    assert len(calls) == 1 and calls[0]["stage"] == "estimate_panel"
    assert log.of("panel_aggregated")[0]["members_ok"] == 1


def test_panel_member_failure_tolerated_and_missing_key_skipped(monkeypatch, no_sleep):
    _wire_settings(monkeypatch,
                   estimator_panel="openrouter:m1,openai:m2,openrouter:m3",
                   openai_api_key="")   # openai member must be skipped (no key)
    usage = {"model": "x", "prompt_tokens": 5, "completion_tokens": 1,
             "cached_prompt_tokens": 0}

    def fake_at(provider, model, messages, schema, **kw):
        if model == "m3":
            raise RuntimeError("boom")                  # fails retry too
        return ce(me(p50=2.0)), usage

    monkeypatch.setattr(ensemble.llm, "complete_structured_at", fake_at)
    log = FakeLog()
    est_map, summary = ensemble.panel_estimate(HD_SPECS, lambda: [], log)

    assert est_map["Net sales"].growth_p50 == 2.0       # lone survivor's value
    assert summary["members_ok"] == ["openrouter:m1"]
    failed = {f["member"]: f["error"] for f in summary["members_failed"]}
    assert "missing API key for openai" in failed["openai:m2"]
    assert "boom" in failed["openrouter:m3"]
    assert len(log.of("panel_member_failed")) == 2
    assert log.of("llm_retry")                          # m3 got its one retry
    agg_ev = log.of("panel_aggregated")[0]
    assert agg_ev["members_ok"] == 1 and agg_ev["members_failed"] == 2


def test_all_members_failed_returns_none(monkeypatch, no_sleep):
    _wire_settings(monkeypatch, estimator_panel="openai:m2", openai_api_key="")
    log = FakeLog()
    est_map, summary = ensemble.panel_estimate(HD_SPECS, lambda: [], log)
    assert est_map is None
    assert summary["members_ok"] == []
    assert log.of("panel_aggregated")[0]["members_ok"] == 0


def test_two_member_aggregation_end_to_end(monkeypatch):
    _wire_settings(monkeypatch, estimator_panel="openrouter:m1,openai:m2")
    usage = {"model": "x", "prompt_tokens": 5, "completion_tokens": 1,
             "cached_prompt_tokens": 0}

    def fake_at(provider, model, messages, schema, **kw):
        p50 = 1.0 if provider == "openrouter" else 3.0
        return ce(me(p50=p50, p10=p50 - 1, p90=p50 + 1)), usage

    monkeypatch.setattr(ensemble.llm, "complete_structured_at", fake_at)
    log = FakeLog()
    est_map, summary = ensemble.panel_estimate(HD_SPECS, lambda: [], log)
    agg = est_map["Net sales"]
    assert agg.growth_p50 == 2.0                        # median of 1, 3
    assert agg.growth_p10 == 1.0                        # min(median=1, 2-1) = 1
    assert agg.growth_p90 == 3.0                        # max(median=3, 2+1) = 3
    assert summary["p50_spread"]["net sales"] == 2.0
    assert log.of("panel_aggregated")[0]["p50_spread"]["net sales"] == 2.0
