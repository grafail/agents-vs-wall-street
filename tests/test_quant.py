"""Tests for the quant layer: baselines, backtest, nudges. Pure offline — no
network, no LLM. All expected values are hand-computed from the fixtures."""
import pytest

from pipeline.backtest import beat_factor, best_method, walk_forward
from pipeline.baselines import (
    Series,
    growth_drift,
    guidance_mid,
    guidance_x_beat,
    period_key,
    prior_year_period,
    run_all,
    seasonal_yoy,
    yoy_sigma,
)
from pipeline.nudges import K_CAP, apply_nudges, reconcile_blend
from pipeline.types import (
    Estimate,
    Fact,
    Grounded,
    GuidanceStyle,
    MetricSpec,
    SourceRef,
    SurpriseSkew,
    Unit,
    VibeLabel,
)

# ---------------------------------------------------------------- fixtures

SRC = SourceRef(doc_id="offline-data/test/doc.md", trust_tier=1, kind="filing")


def fact(period: str, value: float, fact_type: str = "actual") -> Fact:
    return Fact(metric_label="Revenue", period=period, value=value,
                raw_text=str(value), raw_unit="USDm", quote=f"revenue was {value}",
                source=SRC, fact_type=fact_type)


def make_spec(kind: str = "flow_absolute", period: str = "FY2026Q2",
              period_type: str = "quarter") -> MetricSpec:
    return MetricSpec(
        company="TestCo", ticker="HD", corpus_dir="home-depot", period=period,
        output_file="t.xlsx", label="Revenue", unit_str="USDm",
        unit=Unit(currency="USD", scale="millions"), kind=kind,
        basis="as_reported", period_type=period_type,
    )


def grounded_vibe(label: str, citations: list[str]) -> Grounded[VibeLabel]:
    return Grounded[VibeLabel](explanation="test", label=label, citations=citations)


def make_estimate(p10: float, p50: float, p90: float, momentum: str = "neutral",
                  skew: str = "balanced", mom_cites: list[str] | None = None,
                  skew_cites: list[str] | None = None) -> Estimate:
    return Estimate(
        method="seasonal_yoy", growth_p10=p10, growth_p50=p50, growth_p90=p90,
        momentum=grounded_vibe(momentum, mom_cites or []),
        guidance_style=Grounded[GuidanceStyle](explanation="test", label="accurate", citations=[]),
        surprise_skew=Grounded[SurpriseSkew](explanation="test", label=skew,
                                             citations=skew_cites or []),
        confidence="medium", rationale="test",
    )


def seasonal_quarterly_series() -> Series:
    """Seasonal quarterly grower: per-quarter bases 100/120/110/90, +10% YoY."""
    bases = {1: 100.0, 2: 120.0, 3: 110.0, 4: 90.0}
    facts = [fact(f"FY{y}Q{q}", bases[q] * 1.10 ** (y - 2023))
             for y in (2023, 2024, 2025) for q in (1, 2, 3, 4)]
    return Series(facts)


def annual_steady_series(last_actual: float | None = None) -> Series:
    """Annual +10% grower FY2020..FY2025; optionally override the last actual."""
    vals = [100.0, 110.0, 121.0, 133.1, 146.41, 161.051]
    if last_actual is not None:
        vals[-1] = last_actual
    return Series([fact(f"FY{2020 + i}", v) for i, v in enumerate(vals)])


def annual_accelerating_series() -> Series:
    """Annual grower with linearly accelerating growth .10/.12/.14/.16/.18."""
    vals = [100.0]
    for g in (0.10, 0.12, 0.14, 0.16, 0.18):
        vals.append(vals[-1] * (1 + g))
    return Series([fact(f"FY{2020 + i}", v) for i, v in enumerate(vals)])


# ---------------------------------------------------------------- period ordering

def test_period_key_ordering():
    assert period_key("FY2025Q4") < period_key("FY2026Q1")
    assert period_key("FY2025") < period_key("FY2026")
    assert period_key("FY2026Q1") < period_key("FY2026Q2")
    assert period_key("FY2026") == (2026, 0)
    with pytest.raises(ValueError):
        period_key("2026-Q2")


def test_prior_year_period():
    assert prior_year_period("FY2026Q2") == "FY2025Q2"
    assert prior_year_period("FY2026") == "FY2025"


def test_series_sorts_shuffled_facts_and_filters_guidance():
    facts = [fact("FY2026Q1", 3), fact("FY2025Q4", 2), fact("FY2025Q1", 1),
             fact("FY2026Q2", 99, fact_type="guidance_mid")]
    s = Series(facts)
    assert s.periods == ["FY2025Q1", "FY2025Q4", "FY2026Q1"]
    assert s.value("FY2026Q2") is None  # guidance never enters the actuals series


# ---------------------------------------------------------------- baselines

def test_seasonal_yoy_quarterly():
    s = seasonal_quarterly_series()
    spec = make_spec(period="FY2026Q2")
    cand = seasonal_yoy(s, spec)
    # anchor FY2025Q2 = 120*1.21 = 145.2; recent YoY growths all 10%
    assert cand["method"] == "seasonal_yoy"
    assert cand["value"] == pytest.approx(145.2 * 1.10)
    assert cand["inputs_used"]["anchor_period"] == "FY2025Q2"


def test_seasonal_yoy_annual():
    s = annual_steady_series()
    cand = seasonal_yoy(s, make_spec(period="FY2026", period_type="fiscal_year"))
    assert cand["value"] == pytest.approx(161.051 * 1.10)


def test_seasonal_yoy_missing_anchor_returns_none():
    s = Series([fact("FY2024Q1", 100.0)])
    assert seasonal_yoy(s, make_spec(period="FY2026Q2")) is None


def test_growth_drift_linear_acceleration():
    # growths .10, .12, .14 -> slope .02, regression predicts .16 next
    s = Series([fact("FY2022", 100.0), fact("FY2023", 110.0),
                fact("FY2024", 123.2), fact("FY2025", 140.448)])
    cand = growth_drift(s, make_spec(period="FY2026", period_type="fiscal_year"))
    assert cand["inputs_used"]["growth_applied"] == pytest.approx(0.16)
    assert cand["value"] == pytest.approx(140.448 * 1.16)


def test_ratio_pct_is_level_not_growth_of_growth():
    # comp-sales style points: FY2024Q2=2.0, FY2025Q2=3.0 -> mean + trend = 3.5
    s = Series([fact("FY2023Q2", 1.0), fact("FY2024Q2", 2.0), fact("FY2025Q2", 3.0),
                fact("FY2025Q1", -1.0)])  # other quarters must not pollute
    spec = make_spec(kind="ratio_pct", period="FY2026Q2")
    cand = seasonal_yoy(s, spec)
    assert cand["value"] == pytest.approx((2.0 + 3.0) / 2 + (3.0 - 2.0))  # 3.5
    # growth-of-a-growth is forbidden: drift method refuses ratio metrics
    assert growth_drift(s, spec) is None


def test_guidance_mid_and_precedence():
    spec = make_spec(period="FY2026Q2")
    g = [fact("FY2026Q2", 98.0, "guidance_low"), fact("FY2026Q2", 102.0, "guidance_high")]
    assert guidance_mid(Series([]), spec, g)["value"] == pytest.approx(100.0)
    g.append(fact("FY2026Q2", 101.0, "guidance_mid"))  # explicit mid wins
    assert guidance_mid(Series([]), spec, g)["value"] == pytest.approx(101.0)
    assert guidance_mid(Series([]), spec, None) is None
    assert guidance_mid(Series([]), spec, [fact("FY2025Q2", 90.0, "guidance_mid")]) is None


def test_guidance_x_beat():
    spec = make_spec(period="FY2026Q2")
    g = [fact("FY2026Q2", 100.0, "guidance_mid")]
    beat = {"avg_beat_pct": 2.0, "avg_beat_points": 2.0, "beat_rate": 0.75, "n": 4}
    assert guidance_x_beat(Series([]), spec, g, beat)["value"] == pytest.approx(102.0)
    assert guidance_x_beat(Series([]), spec, g, None) is None
    # ratio_pct applies the beat additively in points
    rspec = make_spec(kind="ratio_pct", period="FY2026Q2")
    rg = [fact("FY2026Q2", 4.0, "guidance_mid")]
    rbeat = {"avg_beat_pct": 7.5, "avg_beat_points": 0.3, "beat_rate": 1.0, "n": 3}
    assert guidance_x_beat(Series([]), rspec, rg, rbeat)["value"] == pytest.approx(4.3)


def test_run_all_collects_applicable_candidates():
    s = annual_steady_series()
    spec = make_spec(period="FY2026", period_type="fiscal_year")
    cands = run_all(s, spec, guidance=None, beat=None)
    assert {c["method"] for c in cands} == {"seasonal_yoy", "growth_drift"}
    for c in cands:
        assert set(c) == {"method", "value", "inputs_used"}


# ---------------------------------------------------------------- backtest

def test_walk_forward_perfect_series_zero_error():
    s = annual_steady_series()
    r = walk_forward(s, make_spec(period="FY2026", period_type="fiscal_year"),
                     "seasonal_yoy", n=3)
    assert r["n"] == 3
    assert r["mae"] == pytest.approx(0.0, abs=1e-9)
    assert r["bias"] == pytest.approx(0.0, abs=1e-9)
    assert r["mae_pct"] == pytest.approx(0.0, abs=1e-9)


def test_walk_forward_known_error():
    # FY2025 actual forced to 150; seasonal_yoy would predict 161.051
    s = annual_steady_series(last_actual=150.0)
    r = walk_forward(s, make_spec(period="FY2026", period_type="fiscal_year"),
                     "seasonal_yoy", n=2)
    err = 161.051 - 150.0  # FY2024 evaluates exactly, FY2025 misses by +11.051
    assert r["n"] == 2
    assert r["mae"] == pytest.approx(err / 2)
    assert r["bias"] == pytest.approx(err / 2)          # signed: we over-predicted
    assert r["mae_pct"] == pytest.approx((err / 150.0 * 100) / 2)


def test_walk_forward_ratio_uses_points_no_pct():
    s = Series([fact(f"FY{y}Q2", v, ) for y, v in
                [(2021, 1.0), (2022, 2.0), (2023, 3.0), (2024, 4.0), (2025, 6.0)]])
    r = walk_forward(s, make_spec(kind="ratio_pct", period="FY2026Q2"), "seasonal_yoy", n=2)
    # FY2024: pred (2+3)/2+1 = 4.5 vs 4.0 -> |0.5| pts; FY2025: pred (3+4)/2+1 = 4.5 vs 6.0 -> 1.5 pts
    assert r["mae"] == pytest.approx(1.0)
    assert r["mae_pct"] is None


def test_walk_forward_inapplicable_returns_none():
    s = annual_steady_series()
    spec = make_spec(period="FY2026", period_type="fiscal_year")
    assert walk_forward(s, spec, "guidance_mid", n=4, guidance=None) is None


def test_beat_factor_arithmetic():
    g = [fact(f"FY2025Q{q}", 100.0, "guidance_mid") for q in (1, 2, 3, 4)]
    a = [fact("FY2025Q1", 102.0), fact("FY2025Q2", 101.0),
         fact("FY2025Q3", 103.0), fact("FY2025Q4", 99.0)]
    b = beat_factor(g, a)
    assert b["n"] == 4
    assert b["avg_beat_pct"] == pytest.approx((2 + 1 + 3 - 1) / 4)   # +1.25%
    assert b["avg_beat_points"] == pytest.approx(1.25)
    assert b["beat_rate"] == pytest.approx(3 / 4)


def test_beat_factor_needs_three_pairs():
    g = [fact("FY2025Q1", 100.0, "guidance_mid"), fact("FY2025Q2", 100.0, "guidance_mid")]
    a = [fact("FY2025Q1", 101.0), fact("FY2025Q2", 102.0)]
    assert beat_factor(g, a) is None


def test_best_method_ranks_by_mae():
    # accelerating growth: drift extrapolates exactly, seasonal (mean of last 2) lags
    s = annual_accelerating_series()
    spec = make_spec(period="FY2026", period_type="fiscal_year")
    ranked = best_method(s, spec, n=2)
    assert [r["method"] for r in ranked][:2] == ["growth_drift", "seasonal_yoy"]
    assert ranked[0]["mae"] == pytest.approx(0.0, abs=1e-6)
    assert ranked[0]["mae"] <= ranked[1]["mae"]


# ---------------------------------------------------------------- nudges

def test_huge_p50_capped_at_k_x_mae():
    spec = make_spec()
    est = make_estimate(49.0, 50.0, 51.0)   # absurd +50% with tight spread
    audit = apply_nudges(1000.0, est, backtest_mae=10.0, sigma=4.0, spec=spec)
    assert audit["cap"] == pytest.approx(K_CAP * 10.0)      # 7.5 canonical units
    assert audit["adjustment"] == pytest.approx(7.5)
    assert audit["pre_reconcile"] == pytest.approx(1007.5)
    assert audit["cap_reason"] == "capped_at_k_x_mae"
    assert audit["raw_adjustment"] > audit["cap"]           # audit shows what was capped


def test_shrink_formula_half_at_spread_equal_sigma():
    # spread == sigma -> shrink = 1/(1+1) = 0.5; p50=2% -> 1% of baseline = 10
    spec = make_spec()
    est = make_estimate(-2.0, 2.0, 2.0)
    audit = apply_nudges(1000.0, est, backtest_mae=100.0, sigma=4.0, spec=spec)
    assert audit["quantiles"]["shrink"] == pytest.approx(0.5)
    assert audit["adjustment"] == pytest.approx(10.0)
    assert audit["cap_reason"] == "within_cap"


def test_empty_citation_categorical_is_zero():
    spec = make_spec()
    est = make_estimate(0.0, 0.0, 0.0, momentum="hot", mom_cites=[])  # uncited "hot"
    audit = apply_nudges(1000.0, est, backtest_mae=100.0, sigma=4.0, spec=spec)
    assert audit["momentum"]["value"] == 0.0
    assert audit["momentum"]["source"] == "empty_citations_zeroed"
    assert audit["adjustment"] == pytest.approx(0.0)


def test_cited_categoricals_use_sigma_fallback():
    spec = make_spec()
    est = make_estimate(0.0, 0.0, 0.0, momentum="warming", skew="upside",
                        mom_cites=["doc-a"], skew_cites=["doc-b"])
    audit = apply_nudges(1000.0, est, backtest_mae=100.0, sigma=4.0, spec=spec)
    assert audit["momentum"]["value"] == pytest.approx(0.25 * 4.0)   # +1.0 pct
    assert audit["surprise_skew"]["value"] == pytest.approx(0.15 * 4.0)
    assert audit["adjustment"] == pytest.approx(1000.0 * (1.0 + 0.6) / 100.0)


def test_calibration_table_overrides_fixed_mapping():
    spec = make_spec()
    est = make_estimate(0.0, 0.0, 0.0, momentum="hot", mom_cites=["doc"])
    audit = apply_nudges(1000.0, est, backtest_mae=100.0, sigma=4.0, spec=spec,
                         calibration={"hot": 1.1})
    assert audit["momentum"]["value"] == pytest.approx(1.1)  # not 0.5*sigma=2.0
    assert audit["momentum"]["source"] == "calibration_table"


def test_no_backtest_mae_disables_nudge():
    est = make_estimate(1.0, 2.0, 3.0, momentum="hot", mom_cites=["doc"])
    audit = apply_nudges(1000.0, est, backtest_mae=None, sigma=4.0, spec=make_spec())
    assert audit["adjustment"] == 0.0
    assert audit["cap_reason"] == "no_backtest_mae_nudge_disabled"
    assert audit["pre_reconcile"] == pytest.approx(1000.0)


def test_nudge_ratio_pct_additive_points():
    spec = make_spec(kind="ratio_pct")
    est = make_estimate(0.1, 0.2, 0.3, momentum="warming", skew="upside",
                        mom_cites=["d"], skew_cites=["d"])
    audit = apply_nudges(4.0, est, backtest_mae=1.0, sigma=0.5, spec=spec)
    shrink = 1.0 / (1.0 + 0.2 / 0.5)
    total = 0.2 * shrink + 0.25 * 0.5 + 0.15 * 0.5
    assert audit["delta_units"] == "points"
    assert audit["adjustment"] == pytest.approx(total)       # additive, NOT % of 4.0
    assert audit["pre_reconcile"] == pytest.approx(4.0 + total)


def test_reconcile_blend():
    r = reconcile_blend(100.0, 110.0, 0.6)
    assert r["final"] == pytest.approx(0.6 * 100 + 0.4 * 110)  # 104
    r2 = reconcile_blend(100.0, None, 0.3)
    assert r2["final"] == 100.0 and r2["weight_ours"] == 1.0


# ---------------------------------------------------------------- sigma helper

def test_yoy_sigma_units():
    s = Series([fact("FY2022", 100.0), fact("FY2023", 110.0),
                fact("FY2024", 123.2), fact("FY2025", 140.448)])
    sig = yoy_sigma(s, make_spec(period="FY2026", period_type="fiscal_year"))
    # growths 10%,12%,14% in pct -> population std = sqrt(8/3)
    assert sig == pytest.approx((8 / 3) ** 0.5)
    rs = Series([fact("FY2024Q2", 2.0), fact("FY2025Q2", 3.0)])
    assert yoy_sigma(rs, make_spec(kind="ratio_pct")) is None  # <2 observations
