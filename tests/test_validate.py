"""Gates + failsafe cascade. No network, no LLM."""
import math

import pytest

from pipeline.types import CONTEST_UNITS, MetricSpec, Unit
from pipeline.validate import (
    guidance_sanity,
    magnitude_gate,
    pence_trap,
    percent_trap,
    resolve_final_value,
    run_all_gates,
    unit_consistency,
)


def make_spec(unit_str="USDm", kind="flow_absolute", **over) -> MetricSpec:
    base = dict(
        company="Home Depot", ticker="HD", corpus_dir="home-depot",
        period="FY2026Q2", output_file="HD-FY2026Q2.xlsx",
        label="Net sales", unit_str=unit_str, unit=CONTEST_UNITS[unit_str],
        kind=kind, basis="as_reported",
    )
    base.update(over)
    return MetricSpec(**base)


HAYS_EPS = make_spec(unit_str="GBp", kind="per_share", ticker="HAS",
                     company="Hays plc", corpus_dir="hays", period="FY2026",
                     output_file="HAS-FY2026.xlsx", label="Pre-exceptional basic EPS",
                     basis="pre_exceptional", period_type="fiscal_year")
PCT_SPEC = make_spec(unit_str="%", kind="ratio_pct", label="Comparable sales, total company")


# ---------------------------------------------------------------- magnitude

def test_magnitude_pass_near_median():
    r = magnitude_gate(43_200.0, [41_000.0, 43_000.0, 45_000.0], kind="flow_absolute")
    assert r["passed"] and r["level"] == "ok"


def test_magnitude_billions_vs_millions_trap():
    # 43.2 (billions left unconverted) vs history median ~43000 (millions) must fail.
    r = magnitude_gate(43.2, [42_000.0, 43_000.0, 44_000.0], kind="flow_absolute")
    assert not r["passed"]
    assert "unit" in r["detail"]


def test_magnitude_bounds():
    hist = [100.0]
    assert magnitude_gate(50.0, hist)["passed"]        # exactly 0.5x
    assert magnitude_gate(200.0, hist)["passed"]       # exactly 2x
    assert not magnitude_gate(49.9, hist)["passed"]
    assert not magnitude_gate(200.1, hist)["passed"]


def test_magnitude_no_history_warns_not_fails():
    r = magnitude_gate(123.0, [])
    assert r["passed"] and r["level"] == "warn"
    r2 = magnitude_gate(123.0, None)
    assert r2["passed"] and r2["level"] == "warn"


def test_magnitude_ratio_pct_uses_point_distance():
    hist = [3.0, 4.0, 5.0]  # median 4.0
    assert magnitude_gate(-8.0, hist, kind="ratio_pct")["passed"]      # 12 pts away
    assert not magnitude_gate(25.0, hist, kind="ratio_pct")["passed"]  # 21 pts away
    # a ratio metric near zero must not be killed by the 0.5x-2x flow rule
    assert magnitude_gate(0.2, [0.1, 0.3, 4.0], kind="ratio_pct")["passed"]


def test_magnitude_never_raises_on_garbage():
    assert not magnitude_gate(float("nan"), [1.0])["passed"]
    assert not magnitude_gate(float("inf"), [1.0])["passed"]


# ---------------------------------------------------------------- pence trap

def test_pence_trap_catches_pounds():
    # pounds passed where pence expected — the ~100x-below-history signature
    r = pence_trap(0.915, HAYS_EPS, history=[88.0, 91.5, 95.0])
    assert not r["passed"]
    assert "pounds" in r["detail"]
    # without history we can only warn, not hard-fail (Hays EPS is legitimately ~1p now)
    r2 = pence_trap(0.915, HAYS_EPS)
    assert r2["passed"] and r2["level"] == "warn"


def test_pence_trap_passes_real_pence():
    assert pence_trap(91.5, HAYS_EPS)["passed"]
    assert pence_trap(6.2, HAYS_EPS)["passed"]
    # genuinely small pence values pass when the company's own history is small
    hays_hist = [9.22, 8.59, 4.03, 1.31]
    assert pence_trap(1.1, HAYS_EPS, history=hays_hist)["passed"]
    # but a pounds-slip of that same small history still fails
    assert not pence_trap(0.011, HAYS_EPS, history=hays_hist)["passed"]


def test_pence_trap_na_for_usd_eps():
    usd_eps = make_spec(unit_str="USD / share", kind="per_share", label="Adjusted diluted EPS")
    assert pence_trap(0.915, usd_eps)["passed"]  # not a pence metric -> n/a


# ---------------------------------------------------------------- percent trap

def test_percent_trap_decimal_form_warns():
    r = percent_trap(0.045, PCT_SPEC)
    assert r["passed"] and r["level"] == "warn"
    assert "decimal" in r["detail"]


def test_percent_trap_points_pass():
    r = percent_trap(4.5, PCT_SPEC)
    assert r["passed"] and r["level"] == "ok"
    assert percent_trap(-2.5, PCT_SPEC)["passed"]


def test_percent_trap_huge_fails():
    assert not percent_trap(450.0, PCT_SPEC)["passed"]
    assert not percent_trap(-81.0, PCT_SPEC)["passed"]


def test_percent_trap_na_for_flows():
    assert percent_trap(0.5, make_spec())["passed"]


# ---------------------------------------------------------------- unit consistency

def test_unit_consistency_match():
    assert unit_consistency(Unit(currency="USD", scale="millions"), make_spec())["passed"]


def test_unit_consistency_mismatch():
    r = unit_consistency(Unit(currency="GBP", scale="millions"), make_spec())
    assert not r["passed"]


def test_unit_consistency_pence_flag_matters():
    gbp_pounds = Unit(currency="GBP", scale="per_share", pence=False)
    assert not unit_consistency(gbp_pounds, HAYS_EPS)["passed"]


def test_unit_consistency_missing_unit_warns():
    r = unit_consistency(None, make_spec())
    assert r["passed"] and r["level"] == "warn"


# ---------------------------------------------------------------- guidance

def test_guidance_inside_range():
    assert guidance_sanity(43_500.0, 43_000.0, 44_000.0)["passed"]


def test_guidance_slightly_outside_tolerated():
    r = guidance_sanity(44_500.0, 43_000.0, 44_000.0)  # ~1.1% above high
    assert r["passed"] and r["level"] == "warn"


def test_guidance_way_outside_fails():
    r = guidance_sanity(60_000.0, 43_000.0, 44_000.0)  # ~36% above high
    assert not r["passed"]


def test_guidance_absent_passes():
    assert guidance_sanity(43_500.0, None, None)["passed"]


# ---------------------------------------------------------------- run_all_gates

def test_run_all_gates_shape():
    results = run_all_gates(43_200.0, make_spec(), history=[43_000.0],
                            guidance=(42_000.0, 44_000.0))
    assert {r["check"] for r in results} == {
        "magnitude", "pence_trap", "percent_trap", "guidance_sanity"}
    for r in results:
        assert set(r) >= {"check", "passed", "detail"}
    assert all(r["passed"] for r in results)


def test_run_all_gates_never_raises():
    results = run_all_gates(float("nan"), HAYS_EPS, history=[91.5])
    assert any(not r["passed"] for r in results)


# ---------------------------------------------------------------- cascade

HIST = [41_000.0, 43_000.0, 45_000.0]


def test_cascade_first_valid_wins():
    value, source, reasons = resolve_final_value(
        [("estimator", 43_500.0), ("baseline", 42_000.0)],
        spec=make_spec(), history=HIST)
    assert value == 43_500.0
    assert source == "estimator"
    assert any("estimator: accepted" in r for r in reasons)


def test_cascade_gate_failing_estimator_falls_through():
    # estimator forgot billions->millions: 43.2 fails magnitude, baseline wins.
    value, source, reasons = resolve_final_value(
        [("estimator", 43.2), ("baseline", 42_500.0)],
        spec=make_spec(), history=HIST)
    assert value == 42_500.0
    assert source == "baseline"
    assert any("estimator: gate-failed" in r for r in reasons)


def test_cascade_skips_none_and_nan():
    value, source, reasons = resolve_final_value(
        [("estimator", None), ("baseline", float("nan")), ("guidance_mid", 43_100.0)],
        spec=make_spec(), history=HIST)
    assert value == 43_100.0
    assert source == "guidance_mid"
    assert any("estimator: skipped (None)" in r for r in reasons)
    assert any("baseline" in r and "skipped" in r for r in reasons)


def test_cascade_all_fail_still_returns_number_with_flag():
    # Never blank: everything fails gates -> last numeric candidate, loudly flagged.
    value, source, reasons = resolve_final_value(
        [("estimator", 5.0), ("consensus", 500_000.0)],
        spec=make_spec(), history=HIST)
    assert value == 500_000.0
    assert "FAILSAFE" in source
    assert any("FAILSAFE" in r for r in reasons)
    assert math.isfinite(value)


def test_cascade_no_gates_when_no_spec():
    value, source, _ = resolve_final_value([("only", 1.0)])
    assert (value, source) == (1.0, "only")


def test_cascade_all_none_raises():
    with pytest.raises(ValueError):
        resolve_final_value([("a", None), ("b", None)], spec=make_spec(), history=HIST)


def test_cascade_pence_trap_falls_through_to_consensus():
    value, source, reasons = resolve_final_value(
        [("estimator", 0.915), ("consensus", 92.0)],
        spec=HAYS_EPS, history=[89.0, 91.5, 95.0])
    assert value == 92.0
    assert source == "consensus"
    assert any("pence" in r for r in reasons)
