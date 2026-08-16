"""Safety gates + failsafe cascade.

1. Unit/plausibility gates driven by MetricSpec.kind. Every gate returns a
   result dict {check, passed, level, detail} and NEVER raises — a broken gate
   must not sink a forecast at 17:55.
2. resolve_final_value: the never-blank failsafe cascade. A missing forecast
   scores worst-possible 5.0, so we always return a number if any candidate is
   numeric.

Unit converters live here too (duplicated intentionally if the data agent's
extract.py also grows converters — integration will dedupe).
"""
from __future__ import annotations

import math
import statistics
from typing import Any

from pipeline.types import MetricSpec, Unit

# ---------------------------------------------------------------- converters
# Canonical units (see pipeline.types.CONTEST_UNITS):
#   USDm / GBPm  -> millions        (43.2 bn reported => 43200.0 canonical)
#   USD / share  -> dollars/share   (as reported)
#   GBp          -> PENCE           (0.915 GBP reported => 91.5 canonical)
#   %            -> percentage pts  (0.045 decimal => 4.5 canonical)


def billions_to_millions(value: float) -> float:
    return value * 1_000.0


def pounds_to_pence(value: float) -> float:
    return value * 100.0


def decimal_to_points(value: float) -> float:
    """0.045 (decimal fraction) -> 4.5 (percentage points)."""
    return value * 100.0


#: raw_unit string -> multiplier into the canonical unit of the matching scale.
_RAW_UNIT_FACTORS: dict[str, float] = {
    # flows -> millions
    "USD_billions": 1_000.0, "GBP_billions": 1_000.0,
    "USD_millions": 1.0, "GBP_millions": 1.0,
    "USD_thousands": 1e-3, "GBP_thousands": 1e-3,
    # per-share
    "USD_per_share": 1.0,
    "GBP_per_share": 100.0,   # pounds -> pence (Hays canonical is pence)
    "GBp_per_share": 1.0,
    "pence": 1.0,
    # percentages -> points
    "pct_points": 1.0,
    "decimal_fraction": 100.0,
}


def to_canonical(value: float, raw_unit: str) -> float:
    """Convert a raw extracted value into its metric's canonical unit.
    Unknown raw_unit raises KeyError — extraction must use known raw units."""
    return value * _RAW_UNIT_FACTORS[raw_unit]


# ---------------------------------------------------------------- gate helpers

def _result(check: str, level: str, detail: str) -> dict[str, Any]:
    return {"check": check, "passed": level != "fail", "level": level, "detail": detail}


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ---------------------------------------------------------------- gates

def magnitude_gate(value: float, history_values: list[float] | None,
                   kind: str = "flow_absolute") -> dict[str, Any]:
    """Unit-error detector with a history-adaptive band.

    Flows/EPS: compare against the LAST reported value; the allowed ratio band
    widens with the volatility the series itself has shown (a series that halves
    yearly gets a band that tolerates halving; a stable one keeps 0.5x-2x).
    Sign flips vs a consistently-signed history fail. ratio_pct: within +/-15
    points of the recent median. No history => warn, not fail."""
    if not _finite(value):
        return _result("magnitude", "fail", f"value not finite: {value!r}")
    hist = [h for h in (history_values or []) if _finite(h)]
    if not hist:
        return _result("magnitude", "warn", "no history to compare against")
    if kind == "ratio_pct":
        med = statistics.median(hist[-3:])
        delta = abs(value - med)
        if delta > 15.0:
            return _result("magnitude", "fail",
                           f"{value} is {delta:.1f} pts from recent median {med} (limit 15)")
        return _result("magnitude", "ok", f"{value} within 15 pts of recent median {med}")
    last = hist[-1]
    if last == 0:
        return _result("magnitude", "warn", "last reported value is 0; cannot ratio-check")
    if value * last < 0 and all(h * last > 0 for h in hist):
        return _result("magnitude", "fail",
                       f"{value} flips sign vs a consistently {'positive' if last > 0 else 'negative'} history")
    # observed YoY ratios widen the band beyond the 0.5x-2x default
    obs = [hist[i] / hist[i - 1] for i in range(1, len(hist)) if hist[i - 1] != 0]
    lo = min([0.5] + [r * 0.7 for r in obs if r > 0])
    hi = max([2.0] + [r * 1.4 for r in obs if r > 0])
    ratio = value / last
    if not (lo <= ratio <= hi):
        return _result("magnitude", "fail",
                       f"{value} is {ratio:.3f}x the last reported value {last} "
                       f"(history-adaptive band {lo:.2f}x-{hi:.2f}x) — possible unit error")
    return _result("magnitude", "ok",
                   f"{value} is {ratio:.2f}x last reported {last} (band {lo:.2f}x-{hi:.2f}x)")


def pence_trap(value: float, spec: MetricSpec, history: list[float] | None = None) -> dict[str, Any]:
    """Hays EPS is in PENCE. Detect the pounds-passed-as-pence signature (a ~100x
    drop) — but HISTORY-AWARE: genuinely small pence values are legitimate when
    the company's own history is small (Hays FY2025 basic EPS was 1.31p)."""
    if spec.kind != "per_share" or not spec.unit.pence:
        return _result("pence_trap", "ok", "n/a (not a pence per-share metric)")
    if not _finite(value):
        return _result("pence_trap", "fail", f"value not finite: {value!r}")
    hist = [h for h in (history or []) if _finite(h)]
    if hist:
        med = statistics.median(hist)
        if med > 0 and value <= med / 25.0:
            return _result("pence_trap", "fail",
                           f"{value} pence is ~{med / value:.0f}x below the historical median "
                           f"({med:.2f}p) — pounds-vs-pence signature (did you mean {value * 100.0}?)")
        if med > 0 and value <= med / 5.0:
            return _result("pence_trap", "warn",
                           f"{value} pence is far below the historical median ({med:.2f}p) — check units")
        return _result("pence_trap", "ok", f"{value} pence plausible vs history (median {med:.2f}p)")
    if value < 5.0:  # no history to compare — keep the conservative absolute check
        return _result("pence_trap", "warn",
                       f"{value} pence is small and no history available — verify not pounds "
                       f"(pence would be {value * 100.0})")
    return _result("pence_trap", "ok", f"{value} pence plausible")


def percent_trap(value: float, spec: MetricSpec) -> dict[str, Any]:
    """Percent metrics use POINTS (4.5 == 4.5%). |v| in (0,1) smells like a
    decimal fraction (warn); |v| > 80 is a broken percentage (fail)."""
    if spec.kind != "ratio_pct":
        return _result("percent_trap", "ok", "n/a (not a percentage metric)")
    if not _finite(value):
        return _result("percent_trap", "fail", f"value not finite: {value!r}")
    if abs(value) > 80.0:
        return _result("percent_trap", "fail",
                       f"|{value}| > 80 points — not a plausible contest percentage")
    if 0.0 < abs(value) < 1.0:
        return _result("percent_trap", "warn",
                       f"{value} may be a decimal fraction — points required "
                       f"(did you mean {value * 100.0}?)")
    return _result("percent_trap", "ok", f"{value} points plausible")


def unit_consistency(record_unit: Unit | None, spec: MetricSpec) -> dict[str, Any]:
    """Canonical unit of the value's source record must equal the spec unit."""
    if record_unit is None:
        return _result("unit_consistency", "warn", "record has no unit attached")
    if record_unit != spec.unit:
        return _result("unit_consistency", "fail",
                       f"record unit {record_unit.model_dump()} != spec unit "
                       f"{spec.unit.model_dump()} ({spec.unit_str})")
    return _result("unit_consistency", "ok", f"unit matches {spec.unit_str}")


def guidance_sanity(value: float, guidance_low: float | None,
                    guidance_high: float | None) -> dict[str, Any]:
    """Flag a value more than 25% (relative) outside the guidance range."""
    if not _finite(value):
        return _result("guidance_sanity", "fail", f"value not finite: {value!r}")
    lo = guidance_low if _finite(guidance_low) else None
    hi = guidance_high if _finite(guidance_high) else None
    if lo is None and hi is None:
        return _result("guidance_sanity", "ok", "no guidance available")
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    lo = lo if lo is not None else hi
    hi = hi if hi is not None else lo
    if lo <= value <= hi:
        return _result("guidance_sanity", "ok", f"{value} within guidance [{lo}, {hi}]")
    edge = lo if value < lo else hi
    denom = max(abs(edge), 1e-9)
    rel = abs(value - edge) / denom
    if rel > 0.25:
        return _result("guidance_sanity", "fail",
                       f"{value} is {rel:.0%} outside guidance [{lo}, {hi}] (limit 25%)")
    return _result("guidance_sanity", "warn",
                   f"{value} is {rel:.0%} outside guidance [{lo}, {hi}] — tolerated (<25%)")


def run_all_gates(value: float, spec: MetricSpec,
                  history: list[float] | None = None,
                  guidance: tuple[float | None, float | None] | None = None,
                  ) -> list[dict[str, Any]]:
    """Run every value-level gate for one candidate. Never raises."""
    g_lo, g_hi = guidance if guidance else (None, None)
    try:
        return [
            magnitude_gate(value, history, kind=spec.kind),
            pence_trap(value, spec, history=history),
            percent_trap(value, spec),
            guidance_sanity(value, g_lo, g_hi),
        ]
    except Exception as exc:  # pragma: no cover — gates must never sink a run
        return [_result("gates", "warn", f"gate machinery error (ignored): {exc!r}")]


# ---------------------------------------------------------------- failsafe cascade

def resolve_final_value(
    candidates: list[tuple[str, float | None]],
    spec: MetricSpec | None = None,
    history: list[float] | None = None,
    guidance: tuple[float | None, float | None] | None = None,
) -> tuple[float, str, list[str]]:
    """The never-blank cascade. `candidates` is caller-ordered
    (estimator -> best baseline -> guidance mid -> consensus).

    - Skips None / NaN / non-numeric / hard-gate-failing values, logging why.
    - Returns the first candidate that survives all hard gates.
    - If every candidate fails gates, returns the LAST non-None numeric one
      with a loud FAILSAFE flag — a written wrong-ish number scores better
      than a blank (blank = worst-possible 5.0).
    - Raises only if there is no numeric candidate at all (caller bug: the
      cascade must always include at least consensus or guidance mid).
    """
    reasons: list[str] = []
    last_numeric: tuple[str, float] | None = None

    for name, raw in candidates:
        if raw is None:
            reasons.append(f"{name}: skipped (None)")
            continue
        if not _finite(raw):
            reasons.append(f"{name}: skipped (non-finite/non-numeric: {raw!r})")
            continue
        value = float(raw)
        last_numeric = (name, value)
        if spec is not None:
            results = run_all_gates(value, spec, history=history, guidance=guidance)
            fails = [r for r in results if not r["passed"]]
            if fails:
                details = "; ".join(f"{r['check']}: {r['detail']}" for r in fails)
                reasons.append(f"{name}: gate-failed ({details})")
                continue
        reasons.append(f"{name}: accepted")
        return value, name, reasons

    if last_numeric is not None:
        name, value = last_numeric
        reasons.append(
            f"FAILSAFE: every candidate failed gates; using last numeric candidate "
            f"{name!r}={value} — NEEDS MANUAL REVIEW before upload")
        return value, f"{name} (FAILSAFE)", reasons

    raise ValueError(
        "resolve_final_value: no numeric candidate at all — never-blank rule requires the "
        f"caller to supply at least one number. Log: {reasons}")
