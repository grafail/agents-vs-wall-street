"""Walk-forward backtesting + guidance-beat calibration (quant layer, no LLM).

Provides the empirical error scale that everything downstream leans on:
- walk_forward: honest out-of-sample error per baseline method (fit strictly
  on data before each evaluated period).
- beat_factor: how much the company historically beats its own guidance
  midpoint (feeds baselines.guidance_x_beat).
- best_method: ranked methods by walk-forward MAE — the winner's MAE is the
  cap scale for nudges (see pipeline.nudges.K_CAP).
"""
from __future__ import annotations

import statistics
from typing import Callable

from pipeline.baselines import METHODS, Series, _guidance_mid_value, period_key
from pipeline.types import Fact, MetricSpec


def walk_forward(series: Series, spec: MetricSpec, method: str | Callable, n: int = 6,
                 guidance: list[Fact] | None = None, beat: dict | None = None) -> dict | None:
    """Walk-forward evaluation of one baseline method over the last `n` periods.

    For each evaluated period: truncate the series strictly before it, predict
    with the method, compare to the actual. Returns
      {"method", "mae", "mae_pct", "bias", "n"}
    - mae: mean |error| in canonical units (for ratio_pct these ARE absolute points)
    - mae_pct: mean |error / actual| * 100 for flow/per_share; None for ratio_pct
    - bias: mean signed (predicted - actual) in canonical units
    Periods where the method is inapplicable (e.g. no guidance, not enough
    history) are skipped. Returns None if nothing could be evaluated.
    """
    if callable(method):
        fn, name = method, getattr(method, "__name__", str(method))
    else:
        name, fn = method, METHODS[method]

    errors: list[float] = []
    pct_errors: list[float] = []
    for period, actual in series.points[-n:]:
        truncated = series.truncate_before(period)
        if len(truncated) == 0:
            continue
        spec_p = spec.model_copy(update={"period": period})
        if name == "guidance_x_beat":
            cand = fn(truncated, spec_p, guidance, beat)
        else:
            cand = fn(truncated, spec_p, guidance)
        if cand is None:
            continue
        err = cand["value"] - actual
        errors.append(err)
        if spec.kind != "ratio_pct" and actual != 0:
            pct_errors.append(abs(err / actual) * 100.0)

    if not errors:
        return None
    return {
        "method": name,
        "mae": statistics.fmean(abs(e) for e in errors),
        "mae_pct": statistics.fmean(pct_errors) if pct_errors else None,
        "bias": statistics.fmean(errors),
        "n": len(errors),
    }


def beat_factor(guidance_history: list[Fact], actuals: list[Fact]) -> dict | None:
    """Calibrated guidance-beat factor from historical (guidance mid, actual) pairs.

    Pairs guidance midpoints (explicit guidance_mid facts, else (low+high)/2)
    with actuals by period. Returns
      {"avg_beat_pct", "avg_beat_points", "beat_rate", "n"}
    e.g. beat 7/8 quarters by avg +1.2% -> {"avg_beat_pct": 1.2, "beat_rate": 0.875, "n": 8}.
    avg_beat_points is the additive mean (actual - mid) for ratio_pct metrics.
    Returns None when fewer than 3 matched pairs exist (not calibratable).
    """
    actual_by_period = {f.period: f.value for f in actuals if f.fact_type == "actual"}

    def _parseable(p: str) -> bool:
        try:
            period_key(p)
            return True
        except ValueError:  # half-year guidance like FY2025H1 — no matching series period
            return False

    periods = sorted({f.period for f in guidance_history if _parseable(f.period)}, key=period_key)

    beat_pcts: list[float] = []
    beat_points: list[float] = []
    beats = 0
    for p in periods:
        mid, _ = _guidance_mid_value(guidance_history, p)
        actual = actual_by_period.get(p)
        if mid is None or actual is None:
            continue
        beat_points.append(actual - mid)
        beat_pcts.append((actual - mid) / mid * 100.0 if mid != 0 else 0.0)
        if actual > mid:
            beats += 1

    n = len(beat_points)
    if n < 3:
        return None
    return {
        "avg_beat_pct": statistics.fmean(beat_pcts),
        "avg_beat_points": statistics.fmean(beat_points),
        "beat_rate": beats / n,
        "n": n,
    }


def best_method(series: Series, spec: MetricSpec, guidance: list[Fact] | None = None,
                beat: dict | None = None, n: int = 6) -> list[dict]:
    """Walk-forward every applicable method and rank by MAE (ascending).

    Returns the ranked list of walk_forward result dicts; [0] is the winner
    whose MAE becomes the nudge cap scale. Empty list if no method evaluates.
    """
    results = []
    for name in METHODS:
        r = walk_forward(series, spec, name, n=n, guidance=guidance, beat=beat)
        if r is not None:
            results.append(r)
    results.sort(key=lambda r: r["mae"])
    return results
