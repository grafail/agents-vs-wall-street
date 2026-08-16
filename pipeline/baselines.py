"""Deterministic baseline forecast methods (quant layer, no LLM).

Layer order (see CLAUDE.md): anchor -> backtested baseline -> calibrated
guidance-beat factor -> categorical nudges -> gated reconcile. This module
owns the first three: it turns a metric's Fact history (+ optional guidance
facts) into candidate values in canonical units.

Every method returns a dict {"method", "value", "inputs_used"} for
auditability, or None when inapplicable (missing anchor, no guidance, ...).

Unit conventions (pipeline.types):
- flow_absolute / per_share: canonical units (millions / per-share);
  growth is multiplicative (YoY fractions).
- ratio_pct: canonical unit is percentage POINTS (4.5 == 4.5%). We forecast
  the LEVEL in points (additive trend), never growth-of-a-growth.
"""
from __future__ import annotations

import re
from datetime import date as _date

_DATE_MIN = _date.min
import statistics
from typing import Callable

from pipeline.types import Fact, MetricSpec

# ---------------------------------------------------------------- periods

_PERIOD_RE = re.compile(r"^FY(\d{4})(?:Q([1-4]))?$")


def period_key(period: str) -> tuple[int, int]:
    """Sort key for contest period strings: 'FY2026Q2' -> (2026, 2),
    annual 'FY2026' -> (2026, 0). Raises ValueError on anything else."""
    m = _PERIOD_RE.match(period.strip())
    if not m:
        raise ValueError(f"unparseable period: {period!r}")
    return int(m.group(1)), int(m.group(2) or 0)


def prior_year_period(period: str) -> str:
    """Same period one fiscal year earlier: FY2026Q2 -> FY2025Q2, FY2026 -> FY2025."""
    year, q = period_key(period)
    return f"FY{year - 1}Q{q}" if q else f"FY{year - 1}"


# ---------------------------------------------------------------- series

def _published_key(f: Fact) -> tuple:
    """Sort key for restatement preference: newest published source wins;
    facts without a published date lose to dated ones."""
    pub = getattr(f.source, "published", None)
    return (pub is not None, pub or _DATE_MIN)


class Series:
    """Ordered per-metric history built from Facts (fact_type='actual' only).

    Handles quarterly ('FY2026Q2') and annual ('FY2026') periods with correct
    year+quarter ordering. Duplicate periods: the last fact in input order wins.
    """

    def __init__(self, facts: list[Fact]):
        # Duplicate periods: prefer the fact from the most recently PUBLISHED source
        # (restated comparatives supersede as-originally-reported figures — e.g. Hays
        # restates prior-year net fees for continuing operations after divestments).
        best: dict[str, Fact] = {}
        for f in facts:
            if f.fact_type != "actual":
                continue
            cur = best.get(f.period)
            if cur is None or _published_key(f) >= _published_key(cur):
                best[f.period] = f
        by_period = {p: f.value for p, f in best.items()}
        self.points: list[tuple[str, float]] = sorted(
            by_period.items(), key=lambda kv: period_key(kv[0])
        )
        self._values = dict(self.points)

    def __len__(self) -> int:
        return len(self.points)

    @property
    def periods(self) -> list[str]:
        return [p for p, _ in self.points]

    def value(self, period: str) -> float | None:
        return self._values.get(period)

    def truncate_before(self, period: str) -> "Series":
        """New Series containing only points strictly before `period` (walk-forward)."""
        cut = period_key(period)
        s = Series([])
        s.points = [(p, v) for p, v in self.points if period_key(p) < cut]
        s._values = dict(s.points)
        return s

    # -- YoY helpers ------------------------------------------------------

    def yoy_growth_series(self) -> list[tuple[str, float]]:
        """[(period, fractional YoY growth)] for every period whose same period
        last year exists and is nonzero. Ordered."""
        out = []
        for p, v in self.points:
            prev = self._values.get(prior_year_period(p))
            if prev is not None and prev != 0:
                out.append((p, (v - prev) / prev))
        return out

    def yoy_point_changes(self) -> list[tuple[str, float]]:
        """[(period, v - v_last_year)] additive changes — used for ratio_pct."""
        out = []
        for p, v in self.points:
            prev = self._values.get(prior_year_period(p))
            if prev is not None:
                out.append((p, v - prev))
        return out

    def same_quarter_history(self, period: str) -> list[tuple[str, float]]:
        """Readings for the same fiscal quarter as `period` across years, ordered.
        For annual periods (q=0) this is simply every annual reading."""
        _, q = period_key(period)
        return [(p, v) for p, v in self.points if period_key(p)[1] == q]


def yoy_sigma(series: Series, spec: MetricSpec) -> float | None:
    """Historical YoY dispersion in NUDGE UNITS: std of YoY growth in PCT for
    flow/per_share metrics, std of YoY point changes for ratio_pct.
    Population std (deterministic, small-n friendly). None if <2 observations."""
    if spec.kind == "ratio_pct":
        obs = [c for _, c in series.yoy_point_changes()]
    else:
        obs = [g * 100.0 for _, g in series.yoy_growth_series()]
    if len(obs) < 2:
        return None
    return statistics.pstdev(obs)


# ---------------------------------------------------------------- guidance helpers

def _guidance_mid_value(guidance: list[Fact] | None, period: str) -> tuple[float | None, list[str]]:
    """Guidance midpoint for `period`: explicit guidance_mid fact wins, else
    (low+high)/2. Returns (value|None, inputs_used notes)."""
    if not guidance:
        return None, []
    for_period = [f for f in guidance if f.period == period]
    mids = [f.value for f in for_period if f.fact_type == "guidance_mid"]
    if mids:
        return mids[-1], [f"guidance_mid[{period}]={mids[-1]}"]
    lows = [f.value for f in for_period if f.fact_type == "guidance_low"]
    highs = [f.value for f in for_period if f.fact_type == "guidance_high"]
    if lows and highs:
        mid = (lows[-1] + highs[-1]) / 2.0
        return mid, [f"guidance_low[{period}]={lows[-1]}", f"guidance_high[{period}]={highs[-1]}"]
    return None, []


# ---------------------------------------------------------------- methods
# Signature: (series, spec, guidance) -> {"method","value","inputs_used"} | None
# (guidance_x_beat additionally takes the calibrated beat factor.)

def seasonal_yoy(series: Series, spec: MetricSpec, guidance: list[Fact] | None = None) -> dict | None:
    """Same-quarter-last-year anchor x (1 + recent YoY growth trend), where the
    trend is the mean of the last (up to) 2 observed YoY growth rates.

    ratio_pct: forecast the LEVEL in points — mean of the last 2 same-quarter
    readings plus their linear trend, i.e. last + 0.5*(last - prev). With a
    single reading, carry it forward. Never growth-of-a-growth.
    """
    target = spec.period
    if spec.kind == "ratio_pct":
        hist = series.same_quarter_history(target)
        hist = [(p, v) for p, v in hist if period_key(p) < period_key(target)]
        if not hist:
            return None
        if len(hist) == 1:
            (p0, v0) = hist[-1]
            return {"method": "seasonal_yoy", "value": v0,
                    "inputs_used": {"level_readings": [f"{p0}={v0}"], "trend_points": 0.0}}
        (p1, v1), (p2, v2) = hist[-2], hist[-1]
        trend = v2 - v1
        value = (v1 + v2) / 2.0 + trend          # == v2 + 0.5*trend
        return {"method": "seasonal_yoy", "value": value,
                "inputs_used": {"level_readings": [f"{p1}={v1}", f"{p2}={v2}"],
                                "trend_points": trend}}

    anchor_period = prior_year_period(target)
    anchor = series.value(anchor_period)
    if anchor is None:
        return None
    growths = [g for p, g in series.yoy_growth_series() if period_key(p) < period_key(target)]
    recent = growths[-2:]
    g = statistics.fmean(recent) if recent else 0.0
    return {"method": "seasonal_yoy", "value": anchor * (1.0 + g),
            "inputs_used": {"anchor_period": anchor_period, "anchor": anchor,
                            "recent_yoy_growths": recent, "growth_applied": g}}


def growth_drift(series: Series, spec: MetricSpec, guidance: list[Fact] | None = None) -> dict | None:
    """Anchor x (1 + drift-adjusted mean of the last 3-4 YoY growth rates).

    Drift = least-squares slope of the growth-rate sequence; the applied growth
    is the regression's prediction one step past the window (equivalently the
    window mean plus linear drift). Inapplicable to ratio_pct (that would be
    growth-of-a-growth) and when fewer than 2 YoY growth rates exist.
    """
    if spec.kind == "ratio_pct":
        return None
    target = spec.period
    anchor_period = prior_year_period(target)
    anchor = series.value(anchor_period)
    if anchor is None:
        return None
    growths = [g for p, g in series.yoy_growth_series() if period_key(p) < period_key(target)]
    window = growths[-4:]
    n = len(window)
    if n < 2:
        return None
    xs = list(range(n))
    mean_x, mean_g = statistics.fmean(xs), statistics.fmean(window)
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (g - mean_g) for x, g in zip(xs, window)) / denom
    g_next = mean_g + slope * (n - mean_x)       # regression prediction at index n
    return {"method": "growth_drift", "value": anchor * (1.0 + g_next),
            "inputs_used": {"anchor_period": anchor_period, "anchor": anchor,
                            "growth_window": window, "drift_slope": slope,
                            "growth_applied": g_next}}


def guidance_mid(series: Series, spec: MetricSpec, guidance: list[Fact] | None = None) -> dict | None:
    """Company guidance midpoint for the target period (explicit mid, else
    average of low/high). None when no guidance facts exist for the period."""
    mid, used = _guidance_mid_value(guidance, spec.period)
    if mid is None:
        return None
    return {"method": "guidance_mid", "value": mid, "inputs_used": {"guidance": used}}


def guidance_x_beat(series: Series, spec: MetricSpec, guidance: list[Fact] | None = None,
                    beat: dict | None = None) -> dict | None:
    """Guidance midpoint adjusted by the calibrated historical beat factor
    (backtest.beat_factor). Multiplicative avg_beat_pct for flows/per-share,
    additive avg_beat_points for ratio_pct. None without guidance or beat."""
    mid, used = _guidance_mid_value(guidance, spec.period)
    if mid is None or beat is None:
        return None
    if spec.kind == "ratio_pct":
        value = mid + beat["avg_beat_points"]
        applied = {"avg_beat_points": beat["avg_beat_points"]}
    else:
        value = mid * (1.0 + beat["avg_beat_pct"] / 100.0)
        applied = {"avg_beat_pct": beat["avg_beat_pct"]}
    return {"method": "guidance_x_beat", "value": value,
            "inputs_used": {"guidance": used, **applied,
                            "beat_rate": beat["beat_rate"], "beat_n": beat["n"]}}


def trend_sheet(series: Series, spec: MetricSpec, guidance: list[Fact] | None = None) -> dict:
    """Deterministic derived structure handed to the estimator (Booth-style):
    the model reasons over pre-computed trends instead of re-deriving arithmetic.

    Returns a plain dict:
      series:            [(period, value)] ordered
      yoy:               [(period, growth)] — % for flows/per-share, points for ratio_pct
      yoy_units:         "pct" | "points"
      acceleration:      [(period, delta-of-yoy)] same units
      same_quarter:      [(period, value)] readings for the target fiscal quarter
      guidance_vs_actual:[(period, guidance_mid, actual)] like-for-like by exact
                         period string (never pairs FY guidance with a quarter)
    """
    if spec.kind == "ratio_pct":
        yoy = [(p, c) for p, c in series.yoy_point_changes()]
        yoy_units = "points"
    else:
        yoy = [(p, g * 100.0) for p, g in series.yoy_growth_series()]
        yoy_units = "pct"
    accel = [(yoy[i][0], yoy[i][1] - yoy[i - 1][1]) for i in range(1, len(yoy))]

    gva: list[tuple[str, float, float]] = []
    if guidance:
        periods = sorted({f.period for f in guidance}, key=_safe_period_key)
        for p in periods:
            try:
                mid, _ = _guidance_mid_value(guidance, p)
            except ValueError:
                continue
            actual = series.value(p)
            if mid is not None and actual is not None:
                gva.append((p, mid, actual))

    try:
        same_q = series.same_quarter_history(spec.period)
    except ValueError:
        same_q = []
    return {
        "series": list(series.points),
        "yoy": yoy,
        "yoy_units": yoy_units,
        "acceleration": accel,
        "same_quarter": same_q,
        "guidance_vs_actual": gva,
    }


def _safe_period_key(period: str) -> tuple[int, int]:
    """period_key that sorts unparseable periods (e.g. FY2025H1) last instead
    of raising — trend/beat helpers must tolerate half-year facts."""
    try:
        return period_key(period)
    except ValueError:
        return (9999, 9)


METHODS: dict[str, Callable] = {
    "seasonal_yoy": seasonal_yoy,
    "growth_drift": growth_drift,
    "guidance_mid": guidance_mid,
    "guidance_x_beat": guidance_x_beat,
}


def run_all(series: Series, spec: MetricSpec, guidance: list[Fact] | None = None,
            beat: dict | None = None) -> list[dict]:
    """Run every baseline method; return the applicable candidates (audit dicts)."""
    out = []
    for name, fn in METHODS.items():
        cand = fn(series, spec, guidance, beat) if name == "guidance_x_beat" else fn(series, spec, guidance)
        if cand is not None:
            out.append(cand)
    return out
