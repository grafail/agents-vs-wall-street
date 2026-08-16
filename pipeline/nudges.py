"""Bounded LLM-judgment nudges on top of a deterministic baseline (no LLM here).

The estimator (pipeline.types.Estimate) never emits absolutes — only relative
quantiles (growth_p10/p50/p90) and Grounded categoricals. This module converts
that judgment into a small, capped adjustment to the baseline value:

  total_delta = quantile_component + momentum_component + skew_component
  (all in NUDGE UNITS: pct of baseline for flow/per_share, points for ratio_pct)

1. Quantile component — the p50 delta shrunk toward 0 by disagreement/spread:
       shrink = 1 / (1 + (p90 - p10) / scale),   scale = max(sigma, 1e-9)
   where sigma is the historical YoY std in the same nudge units
   (baselines.yoy_sigma). Wide, uncertain quantile spreads relative to normal
   metric variability earn heavy shrinkage; a tight spread keeps ~all of p50.
2. Categorical components — fixed sigma-scaled fallback mappings
   (momentum: +/-0.5σ .. +/-0.25σ, surprise_skew: +/-0.15σ), overridden per
   label by an optional empirically-calibrated table (label -> measured mean
   deviation in nudge units). A Grounded label with EMPTY citations is
   ungrounded and contributes exactly zero (enforced here).
3. Cap — the total adjustment in canonical units is clamped to
       |adjustment| <= K_CAP * backtest_mae      (K_CAP = 0.75)
   using the chosen method's walk-forward MAE. No MAE (no backtest possible)
   means no nudge at all: judgment without an error scale gets zero weight.

apply_nudges returns a full audit dict (every component, raw vs applied, what
capped what). reconcile_blend also lives here so ALL final arithmetic is in
deterministic quant code.
"""
from __future__ import annotations

from pipeline.types import Estimate, Grounded, MetricSpec

# Named cap constant: nudges may move the baseline by at most this fraction of
# the chosen method's backtested MAE.
K_CAP = 0.75

# Fallback sigma multipliers (used only when no calibration table entry exists).
MOMENTUM_SIGMA_MULT: dict[str, float] = {
    "cold": -0.5, "cooling": -0.25, "neutral": 0.0, "warming": 0.25, "hot": 0.5,
}
SKEW_SIGMA_MULT: dict[str, float] = {
    "downside": -0.15, "balanced": 0.0, "upside": 0.15,
}


def _categorical_component(grounded: Grounded, mult_table: dict[str, float], sigma: float,
                           calibration: dict[str, float] | None) -> dict:
    """One categorical's contribution in nudge units, with provenance.

    Empty citations => ungrounded => exactly 0 (rule enforced here, not in the
    prompt). Otherwise: calibrated measured deviation if the label is in the
    calibration table, else the fixed sigma-multiple fallback.
    """
    label = grounded.label
    if not grounded.citations:
        return {"label": label, "value": 0.0, "source": "empty_citations_zeroed"}
    if calibration is not None and label in calibration:
        return {"label": label, "value": calibration[label], "source": "calibration_table"}
    return {"label": label, "value": mult_table.get(label, 0.0) * sigma, "source": "fixed_fallback"}


def apply_nudges(baseline_value: float, estimate: Estimate, backtest_mae: float | None,
                 sigma: float | None, spec: MetricSpec,
                 calibration: dict[str, float] | None = None) -> dict:
    """Turn an Estimate into a capped adjustment on `baseline_value`.

    Args:
        baseline_value: chosen baseline candidate, canonical units.
        estimate: blind-estimator output (relative deltas + Grounded categoricals).
        backtest_mae: chosen method's walk-forward MAE in canonical units
            (None => nudge disabled, adjustment forced to 0).
        sigma: historical YoY std in nudge units (baselines.yoy_sigma);
            None/0 => categorical fallbacks contribute 0 and shrinkage uses eps scale.
        spec: metric spec (kind decides pct-of-baseline vs additive points).
        calibration: optional measured label->mean-deviation table (nudge units)
            overriding the fixed categorical mappings.

    Returns an audit dict; "pre_reconcile" is the nudged value.
    """
    # Sigma fallback: with a short series (e.g. 2 annual points) yoy_sigma is
    # None, which used to zero ALL judgment even when cited. The backtest MAE
    # is itself an empirical error scale — reuse it (converted to nudge units)
    # so bounded judgment survives sparse history. The K_CAP x MAE bound below
    # still limits total influence either way.
    sigma_source = "yoy_history"
    sig = sigma if sigma is not None else 0.0
    if (sigma is None or sig == 0.0) and backtest_mae:
        if spec.kind == "ratio_pct":
            sig = backtest_mae                                   # already points
        elif baseline_value:
            sig = abs(backtest_mae / baseline_value) * 100.0     # MAE as % of baseline
        sigma_source = "mae_fallback"

    # 1. quantile component: p50 shrunk toward 0 by spread (see module docstring)
    spread = estimate.growth_p90 - estimate.growth_p10
    scale = max(sig, 1e-9)
    shrink = 1.0 / (1.0 + spread / scale) if spread > 0 else 1.0
    quantile_component = estimate.growth_p50 * shrink

    # 2. categorical components (zeroed when uncited; calibrated when possible)
    momentum = _categorical_component(estimate.momentum, MOMENTUM_SIGMA_MULT, sig, calibration)
    skew = _categorical_component(estimate.surprise_skew, SKEW_SIGMA_MULT, sig, calibration)

    total_delta = quantile_component + momentum["value"] + skew["value"]

    # nudge units -> canonical units
    if spec.kind == "ratio_pct":
        raw_adjustment = total_delta                      # already points
    else:
        raw_adjustment = baseline_value * total_delta / 100.0

    # 3. cap at K_CAP x backtest MAE (no MAE => no nudge)
    if backtest_mae is None:
        cap, adjustment, cap_reason = 0.0, 0.0, "no_backtest_mae_nudge_disabled"
    else:
        cap = K_CAP * backtest_mae
        adjustment = max(-cap, min(cap, raw_adjustment))
        cap_reason = "capped_at_k_x_mae" if adjustment != raw_adjustment else "within_cap"

    return {
        "baseline_value": baseline_value,
        "delta_units": "points" if spec.kind == "ratio_pct" else "pct_of_baseline",
        "sigma": sigma if sigma_source == "yoy_history" else sig,
        "sigma_source": sigma_source,
        "quantiles": {"p10": estimate.growth_p10, "p50": estimate.growth_p50,
                      "p90": estimate.growth_p90, "spread": spread,
                      "shrink": shrink, "component": quantile_component},
        "momentum": momentum,
        "surprise_skew": skew,
        "total_delta": total_delta,
        "raw_adjustment": raw_adjustment,
        "k_cap": K_CAP,
        "backtest_mae": backtest_mae,
        "cap": cap,
        "cap_reason": cap_reason,
        "adjustment": adjustment,
        "pre_reconcile": baseline_value + adjustment,
    }


def reconcile_blend(ours: float, consensus: float | None, weight_ours: float) -> dict:
    """Final gated-reconcile arithmetic: weight_ours*ours + (1-weight_ours)*consensus.

    Trivial on purpose — it lives here so every number-producing operation is in
    deterministic, tested quant code. Missing consensus => ours wins outright.
    """
    if consensus is None:
        return {"final": ours, "ours": ours, "consensus": None,
                "weight_ours": 1.0, "note": "no_consensus_available"}
    w = max(0.0, min(1.0, weight_ours))
    return {"final": w * ours + (1.0 - w) * consensus, "ours": ours,
            "consensus": consensus, "weight_ours": w}
