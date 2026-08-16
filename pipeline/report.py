"""report.json -> report.html renderer + the authoritative report.json schema.

Two jobs:
1. Pydantic models (`RunReport` and friends) that define the shape of
   logs/run-<ts>/report.json. Integration builds the report dict to conform to
   these models; every field below the metric identity is Optional so skipped
   stages never break serialization or rendering.
2. `render_report(path)` -> path of a single self-contained HTML file (inline
   CSS/JS only, no external assets — works over file://) written next to the
   input, plus one per-company report-<TICKER>.html. Layered readability: plain
   for a lay reader (how-to primer, tooltips, plain-English worksheet), precise
   for an expert (full figures, formulas, backtests) — explain, never simplify.
   CLI: `uv run python -m pipeline.report logs/run-.../report.json`

This is the per-run evidence viewer, NOT the judged architecture write-up.
"""
from __future__ import annotations

import html
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline.types import Estimate, Reconciliation

# ================================================================ models

class _Model(BaseModel):
    """Tolerant base: unknown keys from integration are ignored, not fatal."""
    model_config = ConfigDict(extra="ignore")


class RunTotals(_Model):
    llm_calls: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    tool_cache_hits: int | None = None      # read-through cache: served from disk
    tool_cache_live: int | None = None      # read-through cache: live fetches
    est_cost_usd: float | None = None


class RunMeta(_Model):
    run_id: str | None = None               # e.g. run-2026-08-16T14_02_11
    started: datetime | None = None
    finished: datetime | None = None
    enable_reconcile: bool | None = None
    llm_provider: str | None = None
    model_big: str | None = None
    model_small: str | None = None
    panel_members: list[str] = Field(default_factory=list)   # estimator panel, if multi-member
    git_commit: str | None = None
    totals: RunTotals | None = None


class Anchor(_Model):
    """The exact corpus figure everything is computed from (LY actual)."""
    value: float | None = None
    period: str | None = None               # e.g. FY2025Q2
    quote: str | None = None                # source line containing the figure
    source_doc: str | None = None           # corpus-relative path or URL


class BaselineCandidate(_Model):
    method: str
    value: float | None = None
    backtest_mae: float | None = None       # same units as the metric's growth space
    backtest_bias: float | None = None
    backtest_n: int | None = None           # quarters/years in the backtest


class GuidanceBlock(_Model):
    low: float | None = None
    mid: float | None = None
    high: float | None = None
    beat_factor: float | None = None        # calibrated guidance-beat multiplier
    quote: str | None = None
    source_doc: str | None = None


class NudgeComponent(_Model):
    name: str                               # momentum | guidance_style | surprise_skew | ...
    label: str | None = None                # the categorical label that drove it
    raw: float | None = None                # nudge before caps (growth-space units)
    applied: float | None = None            # nudge after caps/shrinkage
    capped: bool = False
    note: str | None = None                 # e.g. "clipped at 0.75×MAE"


class NudgeAudit(_Model):
    components: list[NudgeComponent] = Field(default_factory=list)
    cap: float | None = None                # the k×backtest-MAE cap value
    adjustment: float | None = None         # total applied adjustment (post-cap, canonical units)
    caps_applied: list[str] = Field(default_factory=list)
    pre_reconcile_value: float | None = None  # absolute value going INTO reconcile


class ConsensusBlock(_Model):
    value: float | None = None
    source: str | None = None               # e.g. "openbb equity.estimates.consensus (yfinance)"
    fetched_at: datetime | None = None


class ValidationCheck(_Model):
    check: str
    passed: bool
    detail: str | None = None


class FallbackUsed(_Model):
    """Failsafe cascade record: which source finally produced the number."""
    source_used: str                        # estimator | baseline | guidance_mid | consensus
    reasons: list[str] = Field(default_factory=list)


class CandidateRung(_Model):
    """One rung of the failsafe cascade ladder, in cascade order.

    status: chosen  = this rung produced the final number
            viable  = had a number, never needed (a higher rung was accepted)
            skipped = had a number but was rejected (gate failure / non-finite)
            absent  = the rung produced no number at all
    """
    name: str                               # estimator_nudged | reconciled | baseline:<m> | ...
    value: float | None = None
    status: Literal["chosen", "viable", "skipped", "absent"] = "absent"
    reason: str | None = None               # skip trail from the cascade, if any


class DerivationStep(_Model):
    """One line of the per-metric derivation equation (the explainability layer).

    provenance: where the step's value comes from —
      "data" = extracted/fetched (anchor, guidance, consensus; trust tier in refs)
      "math" = deterministic computation (baselines, shrink, caps, blends)
      "llm"  = model judgment (quantiles, vibe labels, reconcile verdict)
    A step mixing LLM inputs into deterministic arithmetic is tagged "math" and
    marks the LLM-sourced terms inline in `substituted` (suffix "(LLM)").
    """
    name: str                               # anchor | baseline | quantile | ...
    formula: str                            # symbolic, e.g. "B = A × (1 + g)"
    substituted: str                        # numbers in, e.g. "B = 45,277 × (1 + 2.9%)"
    result: float | None = None
    provenance: str = "math"                # data | math | llm
    refs: list[str] = Field(default_factory=list)
    note: str | None = None


class PanelMemberRow(_Model):
    member: str                      # "provider:model"
    p50: float | None = None
    momentum: str | None = None
    guidance_style: str | None = None
    surprise_skew: str | None = None
    confidence: str | None = None


class PanelView(_Model):
    """Per-model results for one metric (only present when a multi-member
    panel ran). Aggregation is deterministic; disagreement widens uncertainty."""
    members: list[PanelMemberRow] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)   # "provider:model: error"
    p50_spread: float | None = None                    # max(p50) - min(p50)


class MetricReport(_Model):
    """One of the 12 contest records, full audit chain in pipeline order."""
    company: str
    ticker: str
    label: str
    unit: str                               # exact contest unit string, e.g. "USDm"
    period: str
    anchor: Anchor | None = None
    baseline_candidates: list[BaselineCandidate] = Field(default_factory=list)
    baseline_chosen: str | None = None      # method id of chosen candidate
    guidance: GuidanceBlock | None = None
    estimate: Estimate | None = None        # reused from pipeline.types (blind output)
    nudges: NudgeAudit | None = None
    consensus: ConsensusBlock | None = None
    reconciliation: Reconciliation | None = None  # reused from pipeline.types
    validation: list[ValidationCheck] = Field(default_factory=list)
    fallback_used: FallbackUsed | None = None
    candidates: list[CandidateRung] | None = None  # failsafe ladder, cascade order
    panel: "PanelView | None" = None               # per-model panel results for THIS metric
    derivation: list[DerivationStep] | None = None
    worksheet: list["WorksheetLine"] | None = None
    final_value: float | None = None


class WorksheetLine(_Model):
    """One line of the hero ledger (accountant's running total, plain English).
    provenance renders as a [D]/[M]/[L] prefix: data / math / model judgment."""
    provenance: str = "math"        # data | math | llm
    label: str                      # e.g. "× seasonal growth (+2.70%)"
    amount: float | None = None     # running total in canonical units
    is_final: bool = False


class RunReport(_Model):
    """Top-level shape of logs/run-<ts>/report.json."""
    meta: RunMeta = Field(default_factory=RunMeta)
    metrics: list[MetricReport] = Field(default_factory=list)


# ================================================================ rendering
#
# Design: a crafted analyst research document, not a dashboard. Serif display
# headings over a system sans text face, one navy accent, tabular numerals
# everywhere a figure appears, hairline rules, generous whitespace. The ledger
# worksheet is the visual centerpiece of every metric. Inline JS (no external
# assets — must work over file://): sticky TOC with scrollspy, expand/collapse
# all, quick filter. Everything degrades gracefully with JS off.

def _e(s: object) -> str:
    return html.escape(str(s), quote=True)


def _num(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "—"
    sign = "+" if (signed and v > 0) else ""
    if abs(v) >= 1000:
        return f"{sign}{v:,.0f}"
    s = f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{sign}{s or '0'}"


def _dt(d: datetime | None) -> str:
    return d.strftime("%Y-%m-%d %H:%M UTC") if d else "—"


_STATUS_HELP = {
    "ok": "the final number passed every sanity check",
    "warnings": "at least one sanity check flagged this number — review before trusting",
    "fallback": "the preferred forecasting path failed, so a simpler safe value was used",
    "unvalidated": "no sanity checks were recorded for this number",
    "no value": "the pipeline produced no number for this metric",
}


def _status_word(m: MetricReport) -> str:
    if m.fallback_used is not None:
        return "fallback"
    if m.final_value is None:
        return "no value"
    if any(not v.passed for v in m.validation):
        return "warnings"
    if not m.validation:
        return "unvalidated"
    return "ok"


_PREFIX = {"data": "D", "math": "M", "llm": "L"}


def _prefix(prov: str) -> str:
    return _PREFIX.get(prov, "M")


def _metric_id(m: MetricReport) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in f"{m.ticker}-{m.label}".lower())
    return "m-" + "-".join(p for p in slug.split("-") if p)


def _metric_help(label: str) -> str:
    """Plain-words tooltip for a contest metric label (labels themselves must
    stay verbatim). Keyword-matched so unknown labels still get something."""
    low = label.lower()
    if "comparable sales" in low:
        return "Sales growth in stores open for at least a year — strips out growth from newly opened stores"
    if "eps" in low or "per share" in low:
        base = "Profit per share of stock"
        if "adjusted" in low:
            return base + ", excluding one-off items the company deems non-recurring"
        if "pre-exceptional" in low:
            return base + ", before exceptional (one-off) items — UK reporting convention"
        if "gaap" in low:
            return base + ", by standard accounting rules (no adjustments)"
        return base
    if "gross margin" in low:
        return "Profit left after production costs, as a percentage of revenue" + \
            (", excluding one-off items" if "adjusted" in low else "")
    if "net fees" in low:
        return "Fee income from placing candidates — the staffing industry's core revenue measure"
    if "operating profit" in low:
        pre = "Profit from core operations"
        if "pre-exceptional" in low:
            return pre + ", before exceptional (one-off) items"
        return pre + (" for this business segment" if "&" in label else "")
    if "sales" in low or "revenue" in low:
        return "Total money taken in for the period"
    return "Reported financial figure for the period"


def _company_id(ticker: str) -> str:
    return f"co-{ticker.lower()}"


# ---------------------------------------------------------------- worksheet (hero)

def _cap_meter_html(n: NudgeAudit | None) -> str:
    """Cap-utilization meter for the judgment line: how much of the allowed
    ±cap (0.75× backtest-MAE) the model's adjustment actually consumed."""
    if n is None or n.cap is None or n.cap <= 0 or n.adjustment is None:
        return ""
    used = abs(n.adjustment)
    pct = max(0.0, min(100.0, used / n.cap * 100.0))
    return (
        f'<div class="ws-meter" title="the model&#39;s total adjustment vs the maximum it '
        f'was allowed (±0.75× the chosen method&#39;s backtest MAE)">'
        f'<span class="ws-meter-t">used {_num(used)} of ±{_num(n.cap)} ({pct:.0f}%)</span>'
        f'<div class="ws-bar"><div class="ws-bar-fill" style="width:{pct:.0f}%"></div></div>'
        f'</div>'
    )


def _worksheet_html(lines: list[WorksheetLine] | None,
                    nudges: NudgeAudit | None = None) -> str:
    if not lines:
        return '<p class="muted">No worksheet recorded for this metric.</p>'
    meter = _cap_meter_html(nudges)
    rows = []
    for ln in lines:
        final = ' <span class="ws-flag">← FINAL</span>' if ln.is_final else ""
        cls = " ws-final" if ln.is_final else ""
        rows.append(
            f'<div class="ws-row{cls}"><span class="ws-p">[{_prefix(ln.provenance)}]</span>'
            f'<span class="ws-label">{_e(ln.label)}</span>'
            f'<span class="ws-amt">{_num(ln.amount)}{final}</span></div>'
        )
        if meter and ln.provenance == "llm":
            rows.append(meter)   # attach to the judgment line only
            meter = ""
    return (
        '<div class="ws"><div class="ws-legend">D data · M math · L model judgment</div>'
        f'{"".join(rows)}</div>'
    )


def _consensus_check_html(m: MetricReport) -> str:
    """Always-visible one-liner: our final vs Wall Street, or why not."""
    kspan = ('<span class="cons-k" title="how our number compares with the average of Wall '
             'Street analysts&#39; published estimates">Consensus check</span>')
    c = m.consensus
    if c is None or c.value is None:
        note = (c.source if (c is not None and c.source) else "no comparable consensus series")
        return (f'<p class="cons">{kspan} '
                f'<span class="muted">not available — {_e(note)}</span></p>')
    src = f' <span class="src">[{_e(c.source)}]</span>' if c.source else ""
    if m.final_value is None:
        return (f'<p class="cons">{kspan} '
                f'street {_num(c.value)}{src} · <span class="muted">no final value</span></p>')
    if m.unit == "%":
        diff = m.final_value - c.value
        rel = f"{diff:+.2f} pts vs street"
    else:
        rel = (f"{(m.final_value - c.value) / abs(c.value) * 100:+.2f}% vs street"
               if c.value else "—")
    return (f'<p class="cons">{kspan} '
            f'ours <b>{_num(m.final_value)}</b> · street <b>{_num(c.value)}</b> · '
            f'<b class="accent">{_e(rel)}</b>{src}</p>')


def _delta_vs_street(m: MetricReport) -> str:
    """Summary-table delta: percent vs consensus, points for '%' metrics, — else."""
    c = m.consensus
    if c is None or c.value is None or m.final_value is None:
        return "—"
    if m.unit == "%":
        return f"{m.final_value - c.value:+.1f}pp"
    if not c.value:
        return "—"
    return f"{(m.final_value - c.value) / abs(c.value) * 100:+.1f}%"


def _ladder_html(rungs: list[CandidateRung] | None) -> str:
    """Always-visible failsafe ladder: every cascade rung, in order, with status."""
    if not rungs:
        return ""
    bits = []
    for r in rungs:
        name, val = _e(r.name), _num(r.value)
        tip = f' title="{_e(r.reason)}"' if r.reason else ""
        if r.status == "chosen":
            bits.append(f'<span class="rung rung-chosen"{tip}>{name} {val} CHOSEN</span>')
        elif r.status == "viable":
            bits.append(f'<span class="rung"{tip}>{name} {val} <span class="muted">viable</span></span>')
        elif r.status == "skipped":
            bits.append(f'<span class="rung rung-skipped"{tip}>{name} {val} skipped</span>')
        else:  # absent
            bits.append(f'<span class="rung muted"{tip}>{name} —</span>')
    kspan = ('<span class="cons-k" title="the never-blank failsafe cascade: rungs are tried '
             'in this order and the first one that passes every hard gate becomes the final '
             'number; later rungs stay on standby">Cascade</span>')
    return f'<p class="ladder">{kspan} {" · ".join(bits)}</p>'


# ---------------------------------------------------------------- formulas details

def _derivation_html(steps: list[DerivationStep] | None) -> str:
    if not steps:
        return ""
    lines = []
    for s in steps:
        refs_bits = list(s.refs)
        if s.note:
            refs_bits.append(s.note)
        refs = (f'<div class="d-refs">↳ {_e("; ".join(refs_bits))}</div>' if refs_bits else "")
        lines.append(
            f'<div class="d-step"><div class="d-head">'
            f'<span class="ws-p">[{_prefix(s.provenance)}]</span>'
            f'<span class="d-name">{_e(s.name)}</span>'
            f'<code class="d-formula">{_e(s.formula)}</code></div>'
            f'<code class="d-sub">{_e(s.substituted)}</code>{refs}</div>'
        )
    return (
        '<details class="stage"><summary>Show full formulas &amp; sources</summary>'
        '<div class="stage-body"><p class="muted small">D data · M math · L model judgment — '
        'every model-judgment term is bounded by a computed cap.</p>'
        f'{"".join(lines)}</div></details>'
    )


# ---------------------------------------------------------------- stage bodies

def _anchor_html(a: Anchor | None) -> str:
    if a is None:
        return '<p class="muted">No anchor recorded.</p>'
    parts = [f'<p><b>{_num(a.value)}</b> <span class="muted">({_e(a.period or "—")})</span></p>']
    if a.quote:
        parts.append(f'<blockquote>{_e(a.quote)}</blockquote>')
    if a.source_doc:
        parts.append(f'<p class="src">source: <code>{_e(a.source_doc)}</code></p>')
    return "".join(parts)


def _baseline_html(cands: list[BaselineCandidate], chosen: str | None) -> str:
    if not cands:
        return '<p class="muted">No baseline candidates recorded.</p>'
    rows = []
    for c in cands:
        star = " ★" if (chosen is not None and c.method == chosen) else ""
        cls = ' class="chosen"' if star else ""
        rows.append(
            f"<tr{cls}><td>{_e(c.method)}{star}</td><td class='num'>{_num(c.value)}</td>"
            f"<td class='num'>{_num(c.backtest_mae)}</td>"
            f"<td class='num'>{_num(c.backtest_bias, signed=True)}</td>"
            f"<td class='num'>{c.backtest_n if c.backtest_n is not None else '—'}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>method</th><th class="num">value</th>'
        '<th class="num"><abbr title="average historical miss: Mean Absolute Error when the '
        'method is replayed on past quarters">backtest MAE</abbr></th>'
        '<th class="num"><abbr title="average signed miss — positive means the method '
        'historically overshoots">bias</abbr></th>'
        '<th class="num"><abbr title="number of past periods evaluated">n</abbr></th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _guidance_html(g: GuidanceBlock | None) -> str:
    if g is None:
        return '<p class="muted">No guidance recorded.</p>'
    parts = [
        f"<p>low <b>{_num(g.low)}</b> · mid <b>{_num(g.mid)}</b> · high <b>{_num(g.high)}</b>"
        f" · beat factor <b>{_num(g.beat_factor)}</b></p>"
    ]
    if g.quote:
        parts.append(f"<blockquote>{_e(g.quote)}</blockquote>")
    if g.source_doc:
        parts.append(f'<p class="src">source: <code>{_e(g.source_doc)}</code></p>')
    return "".join(parts)


def _grounded_html(name: str, g) -> str:
    if g is None:
        return f"<p><b>{_e(name)}:</b> <span class='muted'>—</span></p>"
    cites = (f' <span class="src">[{_e("; ".join(g.citations))}]</span>' if g.citations
             else ' <span class="muted small">(no citations — treated as neutral)</span>')
    return f"<p><b>{_e(name)}:</b> {_e(g.label)} — {_e(g.explanation)}{cites}</p>"


def _panel_html(p: "PanelView | None") -> str:
    """Per-model panel table: one row per member, aggregated row context above."""
    if p is None or (not p.members and not p.failed):
        return ""
    rows = []
    for mm in p.members:
        rows.append(
            f"<tr><td>{_e(mm.member)}</td><td class=\"num\">{_num(mm.p50)}%</td>"
            f"<td>{_e(mm.momentum or '—')}</td><td>{_e(mm.guidance_style or '—')}</td>"
            f"<td>{_e(mm.surprise_skew or '—')}</td><td>{_e(mm.confidence or '—')}</td></tr>")
    failed = "".join(f'<p class="muted" style="font-size:.8rem">member failed: {_e(f)}</p>'
                     for f in p.failed)
    spread = (f'<p class="muted" style="font-size:.82rem">p50 disagreement (spread): '
              f'{_num(p.p50_spread)}% — larger disagreement automatically widens the '
              f'uncertainty band and shrinks the panel\'s influence.</p>'
              if p.p50_spread is not None else "")
    return ('<h4 style="margin:.7rem 0 .2rem">Model panel (per-member answers)</h4>'
            '<div style="overflow-x:auto"><table><tr><th>model</th><th>p50</th>'
            '<th>momentum</th><th>guidance style</th><th>surprise skew</th>'
            '<th>confidence</th></tr>' + "".join(rows) + "</table></div>" + spread + failed)


def _estimate_html(est: Estimate | None) -> str:
    if est is None:
        return '<p class="muted">No blind estimate (estimator skipped or failed).</p>'
    return "".join([
        f"<p>method <code>{_e(est.method)}</code> · growth p10 <b>{_num(est.growth_p10, signed=True)}</b>"
        f" / p50 <b>{_num(est.growth_p50, signed=True)}</b> / p90 <b>{_num(est.growth_p90, signed=True)}</b>"
        f" · confidence <b>{_e(est.confidence)}</b></p>",
        _grounded_html("momentum", est.momentum),
        _grounded_html("guidance_style", est.guidance_style),
        _grounded_html("surprise_skew", est.surprise_skew),
        f"<p><b>Rationale:</b> {_e(est.rationale)}</p>",
    ])


def _nudges_html(n: NudgeAudit | None) -> str:
    if n is None:
        return '<p class="muted">No nudge audit recorded.</p>'
    parts = []
    if n.components:
        rows = "".join(
            f"<tr><td>{_e(c.name)}</td><td>{_e(c.label) if c.label else '—'}</td>"
            f"<td class='num'>{_num(c.raw, signed=True)}</td>"
            f"<td class='num'>{_num(c.applied, signed=True)}</td>"
            f"<td>{'capped' if c.capped else '—'}</td><td>{_e(c.note) if c.note else '—'}</td></tr>"
            for c in n.components
        )
        parts.append(
            '<div class="scroll"><table><thead><tr><th>component</th><th>label</th>'
            '<th class="num">raw</th><th class="num">applied</th><th>cap</th><th>note</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )
    else:
        parts.append('<p class="muted">No nudge components.</p>')
    if n.cap is not None:
        parts.append(f"<p>cap (k×backtest-MAE): <b>{_num(n.cap)}</b></p>")
    if n.caps_applied:
        parts.append("<p>caps applied: " + ", ".join(_e(c) for c in n.caps_applied) + "</p>")
    if n.pre_reconcile_value is not None:
        parts.append(f"<p>value entering reconcile: <b>{_num(n.pre_reconcile_value)}</b></p>")
    return "".join(parts)


def _reconcile_html(c: ConsensusBlock | None, r: Reconciliation | None) -> str:
    parts = []
    if c is not None:
        src = f' <span class="muted">({_e(c.source)})</span>' if c.source else ""
        parts.append(f"<p>consensus: <b>{_num(c.value)}</b>{src}</p>")
    else:
        parts.append('<p class="muted">No consensus fetched.</p>')
    if r is not None:
        parts.append(
            f"<p>verdict <b>{_e(r.verdict)}</b> · weight on our estimate "
            f"<b>{_num(r.weight_ours)}</b></p><p><b>Rationale:</b> {_e(r.rationale)}</p>"
        )
    else:
        parts.append('<p class="muted">No reconciliation (pure-blind mode or skipped).</p>')
    return "".join(parts)


def _validation_html(checks: list[ValidationCheck]) -> str:
    if not checks:
        return '<p class="muted">No validation checks recorded.</p>'
    items = []
    for v in checks:
        mark = "✓" if v.passed else "✗"
        cls = "ok" if v.passed else "bad"
        detail = f' <span class="muted small">{_e(v.detail)}</span>' if v.detail else ""
        items.append(f'<li><span class="{cls}">{mark}</span> {_e(v.check)}{detail}</li>')
    return f'<ul class="checks">{"".join(items)}</ul>'


def _fallback_html(f: FallbackUsed | None) -> str:
    if f is None:
        return ""
    reasons = "".join(f"<li>{_e(r)}</li>" for r in f.reasons) or "<li class='muted'>no reason recorded</li>"
    return (
        f'<div class="fallback"><b>Failsafe cascade fired</b> — final number came from '
        f"<code>{_e(f.source_used)}</code><ul>{reasons}</ul></div>"
    )


def _stage(title: str, body: str, help_html: str | None = None) -> str:
    """help_html is trusted internal copy (may contain <abbr>) — not escaped."""
    help_p = f'<p class="stage-help">{help_html}</p>' if help_html else ""
    return (f'<details class="stage"><summary>{_e(title)}</summary>'
            f'<div class="stage-body">{help_p}{body}</div></details>')


_ABBR_MAE = ('<abbr title="Mean Absolute Error — the average size of the method&#39;s '
             'historical misses">MAE</abbr>')
_ABBR_CONSENSUS = ('<abbr title="the average of Wall Street analysts&#39; published '
                   'estimates">consensus</abbr>')

_STAGE_HELP = {
    1: "The exact figure the company reported for the same period last year — the factual "
       "starting point every forecast below is computed from.",
    2: "Simple statistical forecasts built only from the company&#39;s own reported history — "
       "no AI involved. Each method is <abbr title=\"replayed on past quarters using only the "
       "data available at the time, then compared with what was actually reported\">backtested"
       f"</abbr>; the one with the smallest average historical miss ({_ABBR_MAE}) is chosen (★).",
    3: "Guidance is the company&#39;s own published forecast for this period. The beat factor "
       "measures how much this company historically lands above or below its own guidance.",
    4: "The AI model&#39;s judgment, formed <b>without</b> seeing Wall Street&#39;s numbers. "
       "p10 / p50 / p90 are its pessimistic / central / optimistic calls for how the actual "
       "result deviates from the baseline; the labels are its reads on business momentum and "
       "management style. Any label without cited evidence counts for nothing.",
    5: "How the model&#39;s judgment became a small, bounded adjustment: wide uncertainty "
       "shrinks its own influence, and the total move is capped at 0.75× the chosen "
       f"method&#39;s average historical miss ({_ABBR_MAE}).",
    6: f"{_ABBR_CONSENSUS.capitalize()} is revealed to the model only at this stage; it must "
       "defend any gap with the evidence it already cited, or defer to the street.",
    7: "Automatic sanity checks on the final number: unit traps, plausibility against the "
       "company&#39;s own history, and distance from guidance.",
}


def _metric_section(m: MetricReport) -> str:
    word = _status_word(m)
    final = (f"{_num(m.final_value)} <span class='m-unit'>{_e(m.unit)}</span>"
             if m.final_value is not None else "—")
    stages = [
        _stage("1 · Anchor", _anchor_html(m.anchor), _STAGE_HELP[1]),
        _stage("2 · Baselines (backtested)",
               _baseline_html(m.baseline_candidates, m.baseline_chosen), _STAGE_HELP[2]),
        _stage("3 · Guidance & beat factor", _guidance_html(m.guidance), _STAGE_HELP[3]),
        _stage("4 · Blind estimate", _estimate_html(m.estimate) + _panel_html(m.panel), _STAGE_HELP[4]),
        _stage("5 · Nudge audit", _nudges_html(m.nudges), _STAGE_HELP[5]),
        _stage("6 · Consensus & reconciliation",
               _reconcile_html(m.consensus, m.reconciliation), _STAGE_HELP[6]),
        _stage("7 · Validation gates", _validation_html(m.validation), _STAGE_HELP[7]),
    ]
    status = (f'<span class="m-status{" m-status-bad" if word in ("fallback", "no value") else ""}"'
              f' title="{_e(_STATUS_HELP.get(word, ""))}">{_e(word)}</span>')
    return f"""
<section class="metric" id="{_metric_id(m)}" data-filter="{_e(f'{m.company} {m.ticker} {m.label}'.lower())}">
<div class="m-head">
  <div class="m-id"><h3 title="{_e(_metric_help(m.label))}">{_e(m.label)}</h3>
    <div class="m-sub">{_e(m.period)} · {status}</div></div>
  <div class="m-final">{final}</div>
</div>
{_fallback_html(m.fallback_used)}
{_worksheet_html(m.worksheet, m.nudges)}
{_consensus_check_html(m)}
{_ladder_html(m.candidates)}
{_derivation_html(m.derivation)}
{"".join(stages)}
</section>"""


def _group_by_company(metrics: list[MetricReport]) -> list[tuple[str, str, list[MetricReport]]]:
    """[(company, ticker, metrics)] preserving first-seen order."""
    order: list[str] = []
    groups: dict[str, list[MetricReport]] = {}
    for m in metrics:
        if m.company not in groups:
            order.append(m.company)
            groups[m.company] = []
        groups[m.company].append(m)
    return [(co, groups[co][0].ticker, groups[co]) for co in order]


_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_TIER_RE = re.compile(r"\(tier (\d)\)")


def _bibliography_html(company: str, ms: list[MetricReport]) -> str:
    """Every cited document/URL across the company's metrics, deduped, with
    trust tier (when a derivation ref recorded one) and any date in the name."""
    sources: dict[str, dict] = {}   # key -> {"tier": str|None, "used": [labels]}

    def _add(raw: str | None, label: str) -> None:
        if not raw:
            return
        tier_m = _TIER_RE.search(raw)
        key = _TIER_RE.sub("", raw).strip()
        # citations often carry an inline quote: "doc.md: 'quoted line'"
        if ": '" in key:
            key = key.split(": '", 1)[0].strip()
        if not key:
            return
        entry = sources.setdefault(key, {"tier": None, "used": []})
        if tier_m:
            entry["tier"] = tier_m.group(1)
        if label not in entry["used"]:
            entry["used"].append(label)

    for m in ms:
        if m.anchor is not None:
            _add(m.anchor.source_doc, m.label)
        if m.guidance is not None:
            _add(m.guidance.source_doc, m.label)
        if m.estimate is not None:
            for g in (m.estimate.momentum, m.estimate.guidance_style, m.estimate.surprise_skew):
                if g is not None:
                    for c in g.citations:
                        _add(c, m.label)
        if m.consensus is not None:
            _add(m.consensus.source, m.label)
        for s in m.derivation or []:
            for r in s.refs:
                if _TIER_RE.search(r):    # only refs that name a tiered document
                    _add(r, m.label)
    if not sources:
        return ""
    rows = []
    for key in sorted(sources):
        e = sources[key]
        date_m = _DATE_RE.search(key)
        rows.append(
            f'<tr><td><code>{_e(key)}</code></td>'
            f'<td>{("tier " + e["tier"]) if e["tier"] else "—"}</td>'
            f'<td>{date_m.group(1) if date_m else "—"}</td>'
            f'<td>{_e("; ".join(e["used"]))}</td></tr>')
    return (
        f'<details class="stage biblio"><summary>Sources cited — {_e(company)} '
        f'({len(sources)})</summary><div class="stage-body">'
        '<p class="stage-help">Every document, feed or URL cited anywhere in this '
        'company&#39;s forecasts, deduplicated. Tier 1 = the company&#39;s own filings '
        'and releases.</p>'
        '<div class="scroll"><table><thead><tr><th>source</th><th>trust</th><th>date</th>'
        f'<th>cited by</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        '</div></details>')


def _company_sections(metrics: list[MetricReport]) -> str:
    out = []
    for company, ticker, ms in _group_by_company(metrics):
        out.append(
            f'<section class="company" id="{_company_id(ticker)}">'
            f'<h2 class="co-name">{_e(company)} <span class="co-tick">{_e(ticker)}'
            f' · {_e(ms[0].period)}</span></h2>'
            + "".join(_metric_section(m) for m in ms)
            + _bibliography_html(company, ms)
            + "</section>")
    return "".join(out)


def _toc_html(metrics: list[MetricReport]) -> str:
    groups = _group_by_company(metrics)
    if not groups:
        return ""
    items = []
    for company, ticker, ms in groups:
        links = "".join(
            f'<a class="toc-m" href="#{_metric_id(m)}">{_e(m.label)}'
            f'<span class="toc-v">{_num(m.final_value)}</span></a>'
            for m in ms)
        items.append(
            f'<div class="toc-co"><a class="toc-c" href="#{_company_id(ticker)}">{_e(company)}</a>'
            f'{links}</div>')
    return f"""
<nav id="toc">
<div class="toc-title">Contents</div>
<div class="jsonly toolbar">
  <input id="filter" type="search" placeholder="filter metrics…" autocomplete="off">
  <div class="tb-btns"><button id="expand">expand all</button><button id="collapse">collapse all</button></div>
</div>
{"".join(items)}
</nav>"""


def _howto_html() -> str:
    """Collapsible primer written for a lay reader; experts skip it. Trusted
    internal copy — may contain markup."""
    defs = [
        ("baseline", "a simple statistical starting forecast computed purely from the "
         "company's own reported history — for example, last year's quarter grown at the "
         "recent growth rate, or the midpoint of the company's own guidance. No AI involved."),
        ("backtest", "a rehearsal on the past: each baseline method is replayed on earlier "
         "quarters using only the data that was available at the time, and its forecasts are "
         "compared with what the company actually reported. The average miss (MAE) decides "
         "which method to trust — and limits how much the AI may change its answer."),
        ("judgment", "the AI model's adjustment, produced without seeing Wall Street's "
         "numbers. Every claim must cite a source document, and the total adjustment is "
         "capped at 0.75× the chosen method's average historical miss."),
        ("guidance", "the company's own published forecast for the period."),
        ("consensus & blend", "consensus is the average of Wall Street analysts' estimates; "
         "a blend is a weighted average of our value and consensus, used when the model "
         "judges the street partly right."),
        ("fallback", "if a preferred step fails, the pipeline falls back to the next safest "
         "value — baseline, guidance, consensus, or last year's actual — rather than ever "
         "leaving a blank."),
        ("[D] / [M] / [L]", "every worksheet line is tagged by where its number comes from: "
         "D extracted data, M deterministic math, L model judgment."),
        ("statuses", " · ".join(f"<b>{k}</b>: {v}" for k, v in _STATUS_HELP.items())),
    ]
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in defs)
    return f"""
<details class="howto"><summary>How to read this report</summary>
<div class="howto-body">
<p>Each forecast below is built in visible layers: start from what the company actually
reported last year (<i>data</i>), grow it with a simple statistical method chosen by its
historical accuracy (<i>math</i>), let an AI model adjust it within strict, evidence-cited
limits (<i>model judgment</i>), then sanity-check the result against Wall Street's average
estimate. The worksheet under each metric shows this as a running ledger; every claim
links back to its source. Nothing is rounded away for simplicity — the layers are
explained instead.</p>
<dl>{dl}</dl>
</div></details>"""


def _review_items(m: MetricReport) -> list[str]:
    """Everything on this metric an operator should eyeball before uploading.
    Plain strings; the rollup box turns them into anchor-linked lines."""
    items: list[str] = []
    if m.fallback_used is not None:
        items.append(f"fallback — final number came from {m.fallback_used.source_used}")
    failsafe = (m.fallback_used is not None
                and ("FAILSAFE" in m.fallback_used.source_used
                     or any("FAILSAFE" in r for r in m.fallback_used.reasons)))
    if failsafe:
        items.append("FAILSAFE — every cascade rung failed gates; number needs manual review")
    for v in m.validation:
        if not v.passed:
            detail = f" — {v.detail}" if v.detail else ""
            items.append(f"gate failed: {v.check}{detail}")
        elif v.check == "extraction_flags" and v.detail:
            for flag in re.split(r"[,\s]+", v.detail):
                if flag.startswith("auto_corrected_scale"):
                    items.append(f"auto-corrected unit during extraction: {flag}")
    if m.estimate is not None and m.estimate.confidence == "low":
        items.append("estimator confidence low — judgment layer carries little conviction")
    return items


def _rollup_html(metrics: list[MetricReport]) -> str:
    """The operator's pre-upload checklist: one aggregated box of everything
    that needs a human look before the numbers go to OpenStocks."""
    lines = []
    for m in metrics:
        for item in _review_items(m):
            lines.append(
                f'<li><a href="#{_metric_id(m)}">{_e(m.ticker)} · {_e(m.label)}</a>'
                f' — {_e(item)}</li>')
    if not lines:
        body = ('<p class="rollup-ok">Nothing needs review — every metric passed its '
                'gates on the preferred path.</p>')
    else:
        body = f'<ul>{"".join(lines)}</ul>'
    return (
        '<section class="rollup" id="pre-upload">'
        '<p class="rollup-t">Pre-upload checklist</p>'
        '<p class="rollup-sub">Review each line before manually uploading the workbooks — '
        'fallbacks, failed gates, unit auto-corrections and low-conviction estimates.</p>'
        f'{body}</section>')


def _summary_html(metrics: list[MetricReport]) -> str:
    if not metrics:
        return '<p class="muted">No metrics in report.</p>'
    rows = []
    for m in metrics:
        val = f"{_num(m.final_value)} {_e(m.unit)}" if m.final_value is not None else "—"
        word = _status_word(m)
        rows.append(
            f'<tr><td>{_e(m.company)}</td><td><a href="#{_metric_id(m)}">{_e(m.label)}</a></td>'
            f'<td class="num">{val}</td>'
            f'<td class="num">{_e(_delta_vs_street(m))}</td>'
            f'<td title="{_e(_STATUS_HELP.get(word, ""))}">{_e(word)}</td></tr>'
        )
    legend = " · ".join(f"<b>{_e(k)}</b> {_e(v)}" for k, v in _STATUS_HELP.items()
                        if k != "no value")
    return (
        '<table class="sum"><thead><tr><th>company</th><th>metric</th>'
        '<th class="num">final</th>'
        '<th class="num"><abbr title="our final vs the average of Wall Street analysts&#39; '
        'estimates — percent difference, or points (pp) for percentage metrics">vs street'
        '</abbr></th><th>status</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        f'<p class="sum-legend">{legend}</p>'
    )


def _header_html(meta: RunMeta, subtitle: str | None = None) -> str:
    mode_bits = []
    if meta.enable_reconcile is not None:
        mode_bits.append("reconcile ON" if meta.enable_reconcile else "PURE BLIND")
    if meta.llm_provider:
        mode_bits.append(meta.llm_provider)
    mode = " · ".join(mode_bits)
    kv = [
        ("run", meta.run_id or "—"),
        ("started", _dt(meta.started)),
        ("finished", _dt(meta.finished)),
        ("estimator panel" if meta.panel_members else "model (big)",
         " + ".join(meta.panel_members) if meta.panel_members else (meta.model_big or "—")),
        ("reconciler" if meta.panel_members else "model (small)",
         (meta.model_big or "—") if meta.panel_members else (meta.model_small or "—")),
        *((("reader (small)", meta.model_small or "—"),) if meta.panel_members else ()),
        ("commit", (meta.git_commit or "—")[:12]),
    ]
    t = meta.totals
    if t is not None:
        cached = t.cached_prompt_tokens or 0
        prompt = t.prompt_tokens or 0
        pct = f" ({cached / prompt * 100:.0f}% of prompt)" if prompt and cached else ""
        kv += [
            ("LLM calls", str(t.llm_calls) if t.llm_calls is not None else "—"),
            ("tokens", f"{_num(t.prompt_tokens)} prompt / {_num(t.completion_tokens)} completion"),
            ("cached tokens", f"{_num(t.cached_prompt_tokens)}{pct}"),
            ("tool cache", f"{t.tool_cache_hits if t.tool_cache_hits is not None else '—'} hit / "
                           f"{t.tool_cache_live if t.tool_cache_live is not None else '—'} live"),
            ("est. cost", f"${t.est_cost_usd:.2f}" if t.est_cost_usd is not None else "—"),
        ]
    kv_html = "".join(f'<div class="kv"><span class="k">{_e(k)}</span><span class="v">{_e(v)}</span></div>'
                      for k, v in kv)
    sub = f'<p class="subtitle">{_e(subtitle)}</p>' if subtitle else ""
    mode_html = f'<p class="mode">{_e(mode)}</p>' if mode else ""
    return (f"<header><h1>Forecast run report</h1>{sub}{mode_html}"
            f"<div class='meta-grid'>{kv_html}</div></header>")


_CSS = """
:root { --ink:#191a1c; --muted:#71747a; --line:#e4e4e6; --hair:#eeeeef;
        --accent:#1a4d8f; --paper:#fdfdfc; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; background:var(--paper); color:var(--ink);
       font:15px/1.6 -apple-system,"Segoe UI",system-ui,Roboto,Helvetica,Arial,sans-serif;
       font-variant-numeric:tabular-nums; }
.layout { display:grid; grid-template-columns:15rem minmax(0,46rem); gap:3rem;
          max-width:66rem; margin:0 auto; padding:2.75rem clamp(1rem,4vw,3rem) 5rem; }
main { min-width:0; }
h1, h2, h3, .m-final { font-family:Georgia,"Times New Roman",serif; }
h1 { font-size:1.7rem; margin:0 0 .35rem; font-weight:700; letter-spacing:-.01em; }
.subtitle { margin:.1rem 0 .4rem; color:var(--muted); font-size:1rem; }
h2.sec { font-size:.8rem; margin:2.6rem 0 .8rem; text-transform:uppercase; letter-spacing:.12em;
     color:var(--muted); font-weight:600; font-family:inherit; }
.co-name { font-size:1.35rem; margin:0 0 .2rem; font-weight:700; }
.co-tick { font-family:-apple-system,"Segoe UI",system-ui,sans-serif; font-size:.8rem;
           color:var(--muted); font-weight:400; letter-spacing:.04em; margin-left:.4rem; }
h3 { font-size:1.12rem; margin:0; font-weight:700; }
a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.85em; }
.muted { color:var(--muted); } .small { font-size:.85em; } .accent { color:var(--accent); }
.mode { color:var(--muted); margin:.1rem 0 1rem; font-size:.88rem; letter-spacing:.03em; }
header { border-bottom:2px solid var(--ink); padding-bottom:1.2rem; }
.meta-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:.15rem 2rem; }
.kv { display:flex; justify-content:space-between; gap:1rem; padding:.1rem 0; font-size:.88rem; }
.kv .k { color:var(--muted); } .kv .v { text-align:right; }
/* ---- TOC ---- */
#toc { position:sticky; top:1.5rem; align-self:start; font-size:.85rem;
       max-height:calc(100vh - 3rem); overflow-y:auto; padding-right:.5rem; }
.toc-title { font-family:Georgia,serif; font-weight:700; font-size:.95rem; margin-bottom:.6rem; }
.toc-co { margin:.7rem 0; }
.toc-c { display:block; font-weight:700; color:var(--ink); padding:.15rem 0; }
.toc-m { display:flex; justify-content:space-between; gap:.6rem; color:var(--muted);
         padding:.14rem 0 .14rem .8rem; border-left:1px solid var(--hair); }
.toc-m:hover { color:var(--accent); text-decoration:none; }
.toc-m.active { color:var(--accent); border-left:2px solid var(--accent); padding-left:calc(.8rem - 1px); }
.toc-v { font-family:ui-monospace,Menlo,monospace; font-size:.9em; }
.toolbar { margin:.4rem 0 .9rem; }
#filter { width:100%; padding:.35rem .55rem; border:1px solid var(--line); border-radius:5px;
          font:inherit; background:#fff; }
#filter:focus { outline:none; border-color:var(--accent); }
.tb-btns { display:flex; gap:.5rem; margin-top:.45rem; }
.tb-btns button { flex:1; font:inherit; font-size:.78rem; padding:.25rem 0; background:#fff;
                  border:1px solid var(--line); border-radius:5px; color:var(--muted); cursor:pointer; }
.tb-btns button:hover { color:var(--accent); border-color:var(--accent); }
.jsonly { display:none; } .js .jsonly { display:block; }
/* ---- summary ---- */
table.sum { border-collapse:collapse; width:100%; font-size:.92rem; }
table.sum th, table.sum td { text-align:left; padding:.34rem .6rem .34rem 0; border-bottom:1px solid var(--hair); }
table.sum th { font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); font-weight:600; }
td.num, th.num { text-align:right; font-family:ui-monospace,Menlo,monospace; font-size:.92em; }
/* ---- metric ---- */
.company { margin-top:3rem; border-top:2px solid var(--ink); padding-top:1.2rem; }
.metric { border-top:1px solid var(--line); margin-top:1.6rem; padding-top:1.3rem; }
.metric:first-of-type { border-top:none; margin-top:.6rem; }
.m-head { display:flex; justify-content:space-between; align-items:flex-start; gap:1.5rem;
          flex-wrap:wrap; margin-bottom:.9rem; }
.m-sub { color:var(--muted); font-size:.83rem; margin-top:.2rem; }
.m-status { text-transform:uppercase; letter-spacing:.07em; font-size:.75rem; }
.m-status-bad { color:var(--accent); font-weight:700; }
.m-final { font-size:2.1rem; font-weight:700; white-space:nowrap; line-height:1.1; }
.m-unit { font-size:.5em; color:var(--muted); font-weight:400;
          font-family:-apple-system,"Segoe UI",system-ui,sans-serif; }
/* ---- worksheet ---- */
.ws { margin:.5rem 0 .6rem; padding:.75rem 1rem .6rem; background:#fff;
      border:1px solid var(--line); border-radius:4px;
      box-shadow:0 1px 0 rgba(0,0,0,.02); }
.ws-legend { font-size:.7rem; color:var(--muted); letter-spacing:.06em; margin-bottom:.55rem;
             font-family:ui-monospace,Menlo,monospace; }
.ws-row { display:grid; grid-template-columns:2.4rem 1fr minmax(9.5rem,auto); gap:.7rem;
          padding:.26rem 0; align-items:baseline; }
.ws-row + .ws-row { border-top:1px solid var(--hair); }
.ws-p { font-family:ui-monospace,Menlo,monospace; color:var(--muted); font-size:.82em; }
.ws-amt { text-align:right; font-family:ui-monospace,Menlo,monospace; font-size:.98em;
          font-weight:600; white-space:nowrap; }
.ws-final { border-top:1px solid var(--ink) !important; }
.ws-final .ws-label, .ws-final .ws-amt { font-weight:700; }
.ws-flag { color:var(--accent); font-size:.75em; font-weight:700; margin-left:.45rem;
           font-family:-apple-system,"Segoe UI",system-ui,sans-serif; letter-spacing:.03em; }
.cons { margin:.2rem 0 .5rem; font-size:.9rem; }
.cons-k { font-size:.7rem; text-transform:uppercase; letter-spacing:.09em; color:var(--muted);
          font-weight:700; margin-right:.5rem; }
.ws-meter { display:flex; align-items:center; gap:.6rem; padding:.05rem 0 .35rem;
            margin-left:3.1rem; }
.ws-meter-t { font-size:.72rem; color:var(--muted); white-space:nowrap;
              font-family:ui-monospace,Menlo,monospace; }
.ws-bar { flex:0 0 7rem; height:3px; background:var(--hair); }
.ws-bar-fill { height:3px; background:var(--accent); }
.ladder { margin:.1rem 0 1rem; font-size:.84rem; color:var(--ink); line-height:1.9; }
.ladder .rung { white-space:nowrap; font-family:ui-monospace,Menlo,monospace; font-size:.94em; }
.rung-chosen { color:var(--accent); font-weight:700; }
.rung-skipped { color:var(--muted); text-decoration:line-through; text-decoration-thickness:1px; }
.rollup { margin:1.1rem 0 .2rem; border:1px solid var(--accent); border-radius:4px;
          background:#fff; padding:.65rem 1rem .8rem; }
.rollup-t { font-family:Georgia,"Times New Roman",serif; font-weight:700; font-size:1rem;
            margin:0; }
.rollup-sub { color:var(--muted); font-size:.8rem; margin:.1rem 0 .4rem; }
.rollup ul { margin:.25rem 0 0; padding-left:1.15rem; font-size:.88rem; }
.rollup li { margin:.24rem 0; }
.rollup-ok { color:#256b45; font-size:.9rem; margin:.25rem 0 0; }
.biblio { margin-top:1.4rem; border-top:1px solid var(--line); }
/* ---- stages ---- */
details.stage { border-top:1px solid var(--hair); }
details.stage summary { cursor:pointer; padding:.42rem 0; color:var(--muted); font-size:.86rem;
                        list-style-position:inside; }
details.stage summary:hover { color:var(--accent); }
details.stage[open] summary { color:var(--ink); font-weight:600; }
.stage-body { padding:.25rem 0 1rem 1.1rem; }
.stage-body p { margin:.3rem 0; }
blockquote { margin:.4rem 0; padding:.25rem .85rem; border-left:2px solid var(--line);
             color:var(--muted); font-style:italic; }
.src { font-size:.83em; color:var(--muted); }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; font-size:.88rem; }
th, td { text-align:left; padding:.28rem .7rem .28rem 0; border-bottom:1px solid var(--hair); white-space:nowrap; }
th { font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); font-weight:600; }
tr.chosen td { font-weight:700; }
.checks { list-style:none; margin:0; padding:0; } .checks li { margin:.3rem 0; }
.checks .ok { color:var(--ink); } .checks .bad { color:var(--accent); font-weight:700; }
.fallback { margin:.5rem 0 .9rem; padding:.55rem .85rem; border:1px solid var(--accent);
            border-radius:4px; font-size:.9rem; }
.fallback ul { margin:.3rem 0 0; }
.d-step { margin:.55rem 0; }
.d-head { display:flex; gap:.55rem; align-items:baseline; flex-wrap:wrap; }
.d-name { font-size:.73rem; text-transform:uppercase; letter-spacing:.07em;
          color:var(--muted); min-width:9em; }
.d-formula { color:var(--muted); }
.d-sub { display:block; margin:.1rem 0 0 3rem; font-weight:600; }
.d-refs { margin-left:3rem; font-size:.8em; color:var(--muted); }
footer { margin-top:3.5rem; color:var(--muted); font-size:.8rem; border-top:1px solid var(--line);
         padding-top:.8rem; }
abbr[title] { text-decoration:underline dotted; text-underline-offset:2px; cursor:help; }
.stage-help { color:var(--muted); font-size:.85rem; margin:.1rem 0 .6rem; max-width:44rem; }
.sum-legend { color:var(--muted); font-size:.78rem; margin:.5rem 0 0; }
.sum-legend b { font-weight:600; color:var(--ink); }
.howto { margin:1.4rem 0 .2rem; border:1px solid var(--line); border-radius:4px;
         background:#fff; }
.howto summary { cursor:pointer; padding:.55rem .9rem; font-weight:600; font-size:.9rem; }
.howto summary:hover { color:var(--accent); }
.howto-body { padding:0 1rem 1rem; font-size:.9rem; max-width:46rem; }
.howto-body dl { margin:.6rem 0 0; }
.howto-body dt { font-weight:700; margin-top:.55rem; }
.howto-body dd { margin:0.1rem 0 0 0; color:var(--muted); }
[title] { cursor:help; }
@media (max-width:900px){ .layout { grid-template-columns:1fr; gap:0; } #toc { position:static;
  max-height:none; border-bottom:1px solid var(--line); padding-bottom:1rem; } }
@media (max-width:640px){ .d-sub,.d-refs{ margin-left:0; } .m-final{ font-size:1.6rem; } }
@media print { #toc { display:none; } .layout { display:block; } .metric { break-inside:avoid; } }
"""

_JS = """
(function () {
  document.documentElement.classList.add('js');
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  // expand / collapse all
  var setAll = function (open) {
    $$('details').forEach(function (d) { d.open = open; });
  };
  var be = $('#expand'), bc = $('#collapse');
  if (be) be.addEventListener('click', function () { setAll(true); });
  if (bc) bc.addEventListener('click', function () { setAll(false); });

  // quick filter over metric sections
  var filter = $('#filter');
  if (filter) filter.addEventListener('input', function () {
    var q = filter.value.trim().toLowerCase();
    $$('.metric').forEach(function (sec) {
      sec.style.display = (!q || (sec.getAttribute('data-filter') || '').indexOf(q) !== -1) ? '' : 'none';
    });
    $$('.company').forEach(function (co) {
      var any = $$('.metric', co).some(function (m) { return m.style.display !== 'none'; });
      co.style.display = any ? '' : 'none';
    });
  });

  // scrollspy: highlight the TOC link of the metric in view
  var links = {};
  $$('.toc-m').forEach(function (a) { links[a.getAttribute('href').slice(1)] = a; });
  if ('IntersectionObserver' in window) {
    var active = null;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          if (active) active.classList.remove('active');
          active = links[en.target.id];
          if (active) active.classList.add('active');
        }
      });
    }, { rootMargin: '-10% 0px -70% 0px' });
    $$('.metric').forEach(function (sec) { io.observe(sec); });
  }

  // open the parent details chain when a hash link targets something inside one
  var openHash = function () {
    var el = location.hash && document.getElementById(location.hash.slice(1));
    var p = el;
    while (p) { if (p.tagName === 'DETAILS') p.open = true; p = p.parentElement; }
  };
  window.addEventListener('hashchange', openHash);
  openHash();
})();
"""


def render_html(report: RunReport, source_name: str = "report.json",
                subtitle: str | None = None) -> str:
    """RunReport -> full self-contained HTML document string. Inline CSS + JS
    only (works over file://); degrades gracefully with JS disabled."""
    title = f"Run report — {report.meta.run_id}" if report.meta.run_id else "Run report"
    if subtitle:
        title = f"{subtitle} — {title}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="layout">
{_toc_html(report.metrics)}
<main>
{_header_html(report.meta, subtitle)}
{_howto_html()}
{_rollup_html(report.metrics)}
<h2 class="sec">Final forecasts</h2>
{_summary_html(report.metrics)}
{_company_sections(report.metrics)}
<footer>Rendered from {_e(source_name)} · Agents vs Wall Street</footer>
</main>
</div>
<script>{_JS}</script>
</body>
</html>"""


def render_report(report_path: Path | str) -> Path:
    """Read report.json, write report.html next to it, plus one per-company
    report-<TICKER>.html for focused browsing. Returns the main html path."""
    report_path = Path(report_path)
    report = RunReport.model_validate_json(report_path.read_text())
    html_path = report_path.with_suffix(".html")
    html_path.write_text(render_html(report, source_name=report_path.name))
    for company, ticker, ms in _group_by_company(report.metrics):
        sub = RunReport(meta=report.meta, metrics=ms)
        co_path = report_path.with_name(f"{report_path.stem}-{ticker}.html")
        co_path.write_text(render_html(sub, source_name=report_path.name,
                                       subtitle=f"{company} ({ticker})"))
    return html_path


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m pipeline.report <path/to/report.json>", file=sys.stderr)
        return 2
    out = render_report(args[0])
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
