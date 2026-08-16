"""report.json -> report.html renderer + the authoritative report.json schema.

Two jobs:
1. Pydantic models (`RunReport` and friends) that define the shape of
   logs/run-<ts>/report.json. Integration builds the report dict to conform to
   these models; every field below the metric identity is Optional so skipped
   stages never break serialization or rendering.
2. `render_report(path)` -> path of a single self-contained, JS-FREE HTML file
   (pure HTML/CSS, <details>/<summary> drill-down) written next to the input.
   CLI: `uv run python -m pipeline.report logs/run-.../report.json`

This is the per-run evidence viewer, NOT the judged architecture write-up.
"""
from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path

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
    derivation: list[DerivationStep] | None = None
    final_value: float | None = None


class RunReport(_Model):
    """Top-level shape of logs/run-<ts>/report.json."""
    meta: RunMeta = Field(default_factory=RunMeta)
    metrics: list[MetricReport] = Field(default_factory=list)


# ================================================================ rendering

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
    return d.strftime("%Y-%m-%d %H:%M:%S UTC") if d else "—"


def _status(m: MetricReport) -> tuple[str, str]:
    """(css class, text) for the metric status chip."""
    if m.fallback_used is not None:
        return "fail", f"fallback: {m.fallback_used.source_used}"
    if any(not v.passed for v in m.validation):
        return "warn", "warnings"
    if m.final_value is None:
        return "fail", "no value"
    if not m.validation:
        return "warn", "unvalidated"
    return "pass", "all gates pass"


def _chip(cls: str, text: str) -> str:
    return f'<span class="chip {cls}">{_e(text)}</span>'


_VIBE_CLS = {
    "cold": "fail", "cooling": "warn", "neutral": "neutral",
    "warming": "pass", "hot": "pass",
    "sandbagger": "pass", "accurate": "neutral", "promotional": "warn",
    "downside": "warn", "balanced": "neutral", "upside": "pass",
}


def _grounded_html(name: str, g) -> str:
    if g is None:
        return f'<div class="vibe"><span class="vibe-name">{_e(name)}</span> <span class="muted">—</span></div>'
    cites = "".join(f'<li class="cite">{_e(c)}</li>' for c in (g.citations or []))
    cites_html = f'<ul class="cites">{cites}</ul>' if cites else '<div class="muted small">no citations — treated as neutral</div>'
    cls = _VIBE_CLS.get(str(g.label), "neutral")
    return (
        f'<div class="vibe"><span class="vibe-name">{_e(name)}</span> '
        f'{_chip(cls, str(g.label))}'
        f'<div class="expl">{_e(g.explanation)}</div>{cites_html}</div>'
    )


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
        cls = ' class="chosen"' if (chosen is not None and c.method == chosen) else ""
        star = " ★" if (chosen is not None and c.method == chosen) else ""
        rows.append(
            f"<tr{cls}><td>{_e(c.method)}{star}</td><td>{_num(c.value)}</td>"
            f"<td>{_num(c.backtest_mae)}</td><td>{_num(c.backtest_bias, signed=True)}</td>"
            f"<td>{c.backtest_n if c.backtest_n is not None else '—'}</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>method</th><th>value</th>'
        "<th>backtest MAE</th><th>bias</th><th>n</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _guidance_html(g: GuidanceBlock | None) -> str:
    if g is None:
        return '<p class="muted">No guidance recorded.</p>'
    parts = [
        f"<p>low <b>{_num(g.low)}</b> · mid <b>{_num(g.mid)}</b> · high <b>{_num(g.high)}</b>"
        f' · beat factor <b>{_num(g.beat_factor)}</b></p>'
    ]
    if g.quote:
        parts.append(f"<blockquote>{_e(g.quote)}</blockquote>")
    if g.source_doc:
        parts.append(f'<p class="src">source: <code>{_e(g.source_doc)}</code></p>')
    return "".join(parts)


def _estimate_html(est: Estimate | None) -> str:
    if est is None:
        return '<p class="muted">No blind estimate (estimator skipped or failed).</p>'
    conf_cls = {"low": "warn", "medium": "neutral", "high": "pass"}.get(est.confidence, "neutral")
    return "".join([
        f'<p>method <code>{_e(est.method)}</code> · growth p10 <b>{_num(est.growth_p10, signed=True)}</b>'
        f' / p50 <b>{_num(est.growth_p50, signed=True)}</b> / p90 <b>{_num(est.growth_p90, signed=True)}</b>'
        f" · confidence {_chip(conf_cls, est.confidence)}</p>",
        _grounded_html("momentum", est.momentum),
        _grounded_html("guidance_style", est.guidance_style),
        _grounded_html("surprise_skew", est.surprise_skew),
        f'<p class="rationale"><b>Rationale:</b> {_e(est.rationale)}</p>',
    ])


def _nudges_html(n: NudgeAudit | None) -> str:
    if n is None:
        return '<p class="muted">No nudge audit recorded.</p>'
    parts = []
    if n.components:
        rows = "".join(
            f"<tr><td>{_e(c.name)}</td><td>{_e(c.label) if c.label else '—'}</td>"
            f"<td>{_num(c.raw, signed=True)}</td><td>{_num(c.applied, signed=True)}</td>"
            f"<td>{'capped' if c.capped else '—'}</td><td>{_e(c.note) if c.note else '—'}</td></tr>"
            for c in n.components
        )
        parts.append(
            '<div class="scroll"><table><thead><tr><th>component</th><th>label</th><th>raw</th>'
            f'<th>applied</th><th>cap</th><th>note</th></tr></thead><tbody>{rows}</tbody></table></div>'
        )
    else:
        parts.append('<p class="muted">No nudge components.</p>')
    if n.cap is not None:
        parts.append(f"<p>cap (k×backtest-MAE): <b>{_num(n.cap)}</b></p>")
    if n.caps_applied:
        parts.append("<p>caps applied: " + " ".join(_chip("warn", c) for c in n.caps_applied) + "</p>")
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
        vcls = {"hold": "pass", "partial": "warn", "defer_to_consensus": "fail"}.get(r.verdict, "neutral")
        parts.append(
            f"<p>verdict {_chip(vcls, r.verdict)} · weight on our estimate "
            f"<b>{_num(r.weight_ours)}</b></p><p class='rationale'><b>Rationale:</b> {_e(r.rationale)}</p>"
        )
    else:
        parts.append('<p class="muted">No reconciliation (pure-blind mode or skipped).</p>')
    return "".join(parts)


def _validation_html(checks: list[ValidationCheck]) -> str:
    if not checks:
        return '<p class="muted">No validation checks recorded.</p>'
    items = []
    for v in checks:
        cls = "pass" if v.passed else "fail"
        mark = "✓" if v.passed else "✗"
        detail = f' <span class="muted small">{_e(v.detail)}</span>' if v.detail else ""
        items.append(f'<li>{_chip(cls, f"{mark} {v.check}")}{detail}</li>')
    return f'<ul class="checks">{"".join(items)}</ul>'


def _fallback_html(f: FallbackUsed | None) -> str:
    if f is None:
        return ""
    reasons = "".join(f"<li>{_e(r)}</li>" for r in f.reasons) or "<li class='muted'>no reason recorded</li>"
    return (
        f'<div class="fallback"><b>Failsafe cascade fired</b> — final number came from '
        f"<code>{_e(f.source_used)}</code><ul>{reasons}</ul></div>"
    )


_PROV_LABEL = {"data": "data", "math": "math", "llm": "llm"}


def _derivation_html(steps: list[DerivationStep] | None) -> str:
    if not steps:
        return ""
    lines = []
    for s in steps:
        prov = s.provenance if s.provenance in _PROV_LABEL else "math"
        refs_bits = list(s.refs)
        if s.note:
            refs_bits.append(s.note)
        refs = (f'<div class="d-refs">↳ {_e("; ".join(refs_bits))}</div>'
                if refs_bits else "")
        lines.append(
            f'<div class="d-step"><div class="d-head">'
            f'<span class="prov prov-{prov}">{_e(_PROV_LABEL[prov])}</span>'
            f'<span class="d-name">{_e(s.name)}</span>'
            f'<code class="d-formula">{_e(s.formula)}</code></div>'
            f'<code class="d-sub">{_e(s.substituted)}</code>{refs}</div>'
        )
    return (
        '<div class="deriv"><h4>Derivation</h4>'
        '<p class="d-legend"><span class="prov prov-data">data</span> extracted/fetched value · '
        '<span class="prov prov-math">math</span> deterministic computation · '
        '<span class="prov prov-llm">llm</span> model judgment — '
        'every LLM term is bounded by a computed cap</p>'
        f'{"".join(lines)}</div>'
    )


def _metric_id(m: MetricReport) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in f"{m.ticker}-{m.label}".lower())
    return "m-" + "-".join(p for p in slug.split("-") if p)


def _stage(title: str, body: str) -> str:
    return f'<div class="stage"><h4>{_e(title)}</h4>{body}</div>'


def _metric_details(m: MetricReport) -> str:
    scls, stext = _status(m)
    final = f"{_num(m.final_value)} {_e(m.unit)}" if m.final_value is not None else "—"
    stages = [
        _stage("1 · Anchor", _anchor_html(m.anchor)),
        _stage("2 · Baselines (backtested)", _baseline_html(m.baseline_candidates, m.baseline_chosen)),
        _stage("3 · Guidance & beat factor", _guidance_html(m.guidance)),
        _stage("4 · Blind estimate", _estimate_html(m.estimate)),
        _stage("5 · Nudge audit", _nudges_html(m.nudges)),
        _stage("6 · Consensus & reconciliation", _reconcile_html(m.consensus, m.reconciliation)),
        _stage("7 · Validation gates", _validation_html(m.validation)),
    ]
    final_eq = ""
    if m.derivation:
        last = m.derivation[-1]
        final_eq = f'<div class="final-eq"><code>{_e(last.substituted)}</code></div>'
    return f"""
<details id="{_metric_id(m)}">
<summary><span class="sum-co">{_e(m.company)}</span> <span class="sum-label">{_e(m.label)}</span>
<span class="sum-period">{_e(m.period)}</span> {_chip(scls, stext)}
<span class="sum-final">{final}</span></summary>
<div class="detail-body">
{_fallback_html(m.fallback_used)}
{"".join(stages)}
{_derivation_html(m.derivation)}
<div class="final-box">FINAL VALUE<div class="final-num">{final}</div>{final_eq}</div>
</div>
</details>"""


def _summary_grid(metrics: list[MetricReport]) -> str:
    by_company: dict[str, list[MetricReport]] = {}
    for m in metrics:
        by_company.setdefault(m.company, []).append(m)
    rows = []
    for company, ms in by_company.items():
        cards = []
        for m in ms:
            scls, stext = _status(m)
            final = f"{_num(m.final_value)} <span class='unit'>{_e(m.unit)}</span>" if m.final_value is not None else "—"
            cards.append(
                f'<a class="card" href="#{_metric_id(m)}"><div class="card-label">{_e(m.label)}</div>'
                f'<div class="card-value">{final}</div>{_chip(scls, stext)}</a>'
            )
        rows.append(
            f'<div class="co-row"><div class="co-name">{_e(company)}'
            f'<span class="muted small"> · {_e(ms[0].period)}</span></div>'
            f'<div class="cards">{"".join(cards)}</div></div>'
        )
    return "".join(rows) or '<p class="muted">No metrics in report.</p>'


def _header_html(meta: RunMeta) -> str:
    badges = []
    if meta.enable_reconcile is not None:
        badges.append(_chip("pass" if meta.enable_reconcile else "warn",
                            "reconcile ON" if meta.enable_reconcile else "PURE BLIND"))
    if meta.llm_provider:
        badges.append(_chip("neutral", meta.llm_provider))
    kv = [
        ("run", meta.run_id or "—"),
        ("started", _dt(meta.started)),
        ("finished", _dt(meta.finished)),
        ("model (big)", meta.model_big or "—"),
        ("model (small)", meta.model_small or "—"),
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
    return (
        f'<header><h1>Forecast run report</h1><div class="badges">{"".join(badges)}</div>'
        f'<div class="meta-grid">{kv_html}</div></header>'
    )


_CSS = """
:root { --ink:#1a1e26; --muted:#68707e; --line:#dde1e8; --bg:#f7f8fa; --panel:#ffffff;
        --pass-bg:#e2f3e6; --pass-ink:#1b6b34; --warn-bg:#fdf0d7; --warn-ink:#8a5b00;
        --fail-bg:#fbe3e0; --fail-ink:#a13126; --neutral-bg:#e8ebf1; --neutral-ink:#3d4657; }
* { box-sizing: border-box; }
body { margin:0; padding:2rem clamp(1rem,4vw,3rem); background:var(--bg); color:var(--ink);
       font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
h1 { font-size:1.5rem; margin:0 0 .5rem; }
h2 { font-size:1.1rem; margin:2rem 0 .75rem; border-bottom:1px solid var(--line); padding-bottom:.35rem; }
h4 { margin:0 0 .4rem; font-size:.82rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85em;
       background:var(--neutral-bg); padding:.1em .35em; border-radius:4px; }
.muted { color:var(--muted); } .small { font-size:.85em; }
.chip { display:inline-block; padding:.1em .6em; border-radius:999px; font-size:.78rem; font-weight:600;
        background:var(--neutral-bg); color:var(--neutral-ink); vertical-align:middle; }
.chip.pass { background:var(--pass-bg); color:var(--pass-ink); }
.chip.warn { background:var(--warn-bg); color:var(--warn-ink); }
.chip.fail { background:var(--fail-bg); color:var(--fail-ink); }
header { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:1.25rem 1.5rem; }
.badges { margin:.25rem 0 .75rem; display:flex; gap:.4rem; flex-wrap:wrap; }
.meta-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:.35rem 1.5rem; }
.kv { display:flex; justify-content:space-between; gap:1rem; border-bottom:1px dotted var(--line); padding:.15rem 0; }
.kv .k { color:var(--muted); } .kv .v { font-weight:600; text-align:right; }
.co-row { margin:.9rem 0; }
.co-name { font-weight:700; margin-bottom:.4rem; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:.6rem; }
.card { display:block; background:var(--panel); border:1px solid var(--line); border-radius:10px;
        padding:.7rem .9rem; text-decoration:none; color:inherit; }
.card:hover { border-color:#a9b2c1; }
.card-label { font-size:.82rem; color:var(--muted); min-height:2.2em; }
.card-value { font-size:1.25rem; font-weight:700; margin:.15rem 0 .35rem; }
.card-value .unit { font-size:.7em; color:var(--muted); font-weight:600; }
details { background:var(--panel); border:1px solid var(--line); border-radius:10px; margin:.6rem 0; }
summary { cursor:pointer; padding:.7rem 1rem; display:flex; gap:.7rem; align-items:center; flex-wrap:wrap; }
summary::marker { color:var(--muted); }
.sum-co { font-weight:700; } .sum-label { color:var(--muted); }
.sum-period { font-size:.8rem; color:var(--muted); }
.sum-final { margin-left:auto; font-weight:700; }
.detail-body { padding:0 1.25rem 1.25rem; border-top:1px solid var(--line); }
.stage { margin:1rem 0; padding:.75rem .9rem; background:var(--bg); border:1px solid var(--line); border-radius:8px; }
.stage p { margin:.3rem 0; }
blockquote { margin:.4rem 0; padding:.4rem .8rem; border-left:3px solid #a9b2c1;
             background:var(--neutral-bg); font-style:italic; border-radius:0 6px 6px 0; }
.src { font-size:.85em; color:var(--muted); }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.9rem; background:var(--panel); }
th, td { text-align:left; padding:.35rem .7rem; border-bottom:1px solid var(--line); white-space:nowrap; }
th { font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
tr.chosen td { background:var(--pass-bg); font-weight:600; }
.vibe { margin:.5rem 0; padding:.45rem .6rem; border:1px dashed var(--line); border-radius:6px; background:var(--panel); }
.vibe-name { font-weight:600; margin-right:.4rem; }
.expl { margin-top:.25rem; }
.cites { margin:.25rem 0 0; padding-left:1.2rem; }
.cite { font-size:.85em; color:var(--muted); }
.checks { list-style:none; margin:0; padding:0; } .checks li { margin:.3rem 0; }
.rationale { margin-top:.5rem; }
.fallback { margin:1rem 0; padding:.7rem .9rem; background:var(--fail-bg); color:var(--fail-ink);
            border:1px solid #e5b3ac; border-radius:8px; }
.fallback ul { margin:.3rem 0 0; }
.final-box { margin-top:1rem; padding:.8rem 1rem; border:2px solid var(--ink); border-radius:10px;
             font-size:.75rem; letter-spacing:.08em; font-weight:700; }
.final-num { font-size:1.8rem; letter-spacing:0; margin-top:.1rem; }
.final-eq { margin-top:.4rem; letter-spacing:0; font-weight:400; }
.final-eq code { background:transparent; padding:0; font-size:.95rem; }
.deriv { margin:1rem 0; padding:.75rem .9rem; background:var(--bg);
         border:1px solid var(--line); border-left:4px solid var(--ink); border-radius:8px; }
.d-legend { margin:.2rem 0 .7rem; font-size:.82rem; color:var(--muted); }
.d-step { margin:.55rem 0; }
.d-head { display:flex; gap:.55rem; align-items:baseline; flex-wrap:wrap; }
.d-name { font-size:.75rem; text-transform:uppercase; letter-spacing:.06em;
          color:var(--muted); min-width:7.5em; }
.d-formula { background:transparent; padding:0; color:var(--muted); }
.d-sub { display:block; margin:.1rem 0 0 calc(7.5em + 2.4rem + .55rem);
         background:transparent; padding:0; font-size:.95em; font-weight:600; }
.d-refs { margin-left:calc(7.5em + 2.4rem + .55rem); font-size:.8em; color:var(--muted); }
.prov { display:inline-block; min-width:2.4rem; text-align:center; padding:.05em .45em;
        border-radius:5px; font-size:.68rem; font-weight:700; text-transform:uppercase;
        letter-spacing:.05em; }
.prov-data { background:var(--neutral-bg); color:var(--neutral-ink); }
.prov-math { background:#dbe7fb; color:#1d4f9c; }
.prov-llm  { background:#ecdff7; color:#6b2fa3; }
@media (max-width:640px){ .d-sub,.d-refs{ margin-left:0; } }
footer { margin-top:2rem; color:var(--muted); font-size:.8rem; }
@media print { body { background:#fff; } details { break-inside:avoid; } .card:hover { border-color:var(--line); } }
"""


def render_html(report: RunReport, source_name: str = "report.json") -> str:
    """RunReport -> full self-contained HTML document string (no scripts)."""
    title = f"Run report — {report.meta.run_id}" if report.meta.run_id else "Run report"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>{_CSS}</style>
</head>
<body>
{_header_html(report.meta)}
<h2>Summary — final forecasts</h2>
{_summary_grid(report.metrics)}
<h2>Per-metric audit trail</h2>
{"".join(_metric_details(m) for m in report.metrics)}
<footer>Rendered from {_e(source_name)} · pure HTML/CSS, no scripts · Agents vs Wall Street</footer>
</body>
</html>"""


def render_report(report_path: Path | str) -> Path:
    """Read report.json, write report.html next to it. Returns the html path."""
    report_path = Path(report_path)
    report = RunReport.model_validate_json(report_path.read_text())
    html_path = report_path.with_suffix(".html")
    html_path.write_text(render_html(report, source_name=report_path.name))
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
