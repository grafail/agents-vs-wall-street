"""Core pydantic models shared by every pipeline stage.

Contracts, not conveniences: subagents build against these. Extend cautiously;
never rename existing fields without checking every consumer.
"""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- units

Scale = Literal["millions", "per_share", "pct_points"]


class Unit(BaseModel):
    """Canonical unit for a metric. The five contest units map to:
    USDm      -> currency=USD scale=millions
    GBPm      -> currency=GBP scale=millions
    USD/share -> currency=USD scale=per_share
    GBp       -> currency=GBP scale=per_share pence=True   (PENCE, e.g. 91.5)
    %         -> currency=None scale=pct_points            (points: 4.5 means 4.5%)
    """
    currency: Literal["USD", "GBP"] | None = None
    scale: Scale
    pence: bool = False


CONTEST_UNITS: dict[str, Unit] = {
    "USDm": Unit(currency="USD", scale="millions"),
    "GBPm": Unit(currency="GBP", scale="millions"),
    "USD / share": Unit(currency="USD", scale="per_share"),
    "GBp": Unit(currency="GBP", scale="per_share", pence=True),
    "%": Unit(scale="pct_points"),
}

# ---------------------------------------------------------------- metric spec

Kind = Literal["flow_absolute", "per_share", "ratio_pct"]
Basis = Literal["gaap", "adjusted", "pre_exceptional", "as_reported"]


class MetricSpec(BaseModel):
    """Generic typing that drives ALL pipeline logic. Exact `label` strings are
    used only at extraction (search terms) and workbook write (verbatim)."""
    company: str
    ticker: str            # HD | ADI | HAS | DE  (short selector)
    corpus_dir: str        # folder under challenge/offline-data/
    period: str            # e.g. FY2026Q2 (contest column header, verbatim)
    output_file: str       # e.g. HD-FY2026Q2.xlsx
    label: str             # exact contest metric label, verbatim
    unit_str: str          # exact contest unit string, verbatim
    unit: Unit
    kind: Kind
    basis: Basis
    scope: Literal["group", "segment"] = "group"
    period_type: Literal["quarter", "fiscal_year"] = "quarter"
    derivation: Literal["direct", "derived"] = "direct"


def metric_specs() -> list[MetricSpec]:
    """The 12 contest records, typed. Source of truth for labels/units/files is
    challenge/companies.json; typing added here."""
    from pipeline.config import load_companies

    typing: dict[tuple[str, str], dict] = {
        ("HD", "Net sales"): dict(kind="flow_absolute", basis="as_reported"),
        ("HD", "Adjusted diluted EPS"): dict(kind="per_share", basis="adjusted"),
        ("HD", "Comparable sales, total company"): dict(kind="ratio_pct", basis="as_reported"),
        ("ADI", "Revenue"): dict(kind="flow_absolute", basis="as_reported"),
        ("ADI", "Adjusted diluted EPS"): dict(kind="per_share", basis="adjusted"),
        ("ADI", "Adjusted gross margin"): dict(kind="ratio_pct", basis="adjusted"),
        ("HAS", "Net fees"): dict(kind="flow_absolute", basis="as_reported", period_type="fiscal_year"),
        ("HAS", "Pre-exceptional basic EPS"): dict(kind="per_share", basis="pre_exceptional", period_type="fiscal_year"),
        ("HAS", "Pre-exceptional operating profit"): dict(kind="flow_absolute", basis="pre_exceptional", period_type="fiscal_year"),
        ("DE", "Worldwide net sales and revenues"): dict(kind="flow_absolute", basis="as_reported"),
        ("DE", "Diluted EPS (GAAP)"): dict(kind="per_share", basis="gaap"),
        ("DE", "Production & Precision Ag operating profit"): dict(kind="flow_absolute", basis="as_reported", scope="segment"),
    }
    corpus_dirs = {"HD": "home-depot", "ADI": "analog-devices", "LSE:HAS": "hays", "DE": "deere"}
    short = {"HD": "HD", "ADI": "ADI", "LSE:HAS": "HAS", "DE": "DE"}

    specs: list[MetricSpec] = []
    for co in load_companies():
        tick = short[co["ticker"]]
        for m in co["metrics"]:
            t = typing[(tick, m["label"])]
            specs.append(MetricSpec(
                company=co["company"], ticker=tick,
                corpus_dir=corpus_dirs[co["ticker"]],
                period=co["period"], output_file=co["outputFile"],
                label=m["label"], unit_str=m["units"],
                unit=CONTEST_UNITS[m["units"]], **t,
            ))
    return specs

# ---------------------------------------------------------------- sources & facts

TrustTier = Literal[1, 2, 3, 4]  # 1 corpus/SEC/RNS · 2 IR/official macro · 3 aggregators/peer filings · 4 news (leads only)
SourceKind = Literal["filing", "transcript", "slides", "ir_page", "consensus_aggregator", "news", "macro_release", "market_data"]


class SourceRef(BaseModel):
    doc_id: str                       # corpus-relative path or URL
    published: date | None = None
    fetched_at: datetime | None = None
    trust_tier: TrustTier
    kind: SourceKind


class Fact(BaseModel):
    """One extracted figure. `value` is ALWAYS in the metric's canonical unit;
    raw_text/raw_unit preserve what the document said (auditable conversion)."""
    metric_label: str
    period: str                       # e.g. FY2025Q2 / FY2025
    value: float                      # canonical unit
    raw_text: str                     # value as written, e.g. "$43.2 billion"
    raw_unit: str                     # e.g. "USD_billions"
    quote: str                        # the source line; extraction must string-match raw value inside it
    source: SourceRef
    fact_type: Literal["actual", "guidance_low", "guidance_mid", "guidance_high",
                       "consensus_low", "consensus_mid", "consensus_high"] = "actual"
    # consensus_* = figures a document attributes to analyst/market/compiled consensus.
    # These are NEVER shown to the blind estimator; they feed the reconciler.

# ---------------------------------------------------------------- LLM judgment outputs

VibeLabel = Literal["cold", "cooling", "neutral", "warming", "hot"]
GuidanceStyle = Literal["sandbagger", "accurate", "promotional"]
SurpriseSkew = Literal["downside", "balanced", "upside"]


class Grounded[T](BaseModel):
    """A categorical that must carry its grounding. explanation comes FIRST so the
    model reasons before it classifies. Empty citations => treated as neutral/zero nudge."""
    explanation: str
    label: T
    citations: list[str] = Field(default_factory=list)


class Estimate(BaseModel):
    """Blind estimator output for one metric. NO absolute forecast here —
    code computes the number from anchor + these bounded assumptions."""
    method: str                        # chosen baseline method id
    growth_p10: float                  # relative deltas, in pct (flow/per_share) or points (ratio_pct)
    growth_p50: float
    growth_p90: float
    momentum: Grounded[VibeLabel]
    guidance_style: Grounded[GuidanceStyle]
    surprise_skew: Grounded[SurpriseSkew]
    confidence: Literal["low", "medium", "high"]
    rationale: str


class MetricEstimate(Estimate):
    """Estimate tagged with its contest metric label — one entry of a
    company-level estimation response. Estimate itself stays unchanged for
    downstream compatibility (nudges/report consume plain Estimate)."""
    metric_label: str


class CompanyEstimates(BaseModel):
    """Blind estimator output for ALL of one company's metrics in a single
    call, so cross-metric coherence (e.g. operating profit vs EPS) is reasoned
    about explicitly. coherence_rationale comes FIRST so the model reasons
    across metrics before committing to per-metric numbers."""
    coherence_rationale: str
    estimates: list[MetricEstimate]


class Reconciliation(BaseModel):
    """Reconciler output: weight on our estimate vs consensus, with rationale."""
    verdict: Literal["hold", "partial", "defer_to_consensus"]
    weight_ours: float = Field(ge=0.0, le=1.0)
    rationale: str
