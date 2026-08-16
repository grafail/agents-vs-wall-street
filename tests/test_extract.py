"""Extraction tests: real-corpus narrowing, unit normalization (incl. the
billions/pence/percent traps), mechanical validation, mocked-LLM end-to-end.
No network, no API key."""
import json
from pathlib import Path

import pytest

from pipeline import extract
from pipeline.config import OFFLINE_DATA
from pipeline.extract import (
    Candidate, ExcerptParse, ExtractedFact, ParsedFact, apply_magnitude_gate,
    canonical_period, collect_candidates, normalize_value, parse_number,
    value_in_quote, _facts_from_parse,
)
from pipeline.types import CONTEST_UNITS, metric_specs

HD_Q2_8K = OFFLINE_DATA / "home-depot/filings/2025-08-19__hd-us-20250819-q2-8k__143666.md"


def spec_for(ticker: str, label: str):
    return next(s for s in metric_specs() if s.ticker == ticker and s.label == label)


# ---------------------------------------------------------------- narrowing (real corpus doc)

@pytest.mark.skipif(not HD_Q2_8K.exists(), reason="corpus doc missing")
def test_narrowing_finds_net_sales_in_real_hd_8k():
    spec = spec_for("HD", "Net sales")
    cands = collect_candidates(spec, doc_paths=[HD_Q2_8K])
    assert cands, "no candidates found in a real HD earnings release"
    joined = "\n".join(c.excerpt for c in cands)
    assert "45,277" in joined  # the Q2 FY2025 net sales figure
    c = cands[0]
    assert c.doc_type == "FILING"
    assert c.published == "2025-08-19"
    assert c.doc_id.startswith("home-depot/filings/")


@pytest.mark.skipif(not HD_Q2_8K.exists(), reason="corpus doc missing")
def test_narrowing_finds_comparable_sales_and_guidance():
    spec = spec_for("HD", "Comparable sales, total company")
    cands = collect_candidates(spec, doc_paths=[HD_Q2_8K])
    joined = "\n".join(c.excerpt for c in cands)
    assert "Comparable sales" in joined
    # FY guidance bullet lives in the same 8-K and must be reachable
    assert "approximately 1.0%" in joined or "1.0" in joined


HAYS_FY25_PRELIM = OFFLINE_DATA / "hays/filings/2025-08-21__has-ln-20250821-filing__143845.md"


@pytest.mark.skipif(not HAYS_FY25_PRELIM.exists(), reason="corpus doc missing")
def test_narrowing_reaches_hays_results_through_rns_noise():
    # Hays' filing stream is dominated by administrative RNS (director dealings,
    # AGM notices); the FY2025 preliminary report is named "...-filing", not
    # "-8k". Regression guard: a full-corpus walk must still surface it — this
    # breaks if max_docs is tuned back down or the strong-number gate weakens.
    spec = spec_for("HAS", "Pre-exceptional basic EPS")
    cands = collect_candidates(spec)
    assert cands, "no HAS EPS candidates found in the Hays corpus"
    prelim = [c for c in cands if "2025-08-21" in c.doc_id]
    assert prelim, "FY2025 preliminary report not reached by narrowing"
    assert any("1.31p" in c.excerpt for c in prelim)  # the FY2025 pre-exceptional EPS


def test_narrowing_requires_strong_numbers():
    # bare years/integers must NOT qualify a line; real figures must
    assert not extract._STRONG_NUM_RE.search("the Reform Act of 1995 on net sales growth")
    assert extract._STRONG_NUM_RE.search("Net sales were $9 million.")
    assert extract._STRONG_NUM_RE.search("Net sales | 45,277 |")
    assert extract._STRONG_NUM_RE.search("comparable sales increased 1.0%")


# ---------------------------------------------------------------- periods

@pytest.mark.parametrize("raw,expected", [
    ("Q2 2025", "FY2025Q2"),
    ("FY 2025", "FY2025"),
    ("FY2026Q2", "FY2026Q2"),
    ("H1 2025", "FY2025H1"),
    ("2025 Q3", "FY2025Q3"),
    ("FY26", "FY2026"),
])
def test_canonical_period(raw, expected):
    assert canonical_period(raw) == expected


# ---------------------------------------------------------------- normalization

USDm = CONTEST_UNITS["USDm"]
GBp = CONTEST_UNITS["GBp"]
PCT = CONTEST_UNITS["%"]
USD_SHARE = CONTEST_UNITS["USD / share"]


def test_parse_number():
    assert parse_number("$45,277") == 45277.0
    assert parse_number("(0.49)p") == -0.49
    assert parse_number("69.2 %") == 69.2
    assert parse_number("±$100 million") == 100.0
    assert parse_number("$45.3 billion") == 45.3


def test_billions_to_millions():
    v, flags, reason = normalize_value(USDm, "$45.3 billion", "USD_billions")
    assert reason is None and v == pytest.approx(45300.0)


def test_millions_pass_through():
    v, _, reason = normalize_value(USDm, "45,277", "USD_millions")
    assert reason is None and v == 45277.0


def test_pence_trap():
    # already pence -> unchanged; pounds -> x100
    v, _, r = normalize_value(GBp, "1.31p", "GBP_pence_per_share")
    assert r is None and v == pytest.approx(1.31)
    v2, flags, r2 = normalize_value(GBp, "£0.0131", "GBP_pounds_per_share")
    assert r2 is None and v2 == pytest.approx(1.31)
    assert "pounds_to_pence" in flags


def test_percent_trap():
    v, _, r = normalize_value(PCT, "4.5", "pct_points")
    assert r is None and v == 4.5
    v2, flags, r2 = normalize_value(PCT, "0.045", "pct_decimal")
    assert r2 is None and v2 == pytest.approx(4.5)
    assert "decimal_to_points" in flags
    v3, _, r3 = normalize_value(PCT, "130", "bps")
    assert r3 is None and v3 == pytest.approx(1.3)


def test_currency_mismatch_skipped():
    v, _, reason = normalize_value(USDm, "£972.4", "GBP_millions")
    assert v is None and "currency mismatch" in reason


def test_adi_dollar_line_cannot_pollute_pct_metric():
    # ADI trap: "Adjusted gross margin" is a $ table line; the contest metric is
    # the "... percentage" line. Even if the LLM extracts the dollar figure, the
    # unit gate must skip it rather than emit 1995 "percent".
    v, _, reason = normalize_value(PCT, "1,995", "USD_millions")
    assert v is None and "incompatible with pct metric" in reason


def test_growth_guidance_needs_base():
    # "Adjusted diluted EPS to decline approximately 2% from $15.24"
    v, flags, r = normalize_value(USD_SHARE, "-2", "pct_growth",
                                  base_as_written="$15.24", base_raw_unit="USD_per_share")
    assert r is None and v == pytest.approx(15.24 * 0.98)
    assert "derived_from_growth" in flags
    v2, _, r2 = normalize_value(USD_SHARE, "-2", "pct_growth")
    assert v2 is None and "base" in r2


# ---------------------------------------------------------------- mechanical validation

def _cand(excerpt: str) -> Candidate:
    return Candidate(doc_id="home-depot/filings/x.md", doc_type="FILING",
                     published="2025-08-19", period_hint="Q2 2025", excerpt=excerpt)


def _pf(**kw) -> ParsedFact:
    base = dict(excerpt_index=0, period="FY2025Q2", fact_type="actual",
                value_as_written="45,277", raw_unit="USD_millions",
                quote="Net sales $ 45,277")
    base.update(kw)
    return ParsedFact(**base)


def test_value_in_quote():
    assert value_in_quote("45,277", "Net sales $ 45,277 for the quarter")
    assert value_in_quote("$45.3 billion", "reported sales of $45.3 billion")
    assert not value_in_quote("44,000", "Net sales $ 45,277")


def test_quote_mismatch_rejected():
    spec = spec_for("HD", "Net sales")
    parse = ExcerptParse(facts=[_pf(value_as_written="44,000")])  # not in quote
    facts, skipped = _facts_from_parse(spec, parse, [_cand("Net sales $ 45,277")])
    assert facts == []
    assert skipped and "hallucination" in skipped[0]["reason"]


def test_valid_fact_accepted_with_source():
    spec = spec_for("HD", "Net sales")
    parse = ExcerptParse(facts=[_pf()])
    facts, skipped = _facts_from_parse(spec, parse, [_cand("Net sales $ 45,277 blah")])
    assert skipped == []
    (f,) = facts
    assert f.value == 45277.0 and f.period == "FY2025Q2"
    assert f.source.trust_tier == 1 and f.source.kind == "filing"
    assert f.source.doc_id == "home-depot/filings/x.md"
    assert "quote_not_in_excerpt" not in f.flags


def test_spread_expands_to_low_high():
    spec = spec_for("ADI", "Revenue")
    quote = "we are forecasting revenue of $3.0 billion, +/- $100 million"
    parse = ExcerptParse(facts=[_pf(
        fact_type="guidance_mid", value_as_written="$3.0 billion", raw_unit="USD_billions",
        spread_as_written="$100 million", spread_raw_unit="USD_millions",
        period="FY2025Q4", quote=quote)])
    facts, skipped = _facts_from_parse(spec, parse, [_cand(quote)])
    assert skipped == []
    by_type = {f.fact_type: f.value for f in facts}
    assert by_type == {"guidance_mid": 3000.0, "guidance_low": 2900.0, "guidance_high": 3100.0}


def test_magnitude_gate_flags_outlier():
    spec = spec_for("HD", "Net sales")
    cand = _cand("q")
    facts = []
    for vw in ("40,000", "41,000", "42,000", "4,100"):  # last one 10x off (billions typo)
        parse = ExcerptParse(facts=[_pf(value_as_written=vw, quote=f"Net sales $ {vw}")])
        fs, _ = _facts_from_parse(spec, parse, [cand])
        facts += fs
    apply_magnitude_gate(facts)
    flagged = [f for f in facts if "magnitude_outlier" in f.flags]
    assert len(flagged) == 1 and flagged[0].value == 4100.0


def test_magnitude_gate_lenient_for_pct():
    # comp sales legitimately swings sign/scale: -3.3 vs 1.0 must NOT be flagged
    facts = []
    spec = spec_for("HD", "Comparable sales, total company")
    for vw in ("1.0", "(3.3)", "0.4"):
        parse = ExcerptParse(facts=[_pf(value_as_written=vw, raw_unit="pct_points",
                                        quote=f"Comparable sales {vw} %")])
        fs, _ = _facts_from_parse(spec, parse, [_cand("q")])
        facts += fs
    apply_magnitude_gate(facts)
    assert all("magnitude_outlier" not in f.flags for f in facts)


# ---------------------------------------------------------------- end-to-end, LLM mocked

@pytest.mark.skipif(not HD_Q2_8K.exists(), reason="corpus doc missing")
def test_extract_company_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "FACTS_DIR", tmp_path / "facts")
    # keep the run fast + deterministic: narrow over one real 8-K only
    monkeypatch.setattr(
        extract, "corpus_docs",
        lambda corpus_dir, include=("filings",), max_docs=60: [HD_Q2_8K])

    def fake_llm_parse(spec, candidates):
        facts = []
        if spec.label == "Net sales":
            # only claim the figure when the excerpt really contains it
            for i, c in enumerate(candidates):
                if "45,277" in c.excerpt:
                    facts.append(ParsedFact(
                        excerpt_index=i, period="FY2025Q2", fact_type="actual",
                        value_as_written="45,277", raw_unit="USD_millions",
                        quote="Net sales $ 45,277"))
                    break
        usage = {"model": "mock", "prompt_tokens": 10, "completion_tokens": 5,
                 "cached_prompt_tokens": 0}
        return ExcerptParse(facts=facts), usage

    monkeypatch.setattr(extract, "_llm_parse", fake_llm_parse)

    path = extract.extract_company("HD", max_docs=6)
    assert path == tmp_path / "facts" / "HD.json"
    payload = json.loads(path.read_text())
    assert payload["ticker"] == "HD" and payload["target_period"] == "FY2026Q2"
    assert payload["facts"], "no facts extracted end-to-end"
    # round-trips through the pydantic model
    fact = ExtractedFact.model_validate(payload["facts"][0])
    assert fact.metric_label == "Net sales"
    assert fact.value == 45277.0
    assert fact.period == "FY2025Q2"
    assert fact.source.trust_tier == 1
    # duplicate figures across excerpts/docs were deduped to one fact
    keys = [(f["metric_label"], f["period"], f["fact_type"], f["value"])
            for f in payload["facts"]]
    assert len(keys) == len(set(keys))
    # load_facts helper reads the artifact back
    assert extract.load_facts("HD")[0].value == 45277.0


def test_extract_company_unknown_ticker():
    with pytest.raises(ValueError):
        extract.extract_company("XYZ")


def test_cli_help_parses(capsys):
    with pytest.raises(SystemExit) as e:
        extract.main(["--help"])
    assert e.value.code == 0
    assert "--company" in capsys.readouterr().out
