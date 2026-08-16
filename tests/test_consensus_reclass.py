"""Blind-integrity: consensus-attributed figures must never reach the estimator."""
from pipeline.extract import ExtractedFact
from pipeline.graph import _consensus_facts, _guidance_facts, _reclassify_consensus
from pipeline.types import SourceRef

SRC = SourceRef(doc_id="hays/filings/x.md", trust_tier=1, kind="filing")


def _fact(fact_type: str, quote: str, value: float = 43.5) -> ExtractedFact:
    return ExtractedFact(
        metric_label="Pre-exceptional operating profit", period="FY2026",
        value=value, raw_text=str(value), raw_unit="GBP_millions",
        quote=quote, source=SRC, fact_type=fact_type, flags=[])


def test_compiled_consensus_reclassified():
    facts = _reclassify_consensus([
        _fact("guidance_mid", "company compiled consensus for FY26 is £43.5m"),
    ])
    assert facts[0].fact_type == "consensus_mid"
    assert _consensus_facts(facts, facts[0].metric_label)
    assert not _guidance_facts(facts, facts[0].metric_label)


def test_company_expectation_relative_to_consensus_stays_guidance():
    facts = _reclassify_consensus([
        _fact("guidance_mid",
              "we currently expect FY26 profit at the top of the consensus range", 45.2),
    ])
    assert facts[0].fact_type == "guidance_mid"


def test_plain_guidance_untouched():
    facts = _reclassify_consensus([
        _fact("guidance_low", "we expect operating profit of £37.0m to £46.0m", 37.0),
        _fact("actual", "pre-exceptional operating profit was £45.6m", 45.6),
    ])
    assert [f.fact_type for f in facts] == ["guidance_low", "actual"]
