"""auto_correct_units: unambiguous scale slips fixed in place, ambiguous only flagged."""
from pipeline.extract import ExtractedFact, auto_correct_units
from pipeline.types import SourceRef

SRC = SourceRef(doc_id="x.md", trust_tier=1, kind="filing")


def _f(value, label="Pre-exceptional basic EPS", unit="GBP_pence_per_share"):
    return ExtractedFact(metric_label=label, period="FY2024", value=value,
                         raw_text=str(value), raw_unit=unit, quote="q",
                         source=SRC, fact_type="actual", flags=[])


def test_pounds_slip_corrected_x100():
    facts = [_f(9.22), _f(8.59), _f(4.03), _f(0.0403)]  # last one is pounds
    auto_correct_units(facts)
    assert facts[-1].value == 4.03
    assert any(fl.startswith("auto_corrected_scale_x100") for fl in facts[-1].flags)


def test_billions_slip_corrected_x1000():
    facts = [_f(v, "Net sales", "USD_millions") for v in (43000, 45000, 44000)] + \
            [_f(43.2, "Net sales", "USD_millions")]
    auto_correct_units(facts)
    assert facts[-1].value == 43200.0


def test_legit_small_value_untouched():
    # Hays EPS collapse: 1.31 vs median ~4 is within the 0.2x-5x no-touch zone
    facts = [_f(9.22), _f(8.59), _f(4.03), _f(1.31)]
    auto_correct_units(facts)
    assert facts[-1].value == 1.31
    assert not facts[-1].flags


def test_ambiguous_stays_flagged_not_guessed():
    # 0.9 vs median ~4000: x100 -> 90 (out of band), x1000 -> 900 (out of band)
    # -> no single fit, flag only
    facts = [_f(v, "Net sales", "USD_millions") for v in (4000, 4100, 3900)] + \
            [_f(0.9, "Net sales", "USD_millions")]
    auto_correct_units(facts)
    assert facts[-1].value == 0.9
    assert "magnitude_outlier" in facts[-1].flags


def test_percent_metrics_skipped():
    facts = [_f(v, "Comparable sales, total company", "pct_points") for v in (1.0, -3.3, 0.004)]
    auto_correct_units(facts)
    assert [f.value for f in facts] == [1.0, -3.3, 0.004]
