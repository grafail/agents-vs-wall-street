"""Unit-conversion helpers + the five CONTEST_UNITS mappings.

Converters are imported from pipeline.extract when the data agent has provided
them (concurrent build), otherwise from pipeline.validate (our copy —
integration dedupes later).
"""
import pytest

from pipeline.types import CONTEST_UNITS, Unit


def _converters():
    """Prefer the data agent's extract.py converters if they exist."""
    try:
        from pipeline import extract  # noqa: F401
        if all(hasattr(extract, n) for n in
               ("billions_to_millions", "pounds_to_pence", "decimal_to_points")):
            return extract
    except ImportError:
        pass
    from pipeline import validate
    return validate


conv = _converters()


# ---------------------------------------------------------------- converters

def test_billions_to_millions():
    assert conv.billions_to_millions(43.2) == pytest.approx(43_200.0)
    assert conv.billions_to_millions(0.0) == 0.0
    assert conv.billions_to_millions(-1.5) == pytest.approx(-1_500.0)


def test_pounds_to_pence():
    # The Hays trap: 0.915 GBP must become 91.5 pence.
    assert conv.pounds_to_pence(0.915) == pytest.approx(91.5)
    assert conv.pounds_to_pence(0.062) == pytest.approx(6.2)


def test_decimal_to_points():
    # 0.045 decimal == 4.5 percentage points.
    assert conv.decimal_to_points(0.045) == pytest.approx(4.5)
    assert conv.decimal_to_points(-0.005) == pytest.approx(-0.5)


def test_to_canonical_raw_units():
    from pipeline import validate
    assert validate.to_canonical(43.2, "USD_billions") == pytest.approx(43_200.0)
    assert validate.to_canonical(43_200.0, "USD_millions") == pytest.approx(43_200.0)
    assert validate.to_canonical(1.2, "GBP_billions") == pytest.approx(1_200.0)
    assert validate.to_canonical(0.915, "GBP_per_share") == pytest.approx(91.5)  # pounds->pence
    assert validate.to_canonical(91.5, "pence") == pytest.approx(91.5)
    assert validate.to_canonical(3.35, "USD_per_share") == pytest.approx(3.35)
    assert validate.to_canonical(0.045, "decimal_fraction") == pytest.approx(4.5)
    assert validate.to_canonical(4.5, "pct_points") == pytest.approx(4.5)
    with pytest.raises(KeyError):
        validate.to_canonical(1.0, "furlongs_per_fortnight")


# ---------------------------------------------------------------- CONTEST_UNITS

def test_contest_units_has_exactly_five():
    assert set(CONTEST_UNITS) == {"USDm", "GBPm", "USD / share", "GBp", "%"}


def test_usdm():
    assert CONTEST_UNITS["USDm"] == Unit(currency="USD", scale="millions", pence=False)


def test_gbpm():
    assert CONTEST_UNITS["GBPm"] == Unit(currency="GBP", scale="millions", pence=False)


def test_usd_per_share():
    assert CONTEST_UNITS["USD / share"] == Unit(currency="USD", scale="per_share", pence=False)


def test_gbp_pence():
    u = CONTEST_UNITS["GBp"]
    assert u == Unit(currency="GBP", scale="per_share", pence=True)
    assert u.pence is True  # Hays EPS is PENCE: 91.5, never 0.915


def test_percent_points():
    u = CONTEST_UNITS["%"]
    assert u == Unit(currency=None, scale="pct_points", pence=False)
    assert u.currency is None
