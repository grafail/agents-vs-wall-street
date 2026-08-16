"""Tool-belt tests: cache read-through + error paths. No network, no API key."""
import pandas as pd
import pytest

from pipeline import cache, tools


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    """Isolate the read-through cache per test."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")


# ---------------------------------------------------------------- search_web

DEFAULT_RESULTS = [{"title": "HD 8-K", "href": "https://www.sec.gov/x/8k.htm", "body": "sales"}]


class FakeDDGS:
    calls = 0
    results = DEFAULT_RESULTS
    fail = False

    def text(self, query, max_results=8):
        FakeDDGS.calls += 1
        if FakeDDGS.fail:
            raise RuntimeError("ddg down")
        return list(self.results)

    def news(self, query, timelimit=None, max_results=8):
        FakeDDGS.calls += 1
        FakeDDGS.last_timelimit = timelimit
        if FakeDDGS.fail:
            raise RuntimeError("ddg down")
        return [{"title": "n", "url": "https://example.com/a", "body": "b",
                 "source": "Wire", "date": "2026-08-15"}]


@pytest.fixture(autouse=True)
def fake_ddgs(monkeypatch):
    FakeDDGS.calls = 0
    FakeDDGS.fail = False
    FakeDDGS.results = DEFAULT_RESULTS
    monkeypatch.setattr(tools, "DDGS", FakeDDGS)


def test_search_web_read_through():
    data, hit = tools.search_web("home depot 8-K")
    assert hit is False and FakeDDGS.calls == 1
    assert data["results"][0]["url"] == "https://www.sec.gov/x/8k.htm"
    # sec.gov is clearly official -> tier 1; SourceRef-compatible metadata present
    src = data["results"][0]["source"]
    assert src["trust_tier"] == 1
    assert {"doc_id", "fetched_at", "trust_tier", "kind"} <= src.keys()

    data2, hit2 = tools.search_web("home depot 8-K")
    assert hit2 is True and FakeDDGS.calls == 1  # served from cache
    assert data2["results"] == data["results"]


def test_search_web_error_not_cached():
    FakeDDGS.fail = True
    data, hit = tools.search_web("boom")
    assert hit is False and "error" in data
    # error was not cached: once the backend recovers, we go live again
    FakeDDGS.fail = False
    data2, hit2 = tools.search_web("boom")
    assert hit2 is False and "results" in data2


def test_search_web_trust_tier_default_4():
    FakeDDGS.results = [{"title": "blog", "href": "https://randomblog.io/p", "body": "x"}]
    data, _ = tools.search_web("blog take")
    assert data["results"][0]["source"]["trust_tier"] == 4


@pytest.mark.parametrize("days,expected", [(1, "d"), (7, "w"), (30, "m")])
def test_search_news_timelimit_and_tier(days, expected):
    data, hit = tools.search_news("ADI earnings", days=days)
    assert hit is False
    assert FakeDDGS.last_timelimit == expected
    assert data["timelimit"] == expected
    # news results are LEADS: always tier 4
    assert all(r["source"]["trust_tier"] == 4 for r in data["results"])


# ---------------------------------------------------------------- fetch_url

class FakeResp:
    def __init__(self, text="<html>hi</html>", status=200):
        self.text = text
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


def test_fetch_url_extracts_and_caps(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, timeout=None, headers=None):
        calls["n"] += 1
        assert timeout == 10
        assert "Mozilla" in headers["User-Agent"]
        return FakeResp()

    monkeypatch.setattr(tools.requests, "get", fake_get)
    monkeypatch.setattr(tools.trafilatura, "extract", lambda html: "clean " * 20_000)

    data, hit = tools.fetch_url("https://www.sec.gov/doc.htm")
    assert hit is False and calls["n"] == 1
    assert len(data["text"]) == tools.FETCH_TEXT_CAP and data["truncated"] is True
    assert data["source"]["trust_tier"] == 1

    _, hit2 = tools.fetch_url("https://www.sec.gov/doc.htm")
    assert hit2 is True and calls["n"] == 1


def test_fetch_url_error(monkeypatch):
    def fake_get(url, timeout=None, headers=None):
        raise TimeoutError("connect timeout")

    monkeypatch.setattr(tools.requests, "get", fake_get)
    data, hit = tools.fetch_url("https://nowhere.example/x")
    assert hit is False
    assert "error" in data and "TimeoutError" in data["error"]


# ---------------------------------------------------------------- get_market_data

class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.info = {"shortName": "Home Depot", "marketCap": 350_000_000_000}
        self.quarterly_financials = pd.DataFrame(
            {pd.Timestamp("2025-08-03"): [45277.0, 4551.0],
             pd.Timestamp("2025-05-04"): [39856.0, 3433.0]},
            index=["Total Revenue", "Net Income"])
        self.news = [{"title": "HD beats", "link": "https://news.example/hd"}]


@pytest.fixture
def fake_yf(monkeypatch):
    monkeypatch.setattr(tools.yf, "Ticker", FakeTicker)
    delays = {"n": 0}
    monkeypatch.setattr(tools, "_polite_delay", lambda: delays.__setitem__("n", delays["n"] + 1))
    return delays


def test_get_market_data_quarterly_financials(fake_yf):
    data, hit = tools.get_market_data("HD", "quarterly_financials")
    assert hit is False and fake_yf["n"] == 1  # politeness delay on live fetch
    cols = data["data"]
    assert set(cols) == {"2025-08-03 00:00:00", "2025-05-04 00:00:00"}  # stringified
    assert cols["2025-08-03 00:00:00"]["Total Revenue"] == 45277.0
    assert data["source"]["kind"] == "market_data"
    assert data["source"]["trust_tier"] == 3

    _, hit2 = tools.get_market_data("HD", "quarterly_financials")
    assert hit2 is True and fake_yf["n"] == 1  # no delay on cache hit


def test_get_market_data_bad_what(fake_yf):
    data, hit = tools.get_market_data("HD", "bogus")
    assert hit is False and "error" in data and fake_yf["n"] == 0


def test_get_market_data_error_not_raised(monkeypatch):
    def boom(symbol):
        raise ConnectionError("no dns")

    monkeypatch.setattr(tools.yf, "Ticker", boom)
    monkeypatch.setattr(tools, "_polite_delay", lambda: None)
    data, hit = tools.get_market_data("HD", "info")
    assert hit is False and "error" in data


# ---------------------------------------------------------------- helpers

def test_trust_tier_for_url():
    assert tools.trust_tier_for_url("https://www.sec.gov/Archives/x.htm") == 1
    assert tools.trust_tier_for_url("https://www.londonstockexchange.com/news/1") == 1
    assert tools.trust_tier_for_url("https://ir.homedepot.com/press") == 2
    assert tools.trust_tier_for_url("https://seekingalpha.com/article") == 4


def test_to_plain_nan_and_nested():
    df = pd.DataFrame({pd.Timestamp("2025-01-01"): [1.0, float("nan")]}, index=["a", "b"])
    out = tools._to_plain({"df": df, "t": (1, "x")})
    assert out["df"]["2025-01-01 00:00:00"] == {"a": 1.0, "b": None}
    assert out["t"] == [1, "x"]
