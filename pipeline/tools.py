"""Cached tool belt for external data: web search, news search, URL fetch, market data.

Every function goes through cache.read_through (pre-baking == warming that cache)
and returns (data, was_cache_hit). Failures NEVER raise: they return
({"error": ...}, False) and are NOT cached, so a later retry goes live again.
Each successful payload carries SourceRef-compatible metadata
(doc_id/url, published if known, fetched_at, trust_tier, kind).

Trust tiers (see pipeline.types.TrustTier): 1 SEC/RNS · 2 company IR/official ·
3 aggregators/market data · 4 general web/news (LEADS only, verify elsewhere).
"""
import time
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlparse

import requests
import trafilatura
import yfinance as yf
from ddgs import DDGS

from pipeline.cache import read_through
from pipeline.runlog import RunLog

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
FETCH_TEXT_CAP = 40_000

# Domains that upgrade the default web-search trust tier.
TIER1_DOMAINS = {"sec.gov", "londonstockexchange.com"}  # primary filings / RNS
TIER2_DOMAINS = {  # official company / IR / macro
    "homedepot.com", "corporate.homedepot.com", "ir.homedepot.com",
    "analog.com", "investor.analog.com",
    "deere.com", "investor.deere.com",
    "haysplc.com", "hays.com",
    "bls.gov", "bea.gov", "federalreserve.gov", "ons.gov.uk", "bankofengland.co.uk",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def trust_tier_for_url(url: str) -> int:
    """4 (general web) unless the domain is clearly official."""
    try:
        host = (urlparse(url).netloc or "").lower().split(":")[0]
    except ValueError:
        return 4
    host = host.removeprefix("www.")
    for d in TIER1_DOMAINS:
        if host == d or host.endswith("." + d):
            return 1
    for d in TIER2_DOMAINS:
        if host == d or host.endswith("." + d):
            return 2
    return 4


def _source(doc_id: str, kind: str, trust_tier: int, published: str | None = None) -> dict:
    """SourceRef-compatible dict."""
    return {
        "doc_id": doc_id,
        "published": published,
        "fetched_at": _now(),
        "trust_tier": trust_tier,
        "kind": kind,
    }


def _run(tool: str, args: dict, fetch: Callable[[], Any],
         refresh: bool, log: RunLog | None) -> tuple[Any, bool]:
    """read_through wrapper: errors -> ({'error':...}, False), never cached."""
    try:
        result, hit = read_through(tool, args, fetch, refresh=refresh)
    except Exception as e:  # noqa: BLE001 - agent loop must continue
        result, hit = {"error": f"{type(e).__name__}: {e}", "tool": tool, "args": args}, False
    if log is not None:
        err = result.get("error") if isinstance(result, dict) else None
        log.event("tool_call", tool=tool, args=args, cache_hit=hit, error=err)
    return result, hit


# ---------------------------------------------------------------- search_web

def search_web(query: str, max_results: int = 8, *,
               refresh: bool = False, log: RunLog | None = None) -> tuple[dict, bool]:
    """DuckDuckGo web search. trust_tier=4 unless the domain is clearly official
    (SEC/RNS -> 1, company IR / official macro -> 2). Returns (data, was_cache_hit)."""
    args = {"query": query, "max_results": max_results}

    def fetch() -> dict:
        raw = list(DDGS().text(query, max_results=max_results))
        results = []
        for r in raw:
            url = r.get("href") or r.get("url") or ""
            results.append({
                "title": r.get("title", ""),
                "url": url,
                "snippet": r.get("body", ""),
                "source": _source(url, "news" if trust_tier_for_url(url) == 4 else "ir_page",
                                  trust_tier_for_url(url)),
            })
        return {"query": query, "results": results, "fetched_at": _now()}

    return _run("search_web", args, fetch, refresh, log)


# ---------------------------------------------------------------- search_news

def search_news(query: str, days: int = 7, max_results: int = 8, *,
                refresh: bool = False, log: RunLog | None = None) -> tuple[dict, bool]:
    """DuckDuckGo news search. Results are LEADS ONLY (trust_tier=4): headlines and
    snippets to be verified against tier-1/2 sources before any number is used.
    `days` maps to ddgs timelimit: <=1 -> "d", <=7 -> "w", else -> "m".
    Returns (data, was_cache_hit)."""
    timelimit = "d" if days <= 1 else ("w" if days <= 7 else "m")
    args = {"query": query, "days": days, "max_results": max_results}

    def fetch() -> dict:
        raw = list(DDGS().news(query, timelimit=timelimit, max_results=max_results))
        results = []
        for r in raw:
            url = r.get("url") or r.get("href") or ""
            results.append({
                "title": r.get("title", ""),
                "url": url,
                "snippet": r.get("body", ""),
                "publisher": r.get("source", ""),
                "date": r.get("date", ""),
                "source": _source(url, "news", 4, published=r.get("date") or None),
            })
        return {"query": query, "timelimit": timelimit, "results": results, "fetched_at": _now()}

    return _run("search_news", args, fetch, refresh, log)


# ---------------------------------------------------------------- fetch_url

def fetch_url(url: str, *, refresh: bool = False, log: RunLog | None = None) -> tuple[dict, bool]:
    """GET a URL (10s timeout, browser UA) and extract clean article text with
    trafilatura. Text capped at ~40k chars. Returns (data, was_cache_hit)."""
    args = {"url": url}

    def fetch() -> dict:
        resp = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        text = trafilatura.extract(resp.text) or ""
        tier = trust_tier_for_url(url)
        return {
            "url": url,
            "text": text[:FETCH_TEXT_CAP],
            "truncated": len(text) > FETCH_TEXT_CAP,
            "fetched_at": _now(),
            "source": _source(url, "ir_page" if tier <= 2 else "news", tier),
        }

    return _run("fetch_url", args, fetch, refresh, log)


# ---------------------------------------------------------------- get_market_data

MARKET_WHATS = ("info", "estimates", "quarterly_financials", "news")


def _polite_delay() -> None:
    """Rate-limit courtesy on LIVE yfinance fetches (cache hits never get here)."""
    time.sleep(0.5)


def _to_plain(value: Any) -> Any:
    """Recursively convert to JSON-serializable plain data; DataFrames get
    stringified index/columns."""
    if hasattr(value, "columns") and hasattr(value, "index"):  # DataFrame duck-type
        out: dict[str, dict[str, Any]] = {}
        for col in value.columns:
            series = value[col]
            out[str(col)] = {str(idx): _to_plain(v) for idx, v in series.items()}
        return out
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    try:
        f = float(value)
        return None if f != f else f  # NaN -> None
    except (TypeError, ValueError):
        return str(value)


def get_market_data(ticker: str, what: str, *,
                    refresh: bool = False, log: RunLog | None = None) -> tuple[dict, bool]:
    """yfinance market data (trust_tier=3 aggregator; its news items are tier 4 leads).
    `what` in {"info", "estimates", "quarterly_financials", "news"}.
    Returns (data, was_cache_hit)."""
    args = {"ticker": ticker, "what": what}
    if what not in MARKET_WHATS:
        err = {"error": f"unknown what={what!r}; expected one of {MARKET_WHATS}", "args": args}
        if log is not None:
            log.event("tool_call", tool="get_market_data", args=args, cache_hit=False,
                      error=err["error"])
        return err, False

    def fetch() -> dict:
        _polite_delay()
        t = yf.Ticker(ticker)
        if what == "info":
            data = _to_plain(t.info)
        elif what == "quarterly_financials":
            data = _to_plain(t.quarterly_financials)
        elif what == "news":
            data = _to_plain(t.news)
        else:  # estimates: collect whatever estimate tables this yfinance exposes
            data = {}
            for attr in ("earnings_estimate", "revenue_estimate", "eps_trend",
                         "growth_estimates", "analyst_price_targets"):
                try:
                    data[attr] = _to_plain(getattr(t, attr))
                except Exception as e:  # noqa: BLE001 - partial data beats none
                    data[attr] = {"error": f"{type(e).__name__}: {e}"}
        tier = 4 if what == "news" else 3
        return {
            "ticker": ticker,
            "what": what,
            "data": data,
            "fetched_at": _now(),
            "source": _source(f"yfinance:{ticker}:{what}",
                              "news" if what == "news" else "market_data", tier),
        }

    return _run("get_market_data", args, fetch, refresh, log)
