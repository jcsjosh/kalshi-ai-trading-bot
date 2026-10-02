"""Normalized Kalshi markets, events and order books (all prices in dollars).

Kalshi quotes one YES-denominated book: a YES bid at ``p`` is a NO offer at
``1 - p``. ``Quote`` keeps both views so engines can price either side, and
``Book`` walks real depth so sizing never assumes liquidity that isn't there.

The nested ``*_dollars`` fields on ``/events`` are a snapshot that can lag the
live book; scan with them, then re-price the shortlist with ``fetch_book``
before deciding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.agent.settle import series_category as series_of
from src.engines.http import KALSHI_API



def _f(raw: Any) -> Optional[float]:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _price(raw: Any) -> Optional[float]:
    """A tradeable price in (0, 1); 0 or 1 means an empty book side."""
    v = _f(raw)
    if v is None or v <= 0.0 or v >= 1.0:
        return None
    return round(v, 4)


def parse_time(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class Quote:
    ticker: str
    event_ticker: str
    title: str = ""
    subtitle: str = ""
    status: str = "active"
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0
    last_price: Optional[float] = None
    volume: float = 0.0
    volume_24h: float = 0.0
    open_interest: float = 0.0
    close_time: Optional[datetime] = None
    strike_type: str = ""
    floor_strike: Optional[float] = None
    cap_strike: Optional[float] = None
    rules: str = ""
    result: str = ""
    expiration_value: Optional[float] = None

    @property
    def series_ticker(self) -> str:
        return series_of(self.ticker)

    @property
    def no_bid(self) -> Optional[float]:
        return round(1.0 - self.yes_ask, 4) if self.yes_ask is not None else None

    @property
    def no_ask(self) -> Optional[float]:
        return round(1.0 - self.yes_bid, 4) if self.yes_bid is not None else None

    def bid(self, side: str) -> Optional[float]:
        return self.yes_bid if side == "yes" else self.no_bid

    def ask(self, side: str) -> Optional[float]:
        return self.yes_ask if side == "yes" else self.no_ask

    def ask_size(self, side: str) -> float:
        # A NO ask is resting YES bid size, and vice versa.
        return self.yes_ask_size if side == "yes" else self.yes_bid_size

    @property
    def mid(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return round((self.yes_bid + self.yes_ask) / 2.0, 4)
        return self.yes_ask if self.yes_ask is not None else self.yes_bid

    @property
    def spread(self) -> Optional[float]:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return round(self.yes_ask - self.yes_bid, 4)

    def hours_to_close(self, now: Optional[datetime] = None) -> Optional[float]:
        if self.close_time is None:
            return None
        now = now or datetime.now(timezone.utc)
        return (self.close_time - now).total_seconds() / 3600.0

    def contains(self, value: float) -> Optional[bool]:
        """Whether a numeric outcome ``value`` resolves this market YES.

        ``between`` is inclusive on both strikes (Kalshi's "75° to 76°" is
        ``floor=75, cap=76``); ``less``/``greater`` are strict. Returns None for
        strike types that are not a numeric interval.
        """
        st = self.strike_type
        lo, hi = self.floor_strike, self.cap_strike
        if not self.strikes_consistent():
            return None
        if st == "between" and lo is not None and hi is not None:
            return lo <= value <= hi
        if st == "less" and hi is not None:
            return value < hi
        if st == "less_or_equal" and hi is not None:
            return value <= hi
        if st == "greater" and lo is not None:
            return value > lo
        if st == "greater_or_equal" and lo is not None:
            return value >= lo
        return None

    def strikes_consistent(self) -> bool:
        """Whether the strike metadata has the shape its type implies.

        Kalshi occasionally mislabels markets: "exactly 5 launches" has shipped
        as ``strike_type=less`` with ``floor=cap=5``. A one-sided type carrying
        both strikes, or rules that say "exactly" on a one-sided type, is not
        what it claims, so nothing numeric is inferred from it.
        """
        st, lo, hi = self.strike_type, self.floor_strike, self.cap_strike
        if st in ("less", "less_or_equal") and lo is not None:
            return False
        if st in ("greater", "greater_or_equal") and hi is not None:
            return False
        if st in ("less", "less_or_equal", "greater", "greater_or_equal") and "exactly" in self.rules.lower():
            return False
        return True

    @classmethod
    def from_api(cls, m: Dict[str, Any], event_ticker: str = "") -> "Quote":
        return cls(
            ticker=m["ticker"],
            event_ticker=m.get("event_ticker") or event_ticker,
            title=m.get("title", "") or "",
            subtitle=m.get("yes_sub_title") or m.get("subtitle") or "",
            status=m.get("status", "") or "",
            yes_bid=_price(m.get("yes_bid_dollars")),
            yes_ask=_price(m.get("yes_ask_dollars")),
            yes_bid_size=_f(m.get("yes_bid_size_fp")) or 0.0,
            yes_ask_size=_f(m.get("yes_ask_size_fp")) or 0.0,
            last_price=_price(m.get("last_price_dollars")),
            volume=_f(m.get("volume_fp")) or 0.0,
            volume_24h=_f(m.get("volume_24h_fp")) or 0.0,
            open_interest=_f(m.get("open_interest_fp")) or 0.0,
            close_time=parse_time(m.get("close_time")),
            strike_type=m.get("strike_type") or "",
            floor_strike=_f(m.get("floor_strike")),
            cap_strike=_f(m.get("cap_strike")),
            rules=m.get("rules_primary", "") or "",
            result=m.get("result", "") or "",
            expiration_value=_f(m.get("expiration_value")),
        )


@dataclass
class Event:
    event_ticker: str
    series_ticker: str
    title: str = ""
    category: str = ""
    mutually_exclusive: bool = False
    markets: List[Quote] = field(default_factory=list)

    @classmethod
    def from_api(cls, e: Dict[str, Any]) -> "Event":
        et = e.get("event_ticker", "")
        return cls(
            event_ticker=et,
            series_ticker=e.get("series_ticker") or series_of(et),
            title=e.get("title", "") or "",
            category=e.get("category", "") or "",
            mutually_exclusive=bool(e.get("mutually_exclusive")),
            markets=[Quote.from_api(m, et) for m in e.get("markets") or []],
        )

    def active_markets(self) -> List[Quote]:
        return [m for m in self.markets if m.status in ("active", "open", "")]


@dataclass
class Book:
    """Live depth for one market. Levels are (price, size), best first."""

    ticker: str
    yes_bids: List[Tuple[float, float]]
    no_bids: List[Tuple[float, float]]

    @classmethod
    def from_api(cls, ticker: str, data: Dict[str, Any]) -> "Book":
        ob = data.get("orderbook_fp") or {}

        def levels(key: str) -> List[Tuple[float, float]]:
            out = []
            for price, size in ob.get(key) or []:
                p, s = _f(price), _f(size)
                if p is not None and s and 0.0 < p < 1.0:
                    out.append((round(p, 4), s))
            return sorted(out, key=lambda lv: -lv[0])

        return cls(ticker=ticker, yes_bids=levels("yes_dollars"), no_bids=levels("no_dollars"))

    def asks(self, side: str) -> List[Tuple[float, float]]:
        """Offers to BUY ``side`` (cheapest first): a NO bid at q is a YES ask at 1-q."""
        opposite = self.no_bids if side == "yes" else self.yes_bids
        return [(round(1.0 - p, 4), s) for p, s in opposite]

    def bids(self, side: str) -> List[Tuple[float, float]]:
        return list(self.yes_bids if side == "yes" else self.no_bids)

    def best_ask(self, side: str) -> Optional[float]:
        a = self.asks(side)
        return a[0][0] if a else None

    def best_bid(self, side: str) -> Optional[float]:
        b = self.bids(side)
        return b[0][0] if b else None

    def apply_to(self, quote: Quote) -> Quote:
        """Overwrite a snapshot quote's top of book with this live book."""
        yb = self.yes_bids[0] if self.yes_bids else None
        ya = self.asks("yes")[0] if self.no_bids else None
        quote.yes_bid, quote.yes_bid_size = (yb[0], yb[1]) if yb else (None, 0.0)
        quote.yes_ask, quote.yes_ask_size = (ya[0], ya[1]) if ya else (None, 0.0)
        return quote


def walk_asks(levels: List[Tuple[float, float]], max_price: float, max_count: float) -> Tuple[float, float]:
    """Contracts and average price available buying up the book to ``max_price``."""
    filled, cost = 0.0, 0.0
    for price, size in levels:
        if price > max_price + 1e-9 or filled >= max_count:
            break
        take = min(size, max_count - filled)
        filled += take
        cost += take * price
    return filled, (cost / filled if filled else 0.0)


def devig(quotes: List[Quote]) -> Dict[str, float]:
    """Market-implied probabilities from mids, normalized across an exhaustive event."""
    mids = {}
    for q in quotes:
        if q.yes_bid is None and q.yes_ask is None:
            mids[q.ticker] = 0.005
        else:
            lo = q.yes_bid if q.yes_bid is not None else 0.0
            hi = q.yes_ask if q.yes_ask is not None else min(lo + 0.02, 1.0)
            mids[q.ticker] = max((lo + hi) / 2.0, 0.005)
    total = sum(mids.values())
    return {t: v / total for t, v in mids.items()} if total > 0 else {}


class KalshiPublic:
    """Read-only Kalshi market data through a throttled ``Fetcher``."""

    def __init__(self, fetcher, base_url: str = KALSHI_API):
        self.fetcher = fetcher
        self.base_url = base_url.rstrip("/")

    def get(self, path: str, params: Optional[Dict[str, Any]] = None, **kw) -> Dict[str, Any]:
        return self.fetcher.get(f"{self.base_url}{path}", params, **kw)

    def iter_events(
        self,
        status: str = "open",
        series_ticker: Optional[str] = None,
        max_pages: int = 100,
        cache: bool = False,
        cache_ttl: Optional[float] = None,
        min_close_ts: Optional[int] = None,
    ) -> Iterator[Event]:
        cursor = None
        for _ in range(max_pages):
            params: Dict[str, Any] = {"status": status, "with_nested_markets": "true", "limit": 200}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if min_close_ts:
                params["min_close_ts"] = min_close_ts
            if cursor:
                params["cursor"] = cursor
            data = self.get("/events", params, cache=cache, cache_ttl=cache_ttl)
            for raw in data.get("events", []):
                yield Event.from_api(raw)
            cursor = data.get("cursor")
            if not cursor:
                return

    def series(self, series_ticker: str) -> Dict[str, Any]:
        return self.get(f"/series/{series_ticker}", cache=True, cache_ttl=86400).get("series", {})

    def list_series(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"category": category} if category else None
        return self.get("/series", params, cache=True, cache_ttl=86400).get("series", [])

    def book(self, ticker: str, depth: int = 20) -> Book:
        return Book.from_api(ticker, self.get(f"/markets/{ticker}/orderbook", {"depth": depth}))

    def event_candles(
        self, series_ticker: str, event_ticker: str, start_ts: int, end_ts: int, period: int = 60,
        cache: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Candlesticks (``period`` minutes) for every market in an event. Cache
        only settled events: their history no longer changes."""
        data = self.get(
            f"/series/{series_ticker}/events/{event_ticker}/candlesticks",
            {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period},
            cache=cache,
        )
        return dict(zip(data.get("market_tickers") or [], data.get("market_candlesticks") or []))
