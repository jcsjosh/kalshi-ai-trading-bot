"""Structural arbitrage: baskets whose worst-case payout beats their all-in cost.

No forecasting skill involved; these are violations of probability axioms in
the posted prices, after Kalshi's per-series taker fees on every leg:

* **all-NO** (mutually exclusive event, n markets): at most one market resolves
  YES, so n NO contracts pay at least n - 1.
* **all-YES** (exhaustive event): exactly one market resolves YES, so n YES
  contracts pay exactly 1. Only marked risk-free when exhaustiveness is proven
  from the strikes (numeric buckets that tile the number line, like Kalshi's
  temperature events); otherwise a hidden "none of these" outcome could lose
  everything, and the basket is reported but never traded.
* **ladder** (same event, nested strikes): "above X1" can't be less likely than
  "above X2" when X1 < X2, so YES(above X1) + NO(above X2) pays at least 1.

Snapshot prices only nominate candidates. Each candidate is then re-priced off
live order book depth, leg by leg, to find the largest basket count that still
clears the minimum profit; that walked price is what gets reported and traded.

Execution caveat: Kalshi has no atomic multi-leg order. Legs go out as
immediate-or-cancel orders, scarcest leg first, and a partial fill leaves an
unhedged position. The runner stops at the first short leg and reports it.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.engines.fees import FeeSchedule
from src.engines.market import Book, Event, Quote

NUMERIC = ("between", "less", "greater", "less_or_equal", "greater_or_equal")


@dataclass
class Leg:
    ticker: str
    side: str
    price: float  # worst level walked to
    fee: float  # per contract at basket size
    available: float  # contracts visible at or better than ``price``


@dataclass
class Basket:
    kind: str
    event_ticker: str
    series_ticker: str
    title: str
    legs: List[Leg]
    payout: float  # guaranteed $ per basket
    cost: float  # $ per basket, fees included
    profit: float  # guaranteed $ per basket
    count: int  # baskets executable at these prices
    risk_free: bool
    note: str
    meta: Dict = field(default_factory=dict)

    @property
    def roi(self) -> float:
        return self.profit / self.cost if self.cost else 0.0

    @property
    def total_profit(self) -> float:
        return round(self.profit * self.count, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(roi=round(self.roi, 4), total_profit=self.total_profit)
        return d


def integer_outcome(event: Event) -> bool:
    """Contracts whose settlement value is a whole number (NWS climate reports
    are whole degrees). Anything else is checked on a cent grid."""
    rules = " ".join(m.rules for m in event.markets[:1]).lower()
    return "temperature" in rules and "fahrenheit" in rules


def _interval(m: Quote, scale: int) -> Optional[Tuple[float, float]]:
    """The market's YES set as an inclusive interval of outcome units (whole
    numbers when ``scale=1``, cents when ``scale=100``); +-inf for open tails."""
    lo, hi, st = m.floor_strike, m.cap_strike, m.strike_type
    if not m.strikes_consistent():
        return None
    inf = float("inf")
    if st == "between" and lo is not None and hi is not None:
        return math.ceil(lo * scale - 1e-9), math.floor(hi * scale + 1e-9)
    if st == "less" and hi is not None:
        return -inf, math.ceil(hi * scale - 1e-9) - 1
    if st == "less_or_equal" and hi is not None:
        return -inf, math.floor(hi * scale + 1e-9)
    if st == "greater" and lo is not None:
        return math.floor(lo * scale + 1e-9) + 1, inf
    if st == "greater_or_equal" and lo is not None:
        return math.ceil(lo * scale - 1e-9), inf
    return None


def numeric_exhaustive(markets: Sequence[Quote], integer: bool = False) -> bool:
    """True when every possible outcome resolves exactly one market YES.

    The markets' YES sets, as intervals on the outcome grid (whole numbers for
    integer-valued contracts, cents otherwise), must run from -inf to +inf with
    no gap and no overlap. So "6000 to 6024.99" next to "6025 to 6049.99"
    tiles, but "75 to 76" next to "77 to 78" only tiles whole-degree outcomes.
    Never trusts the event's ``mutually_exclusive`` flag for this.
    """
    if len(markets) < 2:
        return False
    ivs = [_interval(m, 1 if integer else 100) for m in markets]
    if any(iv is None for iv in ivs):
        return False
    ivs.sort()
    if ivs[0][0] != float("-inf") or ivs[-1][1] != float("inf"):
        return False
    for (a0, b0), (a1, b1) in zip(ivs, ivs[1:]):
        if a0 > b0 or a1 != b0 + 1:
            return False
    return ivs[-1][0] <= ivs[-1][1]


_NUM = re.compile(r"[-+]?\d[\d,]*(\.\d+)?")


def subject_key(q: Quote) -> str:
    """What a strike market is *about*, with the numbers masked out.

    One Kalshi event can hold ladders on different quantities (every player's
    receiving yards in a game, each team's winning margin), and "above X" vs
    "above Y" only constrains prices on the same quantity. "Noah Fant: 25+
    receiving yards" and "Noah Fant: 50+ receiving yards" share a subject;
    Juwan Johnson's ladder is a different one.
    """
    return _NUM.sub("#", q.title or q.ticker.rsplit("-", 1)[0]).strip().lower()


def _snapshot_cost(legs: List[Tuple[Quote, str]], fees: FeeSchedule, n: int = 100) -> Optional[float]:
    total = 0.0
    for q, side in legs:
        ask = q.ask(side)
        if ask is None:
            return None
        total += ask + fees.per_contract(n, ask, "taker")
    return total


def candidates(
    event: Event, fees: FeeSchedule, max_ladder_pairs: int = 3
) -> List[Tuple[str, List[Tuple[Quote, str]], float, bool, str]]:
    """(kind, legs, payout, risk_free, note) for baskets that look profitable on the snapshot."""
    ms = event.active_markets()
    out = []
    # Exhaustiveness is a property of the whole event; a closed or missing leg breaks it.
    whole = len(ms) == len(event.markets)
    integer = integer_outcome(event)
    # Proven only for whole-number outcomes. A cent-grid tiling is not proof: a
    # settlement value like a 60-second average can land between 1249.99 and 1250.00.
    exhaustive = whole and integer and numeric_exhaustive(ms, integer=True)
    tiles_cents = whole and not integer and numeric_exhaustive(ms, integer=False)
    exclusive = event.mutually_exclusive or exhaustive
    if exclusive and len(ms) >= 2:
        legs = [(q, "no") for q in ms]
        cost = _snapshot_cost(legs, fees)
        if cost is not None and cost < len(ms) - 1:
            out.append(("all-NO", legs, float(len(ms) - 1), True, "At most one outcome can resolve YES."))
        legs = [(q, "yes") for q in ms]
        cost = _snapshot_cost(legs, fees)
        if cost is not None and cost < 1.0:
            if exhaustive:
                note = "Strikes tile every whole-number outcome: exactly one leg pays."
            elif tiles_cents:
                note = ("Strikes tile every cent, but the settlement value may be finer (e.g. an average) "
                        "and fall between buckets. Not proven risk-free; never auto-traded.")
            else:
                note = "Only risk-free if the listed outcomes are exhaustive. Read the rules; never auto-traded."
            out.append(("all-YES", legs, 1.0, exhaustive, note))

    for direction, key in (("greater", "floor_strike"), ("less", "cap_strike")):
        subjects: Dict[str, List[Quote]] = {}
        for q in ms:
            if q.strike_type.startswith(direction) and getattr(q, key) is not None and q.strikes_consistent():
                subjects.setdefault(subject_key(q), []).append(q)
        for ladder in subjects.values():
            ladder.sort(key=lambda q: float(getattr(q, key)))
            found = []
            for lo, hi in combinations(ladder, 2):
                if getattr(lo, key) == getattr(hi, key):
                    continue
                # "above": the lower strike is the likelier one; "below": the higher strike is.
                likely, unlikely = (lo, hi) if direction == "greater" else (hi, lo)
                legs = [(likely, "yes"), (unlikely, "no")]
                cost = _snapshot_cost(legs, fees)
                if cost is not None and cost < 1.0:
                    found.append((cost, legs, likely, unlikely))
            # The deepest violations first; neighbouring pairs mostly repeat the same mispricing.
            for cost, legs, likely, unlikely in sorted(found, key=lambda f: f[0])[:max_ladder_pairs]:
                out.append((f"ladder-{direction}", legs, 1.0, True,
                            f"'{likely.subtitle or likely.ticker}' is at least as likely as "
                            f"'{unlikely.subtitle or unlikely.ticker}'."))
    return out


def price_basket(
    kind: str,
    event: Event,
    legs: List[Tuple[Quote, str]],
    payout: float,
    risk_free: bool,
    note: str,
    books: Dict[str, Book],
    fees: FeeSchedule,
    min_profit: float = 0.01,
    max_count: int = 1000,
) -> Optional[Basket]:
    """Largest basket count whose walked, fee-inclusive cost still clears ``min_profit``."""
    ladders = []
    for q, side in legs:
        book = books.get(q.ticker)
        if book is None:
            return None
        asks = book.asks(side)
        if not asks:
            return None
        ladders.append(asks)

    def cost_at(n: int) -> Optional[Tuple[float, List[Leg]]]:
        total, out = 0.0, []
        for (q, side), asks in zip(legs, ladders):
            filled, spent, worst = 0.0, 0.0, 0.0
            for price, size in asks:
                take = min(size, n - filled)
                filled += take
                spent += take * price
                worst = price
                if filled >= n:
                    break
            if filled < n:
                return None
            fee = fees.fee(n, worst, "taker") / n  # priced at the worst level: conservative
            total += spent / n + fee
            avail = sum(s for p, s in asks if p <= worst + 1e-9)
            out.append(Leg(q.ticker, side, worst, round(fee, 4), avail))
        return total, out

    # Marginal cost only rises as we walk the books, so total profit is concave
    # in the basket count and peaks at a level boundary of some leg.
    breaks = {max_count}
    for asks in ladders:
        depth = 0.0
        for _, size in asks:
            depth += size
            if depth >= 1:
                breaks.add(int(min(depth, max_count)))
    best = None
    for n in sorted(breaks):
        res = cost_at(n)
        if res is None:
            continue
        cost, legs_priced = res
        if payout - cost < min_profit:
            continue
        if best is None or n * (payout - cost) > best[0] * (payout - best[1]):
            best = (n, cost, legs_priced)
    if best is None:
        return None
    n, cost, legs_priced = best
    return Basket(
        kind=kind,
        event_ticker=event.event_ticker,
        series_ticker=event.series_ticker,
        title=event.title,
        legs=legs_priced,
        payout=payout,
        cost=round(cost, 4),
        profit=round(payout - cost, 4),
        count=n,
        risk_free=risk_free,
        note=note,
    )


def scan_events(
    events: Sequence[Event],
    fees_for: Callable[[str], FeeSchedule],
    book_for: Callable[[str], Book],
    min_profit: float = 0.01,
    log=lambda *_: None,
    include_unproven: bool = False,
) -> List[Basket]:
    """Price every snapshot candidate off live depth. Baskets that are only
    risk-free if unlisted outcomes can't happen are skipped unless asked for:
    they are never traded, so their books aren't worth fetching."""
    baskets = []
    nominated = 0
    books: Dict[str, Book] = {}  # shared across candidates: legs repeat within an event
    for ev in events:
        fees = fees_for(ev.series_ticker)
        for kind, legs, payout, risk_free, note in candidates(ev, fees):
            if not risk_free and not include_unproven:
                continue
            nominated += 1
            try:
                for q, _ in legs:
                    if q.ticker not in books:
                        books[q.ticker] = book_for(q.ticker)
            except Exception as exc:
                log(f"  book fetch failed for {ev.event_ticker}: {exc}")
                continue
            b = price_basket(kind, ev, legs, payout, risk_free, note, books, fees, min_profit)
            if b:
                baskets.append(b)
    log(f"arbitrage: {nominated} snapshot candidates, {len(baskets)} survive live depth + fees")
    return sorted(baskets, key=lambda b: (b.risk_free, b.total_profit), reverse=True)
