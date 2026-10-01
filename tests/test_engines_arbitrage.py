"""Structural arbitrage: exhaustiveness proofs, candidate detection, depth-walked pricing."""

import pytest

from src.engines.arbitrage import candidates, numeric_exhaustive, price_basket, scan_events
from src.engines.fees import FeeSchedule
from src.engines.market import Book, Event, Quote

TEMP_RULES = "If the maximum temperature recorded at X (CLINYC) is ... fahrenheit"


def q(ticker, st, lo=None, hi=None, yb=None, ya=None, rules=TEMP_RULES, title=None):
    num = lo if lo is not None else hi
    return Quote(ticker=ticker, event_ticker="E", strike_type=st, floor_strike=lo, cap_strike=hi,
                 yes_bid=yb, yes_ask=ya, status="active", rules=rules, subtitle=ticker,
                 title=title or f"Index above {num}")


def temp_event(prices):
    specs = [("T75", "less", None, 75), ("B75", "between", 75, 76), ("B77", "between", 77, 78),
             ("B79", "between", 79, 80), ("T80", "greater", 80, None)]
    ms = [q(t, st, lo, hi, *prices[i]) for i, (t, st, lo, hi) in enumerate(specs)]
    return Event("E", "KXHIGHNY", title="NYC high", mutually_exclusive=True, markets=ms)


def test_temperature_buckets_tile_integers_only():
    ev = temp_event([(None, None)] * 5)
    assert numeric_exhaustive(ev.markets, integer=True)
    assert not numeric_exhaustive(ev.markets, integer=False)  # 76.5 is in no bucket
    assert not numeric_exhaustive(ev.markets[:-1], integer=True)  # no upper tail


def test_cent_grid_buckets_are_exhaustive_for_decimal_outcomes():
    ms = [q("a", "less", hi=6000), q("b", "between", 6000, 6024.99), q("c", "between", 6025, 6049.99),
          q("d", "greater_or_equal", lo=6050)]
    assert numeric_exhaustive(ms, integer=False)


def test_all_yes_candidate_marked_risk_free_only_when_proven():
    # Sum of YES asks = 0.95 < 1 before fees.
    ev = temp_event([(0.01, 0.02), (0.10, 0.12), (0.45, 0.47), (0.30, 0.31), (0.02, 0.03)])
    found = {c[0]: c for c in candidates(ev, FeeSchedule("quadratic", 0.0))}
    assert "all-YES" in found and found["all-YES"][3] is True
    # Same prices on a non-numeric event: reported but not risk-free.
    plain = Event("E", "KXFOO", mutually_exclusive=True,
                  markets=[Quote(m.ticker, "E", strike_type="custom", yes_bid=m.yes_bid, yes_ask=m.yes_ask,
                                 status="active") for m in ev.markets])
    found = {c[0]: c for c in candidates(plain, FeeSchedule("quadratic", 0.0))}
    assert found["all-YES"][3] is False


def test_all_no_candidate_when_yes_bids_overround():
    # YES bids sum to 1.04 -> NO asks sum to 5 - 1.04 = 3.96 < 4 (before fees)
    ev = temp_event([(0.03, 0.04), (0.18, 0.19), (0.46, 0.47), (0.30, 0.31), (0.07, 0.08)])
    kinds = {c[0] for c in candidates(ev, FeeSchedule("quadratic", 0.0))}
    assert "all-NO" in kinds
    # Taker fees on five NO legs (~4.9c) eat a 4c overround: no candidate.
    kinds = {c[0] for c in candidates(ev, FeeSchedule())}
    assert "all-NO" not in kinds


def test_ladder_violation_detected():
    ms = [q("A", "greater", lo=100, yb=0.30, ya=0.32, rules=""), q("B", "greater", lo=110, yb=0.40, ya=0.42, rules="")]
    ev = Event("E", "KXIDX", markets=ms)
    found = [c for c in candidates(ev, FeeSchedule("quadratic", 0.0)) if c[0] == "ladder-greater"]
    assert found
    (qa, sa), (qb, sb) = found[0][1]
    assert (qa.ticker, sa, qb.ticker, sb) == ("A", "yes", "B", "no")  # YES(>100) 0.32 + NO(>110) 0.60 = 0.92


def test_ladders_only_pair_markets_on_the_same_subject():
    # Two players' receiving-yard ladders in one game event: no cross-player constraint.
    ms = [q("A", "greater", lo=24.5, yb=0.30, ya=0.32, rules="", title="Noah Fant: 25+ receiving yards"),
          q("B", "greater", lo=49.5, yb=0.40, ya=0.42, rules="", title="Juwan Johnson: 50+ receiving yards")]
    assert not candidates(Event("E", "KXNFLRECYDS", markets=ms), FeeSchedule("quadratic", 0.0))
    ms[1] = q("B", "greater", lo=49.5, yb=0.40, ya=0.42, rules="", title="Noah Fant: 50+ receiving yards")
    assert candidates(Event("E", "KXNFLRECYDS", markets=ms), FeeSchedule("quadratic", 0.0))


def test_mislabeled_exactly_markets_are_not_ladders():
    # Real Kalshi data: "exactly 5" shipped as strike_type=less, floor=cap=5.
    exactly = lambda n, yb, ya: Quote(  # noqa: E731
        ticker=f"S-{n}.0", event_ticker="S", strike_type="less", floor_strike=n, cap_strike=n,
        yes_bid=yb, yes_ask=ya, status="active", title="Starship launches reaching space",
        rules=f"If exactly {n} Starship launches reach Space in 2026, then the market resolves to Yes.")
    ms = [exactly(5, 0.20, 0.22), exactly(9, 0.01, 0.03)]
    assert not ms[0].strikes_consistent() and ms[0].contains(5) is None
    assert not candidates(Event("S", "KXSTARSHIPSPACE", mutually_exclusive=False, markets=ms), FeeSchedule())


def test_price_basket_walks_depth_and_stops_when_profit_vanishes():
    ms = [q("A", "greater", lo=100, rules=""), q("B", "greater", lo=110, rules="")]
    ev = Event("E", "KXIDX", markets=ms)
    books = {
        # YES asks on A: 0.32 x10, then 0.45 x100 (from NO bids 0.68, 0.55)
        "A": Book("A", yes_bids=[], no_bids=[(0.68, 10.0), (0.55, 100.0)]),
        # NO asks on B: 0.60 x1000 (from YES bid 0.40)
        "B": Book("B", yes_bids=[(0.40, 1000.0)], no_bids=[]),
    }
    b = price_basket("ladder-greater", ev, [(ms[0], "yes"), (ms[1], "no")], 1.0, True, "", books,
                     FeeSchedule("quadratic", 0.0), min_profit=0.01)
    # Baskets 11+ cost 0.45 + 0.60 > 1 each: total profit peaks at exactly 10.
    assert b is not None and b.count == 10
    assert b.profit == pytest.approx(1.0 - 0.32 - 0.60)
    assert b.total_profit == pytest.approx(0.80)


def test_scan_events_survives_book_errors():
    ev = temp_event([(0.01, 0.02), (0.10, 0.12), (0.45, 0.47), (0.30, 0.31), (0.02, 0.03)])

    def boom(_):
        raise RuntimeError("book down")

    assert scan_events([ev], lambda s: FeeSchedule("quadratic", 0.0), boom) == []
