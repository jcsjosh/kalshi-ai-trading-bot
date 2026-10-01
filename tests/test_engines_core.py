"""Edge-engine core: fees, market normalization, evaluator, portfolio selection."""

import math

import pytest

from src.engines.evaluate import (
    EvalConfig,
    Fair,
    evaluate,
    kelly_fraction,
    select,
    shrink,
)
from src.engines.fees import FeeSchedule, SeriesFees, series_of
from src.engines.market import Book, Event, Quote, walk_asks


# -- fees --------------------------------------------------------------------


def test_taker_fee_matches_published_schedule():
    f = FeeSchedule()
    assert f.fee(100, 0.50, "taker") == pytest.approx(1.75)
    assert f.fee(1, 0.50, "taker") == pytest.approx(0.02)  # 0.0175 rounds up
    assert f.fee(1, 0.10, "taker") == pytest.approx(0.01)
    assert f.fee(10, 0.07, "taker") == pytest.approx(0.05)  # 0.0456 -> 0.05


def test_maker_fee_only_on_maker_fee_series():
    assert FeeSchedule("quadratic").fee(100, 0.5, "maker") == 0.0
    assert FeeSchedule("quadratic_with_maker_fees").fee(100, 0.5, "maker") == pytest.approx(0.44)
    # Unknown types are priced pessimistically (maker fees on).
    assert FeeSchedule("something_new").fee(100, 0.5, "maker") > 0


def test_fee_multiplier_scales_and_zero_is_free():
    assert FeeSchedule("quadratic", 0.5).fee(100, 0.5, "taker") == pytest.approx(0.88)
    assert FeeSchedule("quadratic", 0.0).fee(100, 0.5, "taker") == 0.0


def test_exact_cent_does_not_round_up():
    # 0.07 * 100 * 0.5 * 0.5 * 4 = 7.00 exactly; float noise must not make it 7.01
    assert FeeSchedule().fee(400, 0.5, "taker") == pytest.approx(7.0)


def test_from_series_reads_api_shape():
    s = FeeSchedule.from_series({"series": {"fee_type": "quadratic_with_maker_fees", "fee_multiplier": 0.5}})
    assert s.fee_type == "quadratic_with_maker_fees" and s.multiplier == 0.5


def test_series_fees_falls_back_pessimistically_on_error():
    class Boom:
        def get(self, *a, **k):
            raise RuntimeError("down")

    sched = SeriesFees(Boom()).get("KXFOO")
    assert sched.maker_fees and sched.multiplier == 1.0
    assert series_of("KXHIGHNY-26OCT02-B83.5") == "KXHIGHNY"


# -- market ------------------------------------------------------------------

RAW_BUCKET = {
    "ticker": "KXHIGHNY-26OCT02-B83.5",
    "event_ticker": "KXHIGHNY-26OCT02",
    "status": "active",
    "strike_type": "between",
    "floor_strike": 83,
    "cap_strike": 84,
    "yes_bid_dollars": "0.4600",
    "yes_ask_dollars": "0.4700",
    "yes_bid_size_fp": "120.00",
    "yes_ask_size_fp": "80.00",
    "volume_24h_fp": "1540.83",
    "close_time": "2026-10-03T05:00:00Z",
}


def test_quote_sides_and_strikes():
    q = Quote.from_api(RAW_BUCKET)
    assert q.series_ticker == "KXHIGHNY"
    assert q.no_ask == pytest.approx(0.54) and q.no_bid == pytest.approx(0.53)
    assert q.ask_size("no") == 120.0  # NO offers are resting YES bids
    assert q.contains(83) and q.contains(84) and not q.contains(85) and not q.contains(82)
    tail_lo = Quote.from_api({**RAW_BUCKET, "strike_type": "less", "floor_strike": None, "cap_strike": 81})
    assert tail_lo.contains(80) and not tail_lo.contains(81)  # "80 or below"
    tail_hi = Quote.from_api({**RAW_BUCKET, "strike_type": "greater", "floor_strike": 88, "cap_strike": None})
    assert tail_hi.contains(89) and not tail_hi.contains(88)  # "89 or above"


def test_empty_book_side_is_none():
    q = Quote.from_api({**RAW_BUCKET, "yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000"})
    assert q.yes_bid is None and q.yes_ask is None and q.mid is None


def test_book_converts_opposite_bids_to_asks():
    book = Book.from_api(
        "T",
        {"orderbook_fp": {"yes_dollars": [["0.43", "44"], ["0.46", "27"]], "no_dollars": [["0.50", "123"], ["0.51", "162"]]}},
    )
    assert book.best_bid("yes") == 0.46
    assert book.best_ask("yes") == pytest.approx(0.49)  # 1 - best NO bid 0.51
    assert book.asks("no")[0] == (pytest.approx(0.54), 27.0)
    filled, avg = walk_asks(book.asks("yes"), max_price=0.50, max_count=1000)
    assert filled == 285 and avg == pytest.approx((162 * 0.49 + 123 * 0.50) / 285)
    q = book.apply_to(Quote.from_api(RAW_BUCKET))
    assert q.yes_ask == pytest.approx(0.49) and q.yes_ask_size == 162


def test_event_from_api():
    ev = Event.from_api({"event_ticker": "KXHIGHNY-26OCT02", "mutually_exclusive": True, "markets": [RAW_BUCKET]})
    assert ev.series_ticker == "KXHIGHNY" and ev.mutually_exclusive and len(ev.active_markets()) == 1


# -- evaluator ---------------------------------------------------------------


def test_shrink_respects_weight():
    assert shrink(0.9, 0.5, 0.0) == pytest.approx(0.5)
    assert shrink(0.9, 0.5, 1.0) == pytest.approx(0.9)
    mid = shrink(0.9, 0.5, 0.5)
    assert 0.5 < mid < 0.9
    assert shrink(0.7, None, 0.2) == pytest.approx(0.7)


def test_kelly():
    assert kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert kelly_fraction(0.5, 0.5) == 0.0


def _quote(yb, ya, size=500.0):
    return Quote.from_api({**RAW_BUCKET, "yes_bid_dollars": str(yb), "yes_ask_dollars": str(ya),
                           "yes_bid_size_fp": str(size), "yes_ask_size_fp": str(size)})


def test_zero_weight_engine_never_trades():
    q = _quote(0.46, 0.47)
    assert evaluate("x", q, Fair(q.ticker, p_yes=0.95, weight=0.0), EvalConfig()) == []


def test_finds_fee_aware_taker_edge_and_sizes_with_kelly():
    q = _quote(0.46, 0.47)
    cfg = EvalConfig(bankroll=1000, include_maker=False)
    opps = evaluate("x", q, Fair(q.ticker, p_yes=0.60, weight=1.0), cfg)
    assert len(opps) == 1
    o = opps[0]
    assert (o.side, o.role, o.price) == ("yes", "taker", 0.47)
    assert o.ev == pytest.approx(0.60 - 0.47 - o.fee, abs=1e-4)
    assert o.fee == pytest.approx(math.ceil(0.07 * o.contracts * 0.47 * 0.53 * 100 - 1e-9) / 100 / o.contracts, abs=1e-4)
    assert o.stake <= cfg.bankroll * cfg.max_bet_fraction + 1e-6


def test_stderr_raises_the_bar():
    q = _quote(0.46, 0.47)
    cfg = EvalConfig(include_maker=False, min_edge=0.03)
    assert evaluate("x", q, Fair(q.ticker, p_yes=0.53, weight=1.0), cfg)
    assert not evaluate("x", q, Fair(q.ticker, p_yes=0.53, weight=1.0, stderr=0.05), cfg)


def test_maker_entry_inside_spread_with_adverse_selection():
    q = _quote(0.40, 0.50)
    cfg = EvalConfig(maker_adverse_selection=0.01)
    opps = {o.role: o for o in evaluate("x", q, Fair(q.ticker, p_yes=0.53, weight=1.0), cfg)}
    assert opps["maker"].price == 0.41
    assert opps["maker"].ev == pytest.approx(0.53 - 0.01 - 0.41, abs=1e-4)  # quadratic series: no maker fee
    assert opps["maker"].p_fair == pytest.approx(0.53)  # the fill-conditional probability it was priced at
    assert "taker" not in opps  # 0.53 - 0.50 - 0.0175 fee < 0.03


def test_uninformed_engine_does_not_post_inside_wide_spreads():
    # Mid 0.11 on a 0.03/0.19 book: a 0.04 bid "beats" the mid, but it only fills when
    # someone sells at 0.04. Without engine information that is not edge.
    q = _quote(0.03, 0.19)
    assert evaluate("x", q, Fair(q.ticker, p_yes=0.11, weight=0.0), EvalConfig()) == []
    informed = evaluate("x", q, Fair(q.ticker, p_yes=0.11, weight=1.0), EvalConfig())
    assert informed and informed[0].role == "maker" and informed[0].price == 0.04


def test_reprice_uses_the_live_mid_not_the_snapshot_consensus():
    from src.engines.fees import SeriesFees
    from src.engines.runner import reprice

    class Live:
        def book(self, ticker):  # the market moved down to 0.40/0.42 since the snapshot
            return Book.from_api(ticker, {"orderbook_fp": {"yes_dollars": [["0.40", "500"]],
                                                            "no_dollars": [["0.58", "500"]]}})

    stale = _quote(0.48, 0.52)
    fair = Fair(stale.ticker, p_yes=0.50, weight=0.0, p_market_yes=0.50)  # weight 0: pure market-follow
    fees = SeriesFees()
    fees.seed("KXHIGHNY", {"fee_type": "quadratic", "fee_multiplier": 1})
    # On the snapshot nothing trades; on the live book a stale 0.50 mid would show 6c of fake edge.
    assert reprice(Live(), "x", [(stale, fair)], EvalConfig(include_maker=False), fees) == []
    cheap = _quote(0.40, 0.42)
    assert reprice(Live(), "x", [(cheap, Fair(cheap.ticker, 0.50, weight=0.0, p_market_yes=0.50))],
                   EvalConfig(include_maker=False), fees) == []


def test_no_side_is_priced_from_yes_bid():
    q = _quote(0.20, 0.22)
    o = evaluate("x", q, Fair(q.ticker, p_yes=0.05, weight=1.0), EvalConfig(include_maker=False))[0]
    assert o.side == "no" and o.price == pytest.approx(0.80) and o.p_fair == pytest.approx(0.95)


def test_taker_size_limited_by_visible_depth():
    q = _quote(0.46, 0.47, size=7)
    o = evaluate("x", q, Fair(q.ticker, p_yes=0.70), EvalConfig(include_maker=False, bankroll=100000))[0]
    assert o.contracts == 7


def test_select_enforces_event_budget_and_one_entry_per_market():
    cfg = EvalConfig(bankroll=1000, max_event_fraction=0.05, max_bet_fraction=0.04, include_maker=True)
    q1 = _quote(0.40, 0.50)
    q2 = Quote.from_api({**RAW_BUCKET, "ticker": "KXHIGHNY-26OCT02-B85.5", "yes_bid_dollars": "0.10", "yes_ask_dollars": "0.12"})
    opps = evaluate("x", q1, Fair(q1.ticker, 0.70), cfg) + evaluate("x", q2, Fair(q2.ticker, 0.30), cfg)
    chosen = select(opps, cfg)
    assert len({o.ticker for o in chosen}) == len(chosen)
    assert sum(o.stake for o in chosen) <= 1000 * 0.05 + 1e-6
    for o in chosen:
        assert o.ev >= cfg.min_edge
