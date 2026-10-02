"""Runner: shadow evaluation, live plan de-duplication, basket fill accounting, trust promotion."""

import pytest

from src.engines import ledger
from src.engines.arbitrage import Basket, Leg
from src.engines.evaluate import EvalConfig, Fair, Opportunity
from src.engines.market import Quote
from src.engines.runner import ScanReport, basket_fill_report, live_plan
from src.engines.trust import promotion_from_paper


def _basket(tickers=("A", "B"), count=10):
    return Basket("all-NO", "E", "KXHIGHNY", "", [Leg(t, "no", 0.5, 0.01, 50) for t in tickers],
                  payout=1.0, cost=1.02, profit=0.02, count=count, risk_free=True, note="")


def _opp(ticker):
    return Opportunity("weather", ticker, "E", "KXHIGHNY", "", "", "yes", "taker", 0.4, 0.02, 0.6, 0.45, 0.6,
                       0.18, 0.4, 0.1, 5, 2.1, 0.9, 20.0, "r")


def test_live_plan_never_doubles_a_market():
    report = ScanReport(selected=[_opp("A"), _opp("C")], baskets=[_basket()])
    baskets, orders = live_plan(report)
    assert [o.ticker for o in orders] == ["C"] and len(baskets) == 1


def test_basket_overfill_is_reported_unhedged():
    b = _basket()
    fills = [{"ticker": "A", "filled": 10}, {"ticker": "B", "filled": 6}]
    rep = basket_fill_report(b, fills)
    assert not rep["complete"] and rep["unhedged"] == {"A": 4} and rep["baskets_held"] == 6
    rep = basket_fill_report(b, [{"ticker": "A", "filled": 6}, {"ticker": "B", "filled": 6}])
    assert rep["complete"] and rep["unhedged"] == {}
    rep = basket_fill_report(b, [{"ticker": "A", "filled": 0}])  # stopped after the first leg
    assert not rep["complete"] and rep["note"] == "Nothing filled."


def _shadow(i, won, p_engine, p_market, certain=False):
    o = _opp(f"KXHIGHNY-26OCT{i:02d}-B1")
    o.meta.update(shadow=True, bound_certain=certain)
    r = ledger.record_from_opportunity(o)
    r.update(event_ticker=f"E{i}", p_engine=p_engine, p_market=p_market,
             outcome={"won": won, "pnl": 3.0 if won else -2.0, "result": "yes" if won else "no"})
    return r


def test_promotion_needs_skill_and_profit_and_ignores_certain_orders():
    good = [_shadow(i, i % 4 != 0, 0.75 if i % 4 else 0.3, 0.5) for i in range(120)]
    v = promotion_from_paper(good, "weather")
    assert v["verdict"] == "earned" and v["weight"] > 0
    assert promotion_from_paper(good[:20], "weather")["verdict"] == "not enough settled shadow orders"
    certain = [_shadow(i, True, 1.0, 0.9, certain=True) for i in range(200)]
    assert promotion_from_paper(certain, "weather")["settled"] == 0
    assert promotion_from_paper(good, "games")["settled"] == 0  # other engines' records don't count


def test_shadow_records_are_flagged_and_scored_separately():
    recs = [_shadow(1, True, 0.8, 0.5), ledger.record_from_opportunity(_opp("X"))]
    recs[1]["outcome"] = {"won": False, "pnl": -2.0, "result": "no"}
    s = ledger.summarize(recs)
    assert "weather (shadow)" in s and "weather" in s
    assert s["weather (shadow)"]["log_loss_engine"] < s["weather (shadow)"]["log_loss_market"]
