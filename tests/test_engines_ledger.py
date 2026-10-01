"""Paper ledger: recording, maker-fill inference, settlement and scoring."""

from datetime import datetime, timedelta, timezone

import pytest

from src.engines import ledger
from src.engines.evaluate import Opportunity
from src.engines.runner import basket_opportunities
from src.engines.arbitrage import Basket, Leg

T0 = datetime(2026, 10, 1, 15, tzinfo=timezone.utc)


def opp(role="taker", side="yes", price=0.40, ticker="KXHIGHNY-26OCT02-B83.5", p_fair=0.55, p_market=0.42):
    return Opportunity(engine="weather", ticker=ticker, event_ticker="KXHIGHNY-26OCT02", series_ticker="KXHIGHNY",
                       title="", subtitle="", side=side, role=role, price=price, fee=0.02, p_engine=p_fair,
                       p_market=p_market, p_fair=p_fair, ev=round(p_fair - price - 0.02, 4), roi=0.3, kelly=0.1,
                       contracts=10, stake=4.2, expected_profit=1.3, hours_to_close=30.0, rationale="test")


class FakeKalshi:
    def __init__(self, results, trades=()):
        self.results, self.trades = results, list(trades)

    def get(self, path, params=None):
        if path.startswith("/markets/trades"):
            return {"trades": self.trades}
        return {"market": {"result": self.results.get(path.rsplit("/", 1)[-1], "")}}


def test_roundtrip_and_settle_taker(tmp_path):
    path = tmp_path / "l.jsonl"
    ledger.append([ledger.record_from_opportunity(opp(), now=T0)], path)
    recs = ledger.load(path)
    assert recs[0]["filled"] is True and recs[0]["outcome"] is None
    assert ledger.settle(recs, FakeKalshi({})) == 0  # not settled yet
    assert ledger.settle(recs, FakeKalshi({"KXHIGHNY-26OCT02-B83.5": "yes"})) == 1
    assert recs[0]["outcome"]["pnl"] == pytest.approx(10 * (1 - 0.40 - 0.02))
    ledger.save(recs, path)
    assert ledger.load(path)[0]["outcome"]["won"] is True


def test_maker_fill_needs_a_print_at_or_through_our_price():
    rec = ledger.record_from_opportunity(opp(role="maker", side="yes", price=0.40), now=T0)
    after = (T0 + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    before = (T0 - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    assert ledger.maker_filled(rec, [{"created_time": after, "yes_price_dollars": "0.40"}])
    assert not ledger.maker_filled(rec, [{"created_time": after, "yes_price_dollars": "0.41"}])
    assert not ledger.maker_filled(rec, [{"created_time": before, "yes_price_dollars": "0.30"}])
    no = ledger.record_from_opportunity(opp(role="maker", side="no", price=0.30), now=T0)
    assert ledger.maker_filled(no, [{"created_time": after, "yes_price_dollars": "0.70"}])  # NO at 0.30


def test_unfilled_maker_settles_flat():
    recs = [ledger.record_from_opportunity(opp(role="maker", price=0.40), now=T0)]
    ledger.settle(recs, FakeKalshi({"KXHIGHNY-26OCT02-B83.5": "no"}, trades=[]))
    assert recs[0]["filled"] is False and recs[0]["outcome"]["pnl"] == 0.0


def test_summary_scores_engine_against_market():
    recs = [ledger.record_from_opportunity(opp(p_fair=0.7, p_market=0.5, ticker=f"T{i}"), now=T0) for i in range(4)]
    for i, r in enumerate(recs):
        won = i < 3
        r["outcome"] = {"won": won, "pnl": 10 * ((1 if won else 0) - 0.42), "result": "yes" if won else "no"}
    s = ledger.summarize(recs)["weather"]
    assert s["settled"] == 4 and s["wins"] == 3
    assert s["log_loss_engine"] < s["log_loss_market"]  # 0.7 called 3/4 better than 0.5


def test_basket_legs_become_taker_orders():
    b = Basket(kind="all-NO", event_ticker="E", series_ticker="KXHIGHNY", title="", legs=[
        Leg("A", "no", 0.95, 0.01, 50), Leg("B", "no", 0.60, 0.02, 80)], payout=1.0, cost=1.58,
        profit=0.0, count=10, risk_free=True, note="")
    legs = basket_opportunities(b)
    assert [o.role for o in legs] == ["taker", "taker"] and all(o.contracts == 10 for o in legs)
