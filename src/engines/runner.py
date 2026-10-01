"""Scan -> re-price -> select -> execute, for every engine.

Snapshot prices from ``/events`` nominate opportunities; each nominee is then
re-priced off its live order book and re-evaluated, so nothing is sized or
sent on a stale quote. Selection applies the portfolio budgets. Execution is a
dry run (recorded to the paper ledger) unless ``live=True``, in which case each
order goes through ``place_guarded_order``: risk governor, Edge Policy, size
caps, journal. Engines never get a private path to the exchange.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from src.engines.arbitrage import Basket, scan_events
from src.engines.evaluate import EvalConfig, Fair, Opportunity, evaluate, select
from src.engines.fees import SeriesFees
from src.engines.market import KalshiPublic, Quote

MAKER_TTL_MIN = 60  # resting quotes expire rather than going stale


@dataclass
class ScanReport:
    opportunities: List[Opportunity] = field(default_factory=list)
    selected: List[Opportunity] = field(default_factory=list)
    baskets: List[Basket] = field(default_factory=list)
    priced: Dict[str, Any] = field(default_factory=dict)  # engine -> summary of what it priced
    notes: List[str] = field(default_factory=list)


def reprice(kp: KalshiPublic, engine: str, pairs: Sequence[tuple], cfg: EvalConfig, fees: SeriesFees,
            log=lambda *_: None) -> List[Opportunity]:
    """Evaluate on snapshots, then re-evaluate every nominee on its live book."""
    nominees = []
    for q, fair in pairs:
        if evaluate(engine, q, fair, cfg, fees.get(q.series_ticker)):
            nominees.append((q, fair))
    out = []
    for q, fair in nominees:
        try:
            kp.book(q.ticker).apply_to(q)
        except Exception as exc:
            log(f"  book {q.ticker}: {exc}")
            continue
        # The engine's market consensus came from the snapshot. Against a fresh book
        # it must come from the fresh book too, or a stale mid passes for edge.
        live = replace(fair, p_market_yes=None)
        out.extend(evaluate(engine, q, live, cfg, fees.get(q.series_ticker)))
    return out


def scan_weather(fetcher, kp: KalshiPublic, cfg: EvalConfig, fees: SeriesFees, report: ScanReport,
                 series: Optional[List[str]] = None, log=lambda *_: None) -> None:
    from src.engines.weather.engine import WeatherEngine, quotes_by_ticker

    eng = WeatherEngine(fetcher, kp, log=log)
    priced = eng.scan(series)
    pairs = list(quotes_by_ticker(priced).values())
    report.priced["weather"] = {
        "events": len(priced),
        "markets": len(pairs),
        "weight": eng.weight,
        "calibrated_through": eng.cal.fitted_through,
        "events_detail": [
            {"event": p.event.event_ticker, "station": p.code, "kind": p.kind, "target": p.target,
             "mu": p.mu, "sigma": p.sigma, "run": p.forecast_run,
             "observed": None if p.observed is None else {"max": p.observed.max_f, "min": p.observed.min_f}}
            for p in priced
        ],
    }
    if not eng.weight:
        why = eng.cal.weight_source or "no backtest recorded yet; run `cli.py engines backtest --save-weight`"
        report.notes.append(
            f"weather: trust weight is 0 ({why}). Only outcomes already decided by today's "
            "observations can trade.")
    report.opportunities.extend(reprice(kp, "weather", pairs, cfg, fees, log))


def scan_arbitrage(kp: KalshiPublic, fees: SeriesFees, report: ScanReport, min_profit: float = 0.01,
                   series: Optional[List[str]] = None, log=lambda *_: None) -> None:
    events = []
    if series:
        for s in series:
            events.extend(kp.iter_events(status="open", series_ticker=s))
    else:
        events = list(kp.iter_events(status="open", max_pages=200))
    report.priced["arbitrage"] = {"events": len(events), "markets": sum(len(e.markets) for e in events)}
    log(f"arbitrage: {len(events)} open events loaded; checking candidates against live books")
    report.baskets.extend(scan_events(events, lambda s: fees.get(s), kp.book, min_profit=min_profit, log=log))


def scan(fetcher, engines: Sequence[str], cfg: EvalConfig, max_orders: int = 20,
         series: Optional[List[str]] = None, log=lambda *_: None) -> ScanReport:
    kp = KalshiPublic(fetcher)
    fees = SeriesFees(fetcher)
    report = ScanReport()
    for name in engines:
        try:
            if name == "weather":
                scan_weather(fetcher, kp, cfg, fees, report, series, log)
            elif name == "arbitrage":
                scan_arbitrage(kp, fees, report, series=series, log=log)
            else:
                report.notes.append(f"unknown engine {name!r}")
        except Exception as exc:  # report the failure; other engines still run
            report.notes.append(f"{name}: scan failed: {exc}")
    report.selected = select(report.opportunities, cfg, max_orders=max_orders)
    return report


# -- execution -----------------------------------------------------------------


def basket_opportunities(b: Basket) -> List[Opportunity]:
    """A basket's legs as taker orders for the ledger / guarded execution."""
    per = b.profit / max(len(b.legs), 1)
    return [
        Opportunity(engine="arbitrage", ticker=leg.ticker, event_ticker=b.event_ticker,
                    series_ticker=b.series_ticker, title=b.title, subtitle="", side=leg.side,
                    role="taker", price=leg.price, fee=leg.fee, p_engine=1.0, p_market=None,
                    p_fair=1.0, ev=round(per, 4), roi=round(b.roi, 4), kelly=0.0,
                    contracts=b.count, stake=round(b.count * (leg.price + leg.fee), 2),
                    expected_profit=round(b.count * per, 2), hours_to_close=None,
                    rationale=f"{b.kind} basket: {b.note}", meta={"basket": b.kind})
        for leg in b.legs
    ]


async def execute_live(client, orders: Sequence[Opportunity], log=print) -> List[Dict[str, Any]]:
    """Send each order through the full guard stack. Taker = immediate-or-cancel."""
    from src.agent.toolbelt import place_guarded_order

    results = []
    for o in orders:
        res = await place_guarded_order(
            client, ticker=o.ticker, side=o.side, count=o.contracts, price=o.price,
            type_="market" if o.role == "taker" else "limit",
            rationale=f"[{o.engine}] {o.rationale}"[:500],
            est_prob=o.p_fair, category=o.series_ticker, dry=False,
            expiration_ts=None if o.role == "taker" else int(time.time()) + MAKER_TTL_MIN * 60,
            strategy=f"engine:{o.engine}", method=f"engine:{o.engine}",
        )
        results.append({"ticker": o.ticker, "action": o.action, **res})
        log(f"  {o.action} {o.ticker} x{o.contracts}: {'ok' if res.get('ok') else res.get('reason')}")
    return results


async def execute_basket_live(client, b: Basket, log=print) -> Dict[str, Any]:
    """Legs scarcest-first as immediate-or-cancel; later legs shrink to what filled.

    Kalshi has no atomic multi-leg order, so a leg that comes back short is the
    one real risk here. Sizing every later leg to the filled count keeps the
    position a complete basket whenever any of it fills.
    """
    from src.agent.toolbelt import place_guarded_order

    # Preflight every leg through the same guards (governor, policy, size caps)
    # before sending any of them: a refusal on leg 3 after legs 1-2 filled would
    # leave an unhedged position.
    target = b.count
    for leg in b.legs:
        pre = await place_guarded_order(
            client, ticker=leg.ticker, side=leg.side, count=target, price=leg.price, type_="market",
            category=b.series_ticker, dry=True, method="engine:arbitrage")
        if not pre.get("ok"):
            return {"basket": b.kind, "event": b.event_ticker, "complete": False, "legs": [],
                    "note": f"preflight refused {leg.ticker}: {pre.get('reason')}; nothing sent"}
        target = min(target, int(pre.get("count") or 0))
    if target < 1:
        return {"basket": b.kind, "event": b.event_ticker, "complete": False, "legs": [],
                "note": "preflight sized the basket to 0; nothing sent"}

    fills = []
    for leg in sorted(b.legs, key=lambda l: l.available):
        res = await place_guarded_order(
            client, ticker=leg.ticker, side=leg.side, count=target, price=leg.price, type_="market",
            rationale=f"[arbitrage] {b.kind} {b.event_ticker}: {b.note}"[:500], est_prob=None,
            category=b.series_ticker, dry=False, strategy="engine:arbitrage", method="engine:arbitrage",
        )
        filled = int(float(res.get("fill_count") or 0)) if res.get("ok") else 0
        fills.append({"ticker": leg.ticker, "side": leg.side, "wanted": target, "filled": filled,
                      "reason": None if res.get("ok") else res.get("reason")})
        if filled < target:
            log(f"  leg {leg.ticker} filled {filled}/{target}; remaining legs sized to {filled}")
            target = filled
        if target == 0:
            break
    complete = target > 0 and all(f["filled"] >= target for f in fills) and len(fills) == len(b.legs)
    return {"basket": b.kind, "event": b.event_ticker, "complete": complete, "legs": fills,
            "note": None if complete else "INCOMPLETE BASKET: unhedged legs may be open; check positions."}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
