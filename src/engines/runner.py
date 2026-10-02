"""Scan -> re-price -> select -> execute, for every engine.

Snapshot prices from ``/events`` nominate opportunities; each nominee is then
re-priced off its live order book and re-evaluated, so nothing is sized or
sent on a stale quote. Selection applies the portfolio budgets. Execution is a
dry run (recorded to the paper ledger) unless ``live=True``, in which case each
order goes through ``place_guarded_order``: risk governor, Edge Policy, size
caps, journal. Engines never get a private path to the exchange.

Model engines are also evaluated in **shadow**: the same markets priced as if
the engine were fully trusted (``w = 1``). Shadow orders are never sent; they
go to the paper ledger so an untrusted engine still builds the forward record
it needs to earn trust (``cli.py engines promote``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.engines.arbitrage import Basket, scan_events
from src.engines.evaluate import EvalConfig, Fair, Opportunity, evaluate, select
from src.engines.fees import SeriesFees
from src.engines.market import Book, KalshiPublic, Quote

MAKER_TTL_MIN = 60  # resting quotes expire rather than going stale
SHADOW_WEIGHT = 1.0

Log = Callable[[str], None]
Pairs = List[Tuple[Quote, Fair]]


@dataclass
class ScanContext:
    fetcher: Any
    kp: KalshiPublic
    fees: SeriesFees
    cfg: EvalConfig
    series: Optional[List[str]] = None
    log: Log = lambda *_: None
    books: Dict[str, Book] = field(default_factory=dict)

    def book(self, ticker: str) -> Book:
        if ticker not in self.books:
            self.books[ticker] = self.kp.book(ticker)
        return self.books[ticker]


@dataclass
class ScanReport:
    opportunities: List[Opportunity] = field(default_factory=list)
    selected: List[Opportunity] = field(default_factory=list)
    shadow: List[Opportunity] = field(default_factory=list)
    baskets: List[Basket] = field(default_factory=list)
    priced: Dict[str, Any] = field(default_factory=dict)  # engine -> summary of what it priced
    notes: List[str] = field(default_factory=list)

    def to_dict(self, detail: bool = False) -> Dict[str, Any]:
        priced = self.priced if detail else {
            k: {kk: vv for kk, vv in v.items() if kk != "detail"} for k, v in self.priced.items()}
        out = {
            "selected": [o.to_dict() for o in self.selected],
            "shadow": [o.to_dict() for o in self.shadow],
            "baskets": [b.to_dict() for b in self.baskets],
            "priced": priced,
            "notes": self.notes,
        }
        if detail:
            out["opportunities"] = [o.to_dict() for o in self.opportunities]
        return out


def reprice(ctx: ScanContext, engine: str, pairs: Pairs) -> List[Opportunity]:
    """Evaluate on snapshots, then re-evaluate every nominee on its live book."""
    out = []
    for q, fair in pairs:
        if not evaluate(engine, q, fair, ctx.cfg, ctx.fees.get(q.series_ticker)):
            continue
        try:
            ctx.book(q.ticker).apply_to(q)
        except Exception as exc:
            ctx.log(f"  book {q.ticker}: {exc}")
            continue
        # The engine's market consensus came from the snapshot. Against a fresh book
        # it must come from the fresh book too, or a stale mid passes for edge.
        out.extend(evaluate(engine, q, replace(fair, p_market_yes=None), ctx.cfg, ctx.fees.get(q.series_ticker)))
    return out


# -- model engines: each returns (pairs, summary, notes) -------------------------------


def price_weather(ctx: ScanContext) -> Tuple[Pairs, Dict[str, Any], List[str]]:
    from src.engines.trust import weight_for
    from src.engines.weather.engine import WeatherEngine, quotes_by_ticker

    eng = WeatherEngine(ctx.fetcher, ctx.kp, log=ctx.log)
    weight, source = weight_for("weather", default=float(eng.cal.weight or 0.0))
    eng.cal.weight = weight
    priced = eng.scan(ctx.series)
    pairs = list(quotes_by_ticker(priced).values())
    summary = {
        "events": len(priced), "markets": len(pairs), "weight": weight,
        "calibrated_through": eng.cal.fitted_through,
        "detail": [{"event": p.event.event_ticker, "station": p.code, "kind": p.kind, "target": p.target,
                    "mu": p.mu, "sigma": p.sigma, "run": p.forecast_run,
                    "observed": None if p.observed is None else {"max": p.observed.max_f, "min": p.observed.min_f}}
                   for p in priced],
    }
    notes = []
    if not weight:
        why = source or eng.cal.weight_source or "no evidence yet; run `cli.py engines backtest --save-weight`"
        notes.append(f"weather: trust weight is 0 ({why}). Only outcomes already decided by today's "
                     "observations can trade; the forecast itself is paper-tested in shadow.")
    return pairs, summary, notes


def price_games(ctx: ScanContext) -> Tuple[Pairs, Dict[str, Any], List[str]]:
    from src.engines.games import GamesEngine
    from src.engines.trust import weight_for

    try:
        import sports_skills  # noqa: F401
    except ImportError:
        return [], {"events": 0, "markets": 0}, [
            "games: needs the optional `sports-skills` package (pip install sports-skills)"]
    weight, source = weight_for("games")
    priced = GamesEngine(ctx.fetcher, ctx.kp, weight=weight, log=ctx.log).scan()
    pairs = [(q, f) for g in priced for q in g.event.markets for f in g.fairs if f.ticker == q.ticker]
    summary = {"events": len(priced), "markets": len(pairs), "weight": weight,
               "detail": [{"event": g.event.event_ticker, "sport": g.sport, "start": g.start,
                           "sources": g.sources} for g in priced]}
    notes = [] if weight else [f"games: trust weight is 0 ({source or 'no settled shadow record yet'}); "
                               "paper-tested in shadow until `cli.py engines promote games` says otherwise."]
    return pairs, summary, notes


MODEL_ENGINES: Dict[str, Callable[[ScanContext], Tuple[Pairs, Dict[str, Any], List[str]]]] = {
    "weather": price_weather,
    "games": price_games,
}


# -- arbitrage ---------------------------------------------------------------------


def scan_arbitrage(ctx: ScanContext, report: ScanReport, min_profit: float = 0.01) -> None:
    from src.engines.crosscheck import oracle3_verdict

    if ctx.series:
        events = [e for s in ctx.series for e in ctx.kp.iter_events(status="open", series_ticker=s)]
    else:
        events = list(ctx.kp.iter_events(status="open", max_pages=200))
    report.priced["arbitrage"] = {"events": len(events), "markets": sum(len(e.markets) for e in events)}
    ctx.log(f"arbitrage: {len(events)} open events loaded; checking candidates against live books")
    baskets = scan_events(events, ctx.fees.get, ctx.book, min_profit=min_profit, log=ctx.log)
    event_cap = ctx.cfg.bankroll * ctx.cfg.max_event_fraction
    for b in baskets:
        if not b.risk_free:
            continue
        # Same per-event budget as every other trade.
        if b.count * b.cost > event_cap:
            b.count = int(event_cap // b.cost)
            b.meta["capped_by"] = f"{ctx.cfg.max_event_fraction:.0%} per-event budget"
        verdict = oracle3_verdict(b, ctx.fees.get(b.series_ticker))
        if verdict is not None:
            b.meta["oracle3"] = verdict
            if not verdict["agrees"]:
                b.risk_free = False
                b.note += f" oracle3 does not confirm ({verdict}); not traded."
    report.baskets.extend(b for b in baskets if b.count > 0 or not b.risk_free)


def scan(fetcher, engines: Sequence[str], cfg: EvalConfig, max_orders: int = 20,
         series: Optional[List[str]] = None, log: Log = lambda *_: None) -> ScanReport:
    ctx = ScanContext(fetcher, KalshiPublic(fetcher), SeriesFees(fetcher), cfg, series, log)
    report = ScanReport()
    shadow_opps: List[Opportunity] = []
    for name in engines:
        if name != "arbitrage" and name not in MODEL_ENGINES:
            report.notes.append(f"unknown engine {name!r}")
            continue
        try:
            if name == "arbitrage":
                scan_arbitrage(ctx, report)
                continue
            pairs, summary, notes = MODEL_ENGINES[name](ctx)
            report.priced[name] = summary
            report.notes.extend(notes)
            report.opportunities.extend(reprice(ctx, name, pairs))
            # Shadow: the raw model, fully trusted. Outcomes already certain from
            # observations need no skill, so they stay out of the model's test.
            shadow_pairs = [(q, replace(f, weight=SHADOW_WEIGHT)) for q, f in pairs
                            if not f.meta.get("bound_certain")]
            for o in reprice(ctx, name, shadow_pairs):
                o.meta["shadow"] = True
                shadow_opps.append(o)
        except Exception as exc:  # report the failure; other engines still run
            report.notes.append(f"{name}: scan failed: {exc}")
    report.selected = select(report.opportunities, cfg, max_orders=max_orders)
    report.shadow = select(shadow_opps, cfg, max_orders=max_orders)
    return report


# -- execution -----------------------------------------------------------------


def basket_opportunities(b: Basket) -> List[Opportunity]:
    """A basket's legs as taker orders for the ledger."""
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


def live_plan(report: ScanReport) -> Tuple[List[Basket], List[Opportunity]]:
    """What a live run sends: risk-free baskets first, then single orders on
    markets no basket touches (never two positions in one market from one run)."""
    baskets = [b for b in report.baskets if b.risk_free and b.count > 0]
    taken = {leg.ticker for b in baskets for leg in b.legs}
    return baskets, [o for o in report.selected if o.ticker not in taken]


async def execute_live(client, orders: Sequence[Opportunity], log: Log = print) -> List[Dict[str, Any]]:
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


def basket_fill_report(b: Basket, fills: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Complete only if every leg filled exactly the same count. An earlier leg
    that filled more than a later one leaves that excess unhedged."""
    final = min((f["filled"] for f in fills), default=0) if len(fills) == len(b.legs) else 0
    excess = {f["ticker"]: f["filled"] - final for f in fills if f["filled"] > final}
    complete = final > 0 and not excess
    note = None
    if not complete:
        note = ("INCOMPLETE BASKET: unhedged contracts " +
                ", ".join(f"{t} x{n}" for t, n in excess.items()) + ". Check positions; close or re-hedge."
                if excess else "Nothing filled.")
    return {"basket": b.kind, "event": b.event_ticker, "complete": complete, "baskets_held": final,
            "unhedged": excess, "legs": fills, "note": note}


async def execute_basket_live(client, b: Basket, log: Log = print) -> Dict[str, Any]:
    """Legs scarcest-first as immediate-or-cancel; later legs shrink to what filled.

    Kalshi has no atomic multi-leg order. The preflight runs every leg through
    the same guards (governor, policy, caps) *and* checks the whole basket's
    cost against cash before anything is sent; after that, a short fill is the
    one remaining risk, and it is reported leg by leg.
    """
    from src.agent.toolbelt import place_guarded_order

    def refuse(why: str) -> Dict[str, Any]:
        return {"basket": b.kind, "event": b.event_ticker, "complete": False, "legs": [],
                "note": f"{why}; nothing sent"}

    target = b.count
    for leg in b.legs:
        pre = await place_guarded_order(
            client, ticker=leg.ticker, side=leg.side, count=target, price=leg.price, type_="market",
            category=b.series_ticker, dry=True, method="engine:arbitrage")
        if not pre.get("ok"):
            return refuse(f"preflight refused {leg.ticker}: {pre.get('reason')}")
        target = min(target, int(pre.get("count") or 0))
    bal = await client.get_balance()
    cash = int(bal.get("balance", 0) or 0) / 100.0
    per_basket = sum(leg.price + leg.fee for leg in b.legs)
    target = min(target, int(cash // per_basket)) if per_basket > 0 else 0
    if target < 1:
        return refuse(f"basket costs ${per_basket:.2f} each and cash is ${cash:.2f}")

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
    return basket_fill_report(b, fills)
