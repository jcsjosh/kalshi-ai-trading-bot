"""Turn an engine's probability and a live quote into a sized, fee-aware decision.

The rules every engine shares:

1. **Shrink toward the market.** A liquid Kalshi book is usually sharp, so an
   engine's view only moves the fair price as far as its measured skill allows:
   ``logit(p_fair) = w * logit(p_engine) + (1 - w) * logit(p_market)``. An engine
   with no demonstrated edge has ``w = 0`` and can never trade.
2. **Pay real fees.** Each entry is priced with its series' fee schedule at the
   suggested size (taker at the ask; maker one tick inside the spread).
3. **Demand a margin that grows with uncertainty.** ``ev >= min_edge + z * stderr``.
4. **Size with fractional Kelly**, capped per bet, per event and by visible depth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from src.engines.fees import DEFAULT_SCHEDULE, FeeSchedule
from src.engines.market import Quote

EPS = 1e-4
TICK = 0.01


def clamp_prob(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def logit(p: float) -> float:
    p = clamp_prob(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def shrink(p_engine: float, p_market: Optional[float], weight: float) -> float:
    """Blend an engine probability with the market's in log-odds space."""
    if p_market is None:
        return clamp_prob(p_engine)
    w = min(max(weight, 0.0), 1.0)
    return sigmoid(w * logit(p_engine) + (1.0 - w) * logit(p_market))


def kelly_fraction(win_prob: float, cost: float) -> float:
    """Full-Kelly bankroll fraction for a contract costing ``cost`` that pays $1."""
    if cost <= 0.0 or cost >= 1.0 or win_prob <= cost:
        return 0.0
    return (win_prob - cost) / (1.0 - cost)


@dataclass
class Fair:
    """An engine's opinion of one market."""

    ticker: str
    p_yes: float
    weight: float = 1.0  # trust in the engine vs the market, earned in backtests
    stderr: float = 0.0  # uncertainty of p_yes; raises the edge bar
    p_market_yes: Optional[float] = None  # engine-supplied market consensus (e.g. de-vigged)
    rationale: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    bankroll: float = 1000.0
    kelly: float = 0.25  # fraction of full Kelly
    max_bet_fraction: float = 0.03  # of bankroll, per market
    max_event_fraction: float = 0.06  # of bankroll, across one event's markets
    max_total_fraction: float = 0.30  # of bankroll, across one scan
    min_edge: float = 0.03  # dollars of EV per contract after fees
    z: float = 1.0  # multiples of stderr added to min_edge
    min_price: float = 0.03
    max_price: float = 0.97
    include_maker: bool = True
    maker_adverse_selection: float = 0.01  # resting orders fill when you're wrong more often
    max_book_share: float = 1.0  # fraction of visible ask size we may take
    max_contracts: int = 500


@dataclass
class Opportunity:
    engine: str
    ticker: str
    event_ticker: str
    series_ticker: str
    title: str
    subtitle: str
    side: str
    role: str  # taker | maker
    price: float
    fee: float  # per contract at size
    p_engine: float  # engine probability for this side
    p_market: Optional[float]  # market-implied probability for this side
    p_fair: float  # blended probability for this side
    ev: float  # dollars per contract after fees
    roi: float
    kelly: float
    contracts: int
    stake: float
    expected_profit: float
    hours_to_close: Optional[float]
    rationale: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def action(self) -> str:
        verb = "TAKE" if self.role == "taker" else "POST"
        return f"{verb} BUY {self.side.upper()} @ {self.price:.2f}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action
        return d


def _entries(q: Quote, side: str, cfg: EvalConfig) -> List[tuple]:
    out = []
    ask, bid = q.ask(side), q.bid(side)
    if ask is not None:
        out.append(("taker", ask, q.ask_size(side)))
    if cfg.include_maker and ask is not None:
        post = round((bid if bid is not None else 0.0) + TICK, 2)
        if post < ask - 1e-9 and post >= TICK:
            out.append(("maker", post, float("inf")))
    return out


def evaluate(
    engine: str,
    quote: Quote,
    fair: Fair,
    cfg: EvalConfig,
    fees: FeeSchedule = DEFAULT_SCHEDULE,
) -> List[Opportunity]:
    """Every positive-EV way into ``quote`` under ``fair`` (possibly none).

    Taker entries are valued at the blended fair probability. Maker entries
    are valued as if filled: the market's estimate is then the price we posted,
    so they need the engine itself to disagree with that price.
    """
    p_mkt_yes = fair.p_market_yes if fair.p_market_yes is not None else quote.mid
    p_fair_yes = shrink(fair.p_yes, p_mkt_yes, fair.weight)
    hours = quote.hours_to_close()
    opps: List[Opportunity] = []
    for side in ("yes", "no"):
        p_side = p_fair_yes if side == "yes" else 1.0 - p_fair_yes
        p_eng = fair.p_yes if side == "yes" else 1.0 - fair.p_yes
        p_mk = None if p_mkt_yes is None else (p_mkt_yes if side == "yes" else 1.0 - p_mkt_yes)
        for role, price, depth in _entries(quote, side, cfg):
            if not cfg.min_price <= price <= cfg.max_price:
                continue
            if role == "maker":
                # A resting bid fills when someone sells to us at our price, so on a
                # fill the market's view is our price, not the mid. Only the engine's
                # own disagreement with that price is edge; with weight 0 there is none.
                p_used = shrink(p_eng, price, fair.weight)
                win = p_used - cfg.maker_adverse_selection
            else:
                p_used = win = p_side
            fee1 = fees.per_contract(1, price, role)
            f_star = kelly_fraction(win, price + fee1)
            if f_star <= 0:
                continue
            stake_cap = cfg.bankroll * min(f_star * cfg.kelly, cfg.max_bet_fraction)
            n = int(stake_cap // (price + fee1))
            if role == "taker" and depth != float("inf"):
                n = min(n, int(depth * cfg.max_book_share))
            n = min(n, cfg.max_contracts)
            if n < 1:
                continue
            fee = fees.per_contract(n, price, role)
            ev = win - price - fee
            if ev < cfg.min_edge + cfg.z * fair.stderr:
                continue
            cost = price + fee
            opps.append(
                Opportunity(
                    engine=engine,
                    ticker=quote.ticker,
                    event_ticker=quote.event_ticker,
                    series_ticker=quote.series_ticker,
                    title=quote.title,
                    subtitle=quote.subtitle,
                    side=side,
                    role=role,
                    price=round(price, 4),
                    fee=round(fee, 4),
                    p_engine=round(p_eng, 4),
                    p_market=None if p_mk is None else round(p_mk, 4),
                    p_fair=round(p_used, 4),  # the probability this entry was actually priced at
                    ev=round(ev, 4),
                    roi=round(ev / cost, 4),
                    kelly=round(kelly_fraction(win, cost), 4),
                    contracts=n,
                    stake=round(n * cost, 2),
                    expected_profit=round(n * ev, 2),
                    hours_to_close=None if hours is None else round(hours, 2),
                    rationale=fair.rationale,
                    meta={**fair.meta, "fee_type": fees.fee_type, "fee_multiplier": fees.multiplier},
                )
            )
    return opps


def select(opps: Iterable[Opportunity], cfg: EvalConfig, max_orders: int = 20) -> List[Opportunity]:
    """Greedy portfolio: best expected profit first, one entry per market,
    never both sides of a market, within per-event and total stake budgets."""
    chosen: List[Opportunity] = []
    used_markets: set = set()
    per_event: Dict[str, float] = {}
    total = 0.0
    event_cap = cfg.bankroll * cfg.max_event_fraction
    total_cap = cfg.bankroll * cfg.max_total_fraction
    for o in sorted(opps, key=lambda o: (o.expected_profit, o.roi), reverse=True):
        if len(chosen) >= max_orders or o.ticker in used_markets:
            continue
        room = min(event_cap - per_event.get(o.event_ticker, 0.0), total_cap - total)
        if room < o.price + o.fee:
            continue
        if o.stake > room:
            o = _resize(o, int(room // (o.price + o.fee)))
            if o.contracts < 1 or o.ev < cfg.min_edge:
                continue
        chosen.append(o)
        used_markets.add(o.ticker)
        per_event[o.event_ticker] = per_event.get(o.event_ticker, 0.0) + o.stake
        total += o.stake
    return chosen


def _resize(o: Opportunity, n: int) -> Opportunity:
    """Shrink an opportunity to ``n`` contracts, re-pricing the (rounded) fee."""
    sched = FeeSchedule(o.meta.get("fee_type", "quadratic"), o.meta.get("fee_multiplier", 1.0))
    fee = sched.per_contract(n, o.price, o.role) if n > 0 else o.fee
    adverse = o.p_fair - (o.ev + o.price + o.fee)  # maker haircut baked into the original ev
    ev = o.p_fair - adverse - o.price - fee
    d = asdict(o)
    d.update(
        contracts=n,
        fee=round(fee, 4),
        ev=round(ev, 4),
        roi=round(ev / (o.price + fee), 4),
        stake=round(n * (o.price + fee), 2),
        expected_profit=round(n * ev, 2),
    )
    return Opportunity(**d)
