"""Walk-forward backtest of the weather engine against settled Kalshi markets.

The point is to *measure* edge, not to manufacture it, so every input is
restricted to what was knowable at the decision time:

* **Forecast**: the latest NBM run published (run time + 2h) before the decision.
* **Calibration**: refit each decision day on forecast/outcome pairs whose CLI
  report was already out (target day <= decision day - 1). Training history
  can start months before the first traded day; it uses IEM's CLI archive,
  not Kalshi.
* **Trust in the model vs the market** (the shrinkage weight ``w``): fit on
  earlier decisions' bucket outcomes only. Until enough history exists,
  ``w = 0`` and the engine cannot trade.
* **Prices**: Kalshi's own hourly candle at the decision hour, buying at the ask
  as a taker (resting-order fills can't be known from candles, so none are
  assumed). Exact per-series fees.

Schedules: ``day_ahead`` decides at 20:00 UTC the day before (afternoon in the
US, after the 18Z NBM run); ``same_day`` decides at 14:00 UTC on the day
(morning, before the high is set; highs only, since by then the overnight low
has already been observed by everyone in the market).

Known limits, stated in every report: candle quotes stand in for executable
depth (sized at an assumed ``depth`` contracts per level), and Kalshi only
serves candles for markets settled after its historical cutoff, so the traded
window is recent (~2 months as of 2026-10).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from src.engines.evaluate import EvalConfig, Fair, evaluate, logit, select, sigmoid
from src.engines.fees import FeeSchedule
from src.engines.market import Event, KalshiPublic, Quote
from src.engines.weather.calibration import Sample, WeatherCalibration
from src.engines.weather.climate import fetch_cli_year
from src.engines.weather.model import TempDistribution
from src.engines.weather.nbm import NbmForecast, NbmIndex, fetch_nbs
from src.engines.weather.stations import STATIONS, date_from_event, kind_from_rules, station_from_rules

SCHEDULES = {"day_ahead": (-1, 20), "same_day": (0, 14)}
TEMP_SERIES_PREFIXES = ("KXHIGH", "KXLOW", "KXHOUHIGH", "KXDENHIGH", "KXPHILHIGH")
DEFAULT_OUT = Path("data/engines/backtest")


def decision_time(target: date, schedule: str) -> datetime:
    days, hour = SCHEDULES[schedule]
    d = target + timedelta(days=days)
    return datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc)


@dataclass
class SettledEvent:
    event: Event
    code: str
    kind: str
    target: date
    actual: int


@dataclass
class BacktestConfig:
    start: date
    end: date
    train_start: date
    series: Optional[List[str]] = None
    schedules: Tuple[str, ...] = ("day_ahead", "same_day")
    bankroll: float = 1000.0
    min_edge: float = 0.03
    kelly: float = 0.25
    depth: float = 100.0  # assumed contracts available at the candle's ask
    min_blend_rows: int = 300
    fixed_weight: Optional[float] = None  # skip the walk-forward weight fit
    min_calib_samples: int = 60


# -- data collection -----------------------------------------------------------


def discover_temperature_series(kp: KalshiPublic) -> List[str]:
    out = []
    for s in kp.list_series("Climate and Weather"):
        t = s.get("ticker", "")
        if s.get("frequency") == "daily" and t.startswith(TEMP_SERIES_PREFIXES):
            out.append(t)
    return sorted(out)


def collect_events(kp: KalshiPublic, series: List[str], start: date, end: date, log=print) -> List[SettledEvent]:
    out: List[SettledEvent] = []
    for s in series:
        n, older = 0, 0
        for ev in kp.iter_events(status="settled", series_ticker=s, cache=True, cache_ttl=6 * 3600):
            target = date_from_event(ev.event_ticker)
            if target is None or target > end:
                continue
            if target < start:
                older += 1  # newest-first; tolerate a little ordering noise, then stop paging
                if older >= 20:
                    break
                continue
            rules = next((m.rules for m in ev.markets if m.rules), "")
            st, kind = station_from_rules(rules), kind_from_rules(rules)
            vals = [m.expiration_value for m in ev.markets if m.expiration_value is not None]
            if not st or not kind or not vals:
                continue
            actual = int(round(vals[0]))
            # Sanity: exactly the markets containing the reported value settled YES.
            if any((m.contains(actual) is True) != (m.result == "yes") for m in ev.markets if m.result in ("yes", "no")):
                continue
            out.append(SettledEvent(ev, st[0], kind, target, actual))
            n += 1
        if n:
            log(f"  {s}: {n} settled events")
    return out


def collect_nbs(fetcher, codes, start: date, end: date, log=print) -> Dict[str, NbmIndex]:
    out = {}
    s = datetime(start.year, start.month, start.day, tzinfo=timezone.utc) - timedelta(days=3)
    e = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)
    recent = e > datetime.now(timezone.utc) - timedelta(days=2)
    for code in sorted(codes):
        rows = fetch_nbs(fetcher, STATIONS[code].icao, s, e, cache_ttl=6 * 3600 if recent else None)
        out[code] = NbmIndex(rows)
        log(f"  NBM {code}: {len(rows)} forecast rows")
    return out


def collect_cli(fetcher, codes, start: date, end: date) -> Dict[str, Dict[date, Tuple[Optional[int], Optional[int]]]]:
    this_year = datetime.now(timezone.utc).year
    out: Dict[str, Dict] = {}
    for code in sorted(codes):
        out[code] = {}
        for year in range(start.year, end.year + 1):
            ttl = 6 * 3600 if year >= this_year else None
            out[code].update(fetch_cli_year(fetcher, STATIONS[code].icao, year, cache_ttl=ttl))
    return out


# -- calibration samples ----------------------------------------------------


def build_samples(nbs, cli, schedule: str, start: date, end: date) -> List[Sample]:
    """One (forecast, outcome) pair per station, kind and day, as the schedule would have seen it."""
    out = []
    for code, index in nbs.items():
        truth = cli.get(code, {})
        d = start
        while d <= end:
            hi_lo = truth.get(d)
            if hi_lo:
                for kind, actual in (("high", hi_lo[0]), ("low", hi_lo[1])):
                    if actual is None or (kind == "low" and schedule == "same_day"):
                        continue
                    f = index.latest(d, kind, decision_time(d, schedule))
                    if f:
                        out.append(Sample(code, kind, d, f.txn, f.xnd, int(actual)))
            d += timedelta(days=1)
    return out


# -- market quotes at the decision hour ---------------------------------------


def _candle_quote(candles: List[dict], at: datetime) -> Tuple[Optional[float], Optional[float]]:
    ts = int(at.timestamp())
    best = None
    for c in candles or []:
        end = c.get("end_period_ts")
        if end is None or end > ts or end < ts - 3 * 3600:
            continue
        if best is None or end > best["end_period_ts"]:
            best = c

    def px(side):
        try:
            v = float(best[side]["close_dollars"])
        except (TypeError, KeyError, ValueError):
            return None
        return v if 0.0 < v < 1.0 else None

    if best is None:
        return None, None
    return px("yes_bid"), px("yes_ask")


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


# -- the blend weight ------------------------------------------------------------


def fit_weight(rows: List[dict]) -> float:
    """Log-loss-optimal trust in the model vs the market on resolved rows."""
    if not rows:
        return 0.0
    pm = np.array([logit(r["p_model"]) for r in rows])
    pk = np.array([logit(r["p_market"]) for r in rows])
    y = np.array([r["outcome"] for r in rows], float)

    def nll(w):
        p = 1.0 / (1.0 + np.exp(-(w * pm + (1 - w) * pk)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = optimize.minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded")
    return float(res.x)


def earned_weight(rows: List[dict], z: float = 2.0) -> Tuple[float, float]:
    """The trust weight the record has *earned*, and the z-score behind it.

    Fit the log-loss-optimal blend weight, then ask whether blending actually
    beat the market by more than noise: the per-row log-loss improvement must
    be ``z`` standard errors above zero. Otherwise the answer is 0: a weight
    fitted on noise is exactly how a backtest talks itself into losing trades.
    """
    if len(rows) < 50:
        return 0.0, 0.0
    w = fit_weight(rows)
    if w <= 0:
        return 0.0, 0.0
    pm = np.array([logit(r["p_model"]) for r in rows])
    pk = np.array([logit(r["p_market"]) for r in rows])
    y = np.array([r["outcome"] for r in rows], float)

    def ll(x):
        p = np.clip(1.0 / (1.0 + np.exp(-x)), 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    gain = ll(pk) - ll(w * pm + (1 - w) * pk)
    # Buckets of one event share an outcome, so treat each event-decision as one draw.
    groups: Dict[tuple, float] = defaultdict(float)
    for r, g in zip(rows, gain):
        groups[(r["ticker"].rsplit("-", 1)[0], r.get("schedule"))] += float(g)
    vals = np.array(list(groups.values()))
    if len(vals) < 20:
        return 0.0, 0.0
    zscore = float(vals.mean() / (vals.std(ddof=1) / math.sqrt(len(vals)))) if vals.std(ddof=1) > 0 else 0.0
    return (w if zscore >= z else 0.0), round(zscore, 2)


def apply_edge_policy(trades: List[dict], min_n: int = 5) -> Tuple[List[dict], Dict[str, str]]:
    """Replay the live Edge Policy over a trade sequence.

    Mirrors ``src.agent.policy``: once a group (the engine as a whole, or one
    series) has ``min_n`` settled trades with negative realized P&L, every later
    trade in it is refused. A trade on day D counts as settled for decisions
    made after D (the CLI report is out the next morning). Returns the trades
    the policy would have allowed and when each block fired.
    """
    allowed: List[dict] = []
    blocked_at: Dict[str, str] = {}
    for t in sorted(trades, key=lambda t: (t["decided_at"], t["ticker"])):
        day = t["decided_at"][:10]
        settled = [a for a in allowed if a["target"] < day]

        def losing(group: List[dict]) -> bool:
            return len(group) >= min_n and sum(a["pnl"] for a in group) < 0

        if "engine" not in blocked_at and losing(settled):
            blocked_at["engine"] = day
        series_key = f"series:{t['series']}"
        if series_key not in blocked_at and losing([a for a in settled if a["series"] == t["series"]]):
            blocked_at[series_key] = day
        if "engine" in blocked_at or series_key in blocked_at:
            continue
        allowed.append(t)
    return allowed, blocked_at


# -- the run ------------------------------------------------------------------------


def _scores(rows: List[dict], key: str) -> Dict[str, float]:
    if not rows:
        return {"brier": float("nan"), "log_loss": float("nan")}
    p = np.clip(np.array([r[key] for r in rows]), 1e-6, 1 - 1e-6)
    y = np.array([r["outcome"] for r in rows], float)
    return {
        "brier": round(float(np.mean((p - y) ** 2)), 5),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 5),
    }


def _trade_stats(trades: List[dict]) -> Dict[str, float]:
    if not trades:
        return {"n": 0}
    pnl = np.array([t["pnl"] for t in trades])
    staked = np.array([t["stake"] for t in trades])
    per = pnl / np.maximum(staked, 1e-9)
    n = len(trades)
    se = float(np.std(per, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {
        "n": n,
        "wins": int(sum(1 for t in trades if t["won"])),
        "win_rate": round(float(np.mean([t["won"] for t in trades])), 4),
        "staked": round(float(staked.sum()), 2),
        "pnl": round(float(pnl.sum()), 2),
        "roi": round(float(pnl.sum() / max(staked.sum(), 1e-9)), 4),
        "mean_return": round(float(per.mean()), 4),
        "t_stat": round(float(per.mean() / se), 2) if se and se > 0 else None,
        "avg_pred_edge": round(float(np.mean([t["ev"] for t in trades])), 4),
        "avg_price": round(float(np.mean([t["price"] for t in trades])), 4),
    }


def _drawdown(trades: List[dict]) -> float:
    eq, peak, dd = 0.0, 0.0, 0.0
    for t in sorted(trades, key=lambda t: (t["decided_at"], t["ticker"])):
        eq += t["pnl"]
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return round(dd, 2)


@dataclass
class BacktestResult:
    config: dict
    model: Dict[str, float]
    market: Dict[str, float]
    blend: Dict[str, float]
    final_weight: float
    weight_z: float
    trades: Dict[str, float]
    by_schedule: Dict[str, Dict[str, float]]
    by_kind: Dict[str, Dict[str, float]]
    by_series: Dict[str, Dict[str, float]]
    max_drawdown: float
    with_policy: Dict[str, float]
    policy_blocks: Dict[str, str]
    n_events: int
    n_rows: int
    calibration: dict
    trade_log: List[dict] = field(default_factory=list)
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    segments: Dict[str, dict] = field(default_factory=dict)
    bucket_rows: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_backtest(fetcher, cfg: BacktestConfig, log=print, kp: Optional[KalshiPublic] = None) -> BacktestResult:
    kp = kp or KalshiPublic(fetcher)
    series = cfg.series or discover_temperature_series(kp)
    log(f"Collecting settled events for {len(series)} series, {cfg.start} .. {cfg.end}")
    events = collect_events(kp, series, cfg.start, cfg.end, log=log)
    codes = {e.code for e in events}
    log(f"{len(events)} usable events across {len(codes)} stations; fetching NBM + CLI history from {cfg.train_start}")
    nbs = collect_nbs(fetcher, codes, cfg.train_start, cfg.end, log=log)
    cli = collect_cli(fetcher, codes, cfg.train_start, cfg.end)
    samples = {s: build_samples(nbs, cli, s, cfg.train_start, cfg.end) for s in cfg.schedules}
    fees = {}

    ecfg = EvalConfig(bankroll=cfg.bankroll, kelly=cfg.kelly, min_edge=cfg.min_edge, include_maker=False)
    rows: List[dict] = []
    trades: List[dict] = []
    calib_cache: Dict[Tuple[str, date], WeatherCalibration] = {}

    # Decide in time order so the weight fit only ever sees the past.
    jobs = sorted(
        ((decision_time(e.target, s), s, e) for e in events for s in cfg.schedules
         if not (s == "same_day" and e.kind == "low")),
        key=lambda j: j[0],
    )
    candle_cache: Dict[str, Dict[str, List[dict]]] = {}
    weight = cfg.fixed_weight if cfg.fixed_weight is not None else 0.0
    weight_day: Optional[date] = None
    log(f"Replaying {len(jobs)} decisions")
    for i, (at, sched, se) in enumerate(jobs):
        if i and i % 250 == 0:
            log(f"  {i}/{len(jobs)} decisions, {len(trades)} trades so far")
        known_through = at.date() - timedelta(days=1)

        # Trust weight: refit once per decision day on outcomes known by then.
        if cfg.fixed_weight is None and weight_day != at.date():
            weight_day = at.date()
            past = [r for r in rows if r["target"] <= known_through.isoformat()]
            weight = earned_weight(past)[0] if len(past) >= cfg.min_blend_rows else 0.0

        key = (sched, known_through)
        if key not in calib_cache:
            train = [s for s in samples[sched] if s.target <= known_through]
            calib_cache[key] = WeatherCalibration.fit(train, min_samples=cfg.min_calib_samples)
        cal = calib_cache[key]
        if cal.n.get(se.kind, 0) < cfg.min_calib_samples:
            continue
        f = nbs[se.code].latest(se.target, se.kind, at) if se.code in nbs else None
        if f is None:
            continue
        dist = TempDistribution.from_forecast(f.txn, f.xnd, cal.params(se.code, se.kind))

        ev = se.event
        if ev.event_ticker not in candle_cache:
            s0 = decision_time(se.target, "day_ahead") - timedelta(hours=4)
            s1 = decision_time(se.target, "same_day") + timedelta(hours=1)
            try:
                candle_cache[ev.event_ticker] = kp.event_candles(
                    ev.series_ticker, ev.event_ticker, int(s0.timestamp()), int(s1.timestamp()))
            except Exception as exc:  # one bad event shouldn't sink the run
                log(f"  candles {ev.event_ticker}: {exc}")
                candle_cache[ev.event_ticker] = {}
        candles = candle_cache[ev.event_ticker]

        quotes = []
        for m in ev.markets:
            yb, ya = _candle_quote(candles.get(m.ticker, []), at)
            q = Quote(ticker=m.ticker, event_ticker=ev.event_ticker, subtitle=m.subtitle,
                      strike_type=m.strike_type, floor_strike=m.floor_strike, cap_strike=m.cap_strike,
                      yes_bid=yb, yes_ask=ya, yes_bid_size=cfg.depth, yes_ask_size=cfg.depth,
                      result=m.result, close_time=at + timedelta(hours=24))
            quotes.append(q)
        if not quotes or all(q.yes_bid is None and q.yes_ask is None for q in quotes):
            continue
        implied = devig(quotes)
        if ev.series_ticker not in fees:
            fees[ev.series_ticker] = FeeSchedule.from_series(kp.series(ev.series_ticker))
        fee = fees[ev.series_ticker]

        opps = []
        for q in quotes:
            p = dist.prob(q)
            if p is None or q.result not in ("yes", "no"):
                continue
            outcome = 1 if q.result == "yes" else 0
            p_mkt = implied.get(q.ticker)
            if p_mkt is None:
                continue
            rows.append({"target": se.target.isoformat(), "schedule": sched, "kind": se.kind,
                         "ticker": q.ticker, "p_model": p, "p_market": p_mkt,
                         "p_blend": sigmoid(weight * logit(p) + (1 - weight) * logit(p_mkt)),
                         "outcome": outcome})
            fair = Fair(q.ticker, p_yes=p, weight=weight, p_market_yes=p_mkt)
            opps.extend(evaluate("weather", q, fair, ecfg, fee))
        for o in select(opps, ecfg):
            q = next(q for q in quotes if q.ticker == o.ticker)
            won = (q.result == "yes") == (o.side == "yes")
            pnl = o.contracts * ((1.0 if won else 0.0) - o.price - o.fee)
            trades.append({"decided_at": at.isoformat(), "target": se.target.isoformat(),
                           "schedule": sched, "kind": se.kind, "series": ev.series_ticker,
                           "ticker": o.ticker, "side": o.side, "price": o.price, "fee": o.fee,
                           "contracts": o.contracts, "stake": o.stake, "p_fair": o.p_fair,
                           "p_model": o.p_engine, "p_market": o.p_market, "ev": o.ev,
                           "weight": round(weight, 3), "won": won, "pnl": round(pnl, 2)})

    def group(key):
        g = defaultdict(list)
        for t in trades:
            g[t[key]].append(t)
        return {k: _trade_stats(v) for k, v in sorted(g.items())}

    daily = defaultdict(float)
    for t in trades:
        daily[t["target"]] += t["pnl"]
    scored_all = [r for r in rows if r["p_market"] is not None]
    seg_groups = defaultdict(list)
    for r in scored_all:
        seg_groups[f"{r['schedule']}/{r['kind']}"].append(r)
        seg_groups[f"series/{r['ticker'].split('-')[0]}"].append(r)
    segments = {
        k: {"n": len(v), "model": _scores(v, "p_model"), "market": _scores(v, "p_market"),
            "weight": round(fit_weight(v), 3) if len(v) >= 50 else None,
            "earned": earned_weight(v)[0] > 0}
        for k, v in sorted(seg_groups.items())
    }
    final_cal = WeatherCalibration.fit(samples.get("day_ahead") or next(iter(samples.values()), []))
    scored = [r for r in rows if r["p_market"] is not None]
    final_w, final_z = earned_weight(scored)
    allowed, blocks = apply_edge_policy(trades)
    return BacktestResult(
        config={k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(cfg).items()},
        model=_scores(scored, "p_model"),
        market=_scores(scored, "p_market"),
        blend=_scores(scored, "p_blend"),
        final_weight=round(final_w, 3),
        weight_z=final_z,
        trades=_trade_stats(trades),
        by_schedule=group("schedule"),
        by_kind=group("kind"),
        by_series=group("series"),
        max_drawdown=_drawdown(trades),
        with_policy={**_trade_stats(allowed), "max_drawdown": _drawdown(allowed)},
        policy_blocks=blocks,
        n_events=len(events),
        n_rows=len(scored),
        calibration=final_cal.to_dict(),
        trade_log=trades,
        daily_pnl={k: round(v, 2) for k, v in sorted(daily.items())},
        segments=segments,
        bucket_rows=scored_all,
    )


def render_markdown(r: BacktestResult) -> str:
    c, t = r.config, r.trades
    lines = [
        "# Weather engine: walk-forward backtest",
        "",
        f"Traded window **{c['start']} .. {c['end']}** (calibration history from {c['train_start']}), "
        f"{r.n_events} settled events, {r.n_rows} bucket decisions scored.",
        "",
        "## Forecast quality (out-of-sample, every bucket at every decision)",
        "",
        "| Source | Brier | Log loss |",
        "|---|---|---|",
        f"| Kalshi market (de-vigged mid) | {r.market['brier']} | {r.market['log_loss']} |",
        f"| Engine (calibrated NBM) | {r.model['brier']} | {r.model['log_loss']} |",
        f"| Blend at walk-forward weight | {r.blend['brier']} | {r.blend['log_loss']} |",
        "",
        f"Trust the engine has *earned* over the whole window: **w = {r.final_weight}** "
        f"(blend's log-loss gain over the market: z = {r.weight_z}; a weight is only granted at z >= 2, "
        "otherwise the market already knows what the engine knows and w = 0).",
        "",
        "### By segment",
        "",
        "Per-series rows are many small comparisons: expect a few to beat the market by luck alone.",
        "",
        "| Segment | Buckets | Market Brier | Engine Brier | Fitted w | Earned (z >= 2)? |",
        "|---|---|---|---|---|---|",
        *[f"| {k} | {v['n']} | {v['market']['brier']} | {v['model']['brier']} | {v['weight']} | "
          f"{'yes' if v.get('earned') else 'no'} |" for k, v in r.segments.items()],
        "",
        "## Simulated trading (taker at the candle ask, exact fees, flat bankroll)",
        "",
    ]
    if not t.get("n"):
        lines.append("No trades cleared the bar. That is a valid result: no measured edge, no bets.")
    else:
        lines += [
            "| Trades | Win rate | Staked | P&L | ROI | Mean return / trade | t-stat | Max drawdown |",
            "|---|---|---|---|---|---|---|---|",
            f"| {t['n']} | {t['win_rate']:.1%} | ${t['staked']:,.2f} | ${t['pnl']:,.2f} | {t['roi']:.1%} | "
            f"{t['mean_return']:.1%} | {t['t_stat']} | ${r.max_drawdown:,.2f} |",
            "",
        ]
        wp = r.with_policy
        engine_block = r.policy_blocks.get("engine")
        lines += [
            "### With the Edge Policy in the loop",
            "",
            "The live system refuses any group with 5+ settled trades and negative P&L "
            "(`src/agent/policy.py`). Replayed over the same trades:",
            "",
            f"- Engine blocked on **{engine_block}**" if engine_block else "- The engine as a whole was never blocked.",
            f"- {sum(1 for k in r.policy_blocks if k.startswith('series:'))} series blocked individually.",
            f"- Trades allowed: **{wp.get('n', 0)}**, P&L **${wp.get('pnl', 0):,.2f}**, "
            f"max drawdown ${wp.get('max_drawdown', 0):,.2f}.",
            "",
        ]
        for title, groups in (("schedule", r.by_schedule), ("kind", r.by_kind), ("series", r.by_series)):
            lines += [f"### By {title}", "", "| | Trades | Win rate | P&L | ROI | t-stat |", "|---|---|---|---|---|---|"]
            for k, g in groups.items():
                lines.append(f"| {k} | {g['n']} | {g['win_rate']:.1%} | ${g['pnl']:,.2f} | {g['roi']:.1%} | {g['t_stat']} |")
            lines.append("")
    lines += [
        "## Caveats",
        "",
        f"- Fills assume {c['depth']:.0f} contracts at the hourly candle's closing ask; real depth and slippage vary.",
        "- No resting-order (maker) fills are simulated, so maker edge is not counted.",
        "- The traded window is limited to markets Kalshi still serves candles for (settled after its historical cutoff).",
        "- A positive backtest is evidence, not proof. Paper-trade forward (`cli.py engines paper`) before risking money.",
        "",
    ]
    return "\n".join(lines)


def save_result(r: BacktestResult, out_dir: Path | str = DEFAULT_OUT) -> Tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    j = out / f"weather_{stamp}.json"
    m = out / f"weather_{stamp}.md"
    j.write_text(json.dumps(r.to_dict(), indent=1, default=str))
    m.write_text(render_markdown(r))
    return j, m
