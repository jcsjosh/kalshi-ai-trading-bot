"""``cli.py engines ...``: scan, run, paper-trade, promote, backtest and calibrate the edge engines."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List

from src.engines import ENGINE_NAMES

REPORT_DIR = Path("data/engines/reports")


def _eval_config(args: argparse.Namespace, bankroll: float):
    from src.engines.evaluate import EvalConfig

    return EvalConfig(bankroll=bankroll, kelly=args.kelly, min_edge=args.min_edge,
                      max_bet_fraction=args.max_bet, include_maker=not args.taker_only)


def _engines(args: argparse.Namespace) -> List[str]:
    names = [e.strip() for e in (args.engines or ",".join(ENGINE_NAMES)).split(",") if e.strip()]
    bad = [n for n in names if n not in ENGINE_NAMES]
    if bad:
        sys.exit(f"unknown engine(s): {', '.join(bad)} (choose from {', '.join(ENGINE_NAMES)})")
    return names


def _log(args: argparse.Namespace) -> Callable[[str], None]:
    return (lambda m: print(m, file=sys.stderr)) if args.verbose else (lambda *_: None)


def _print_orders(title: str, orders) -> None:
    print(title)
    print(f"  {'engine':8} {'action':24} {'ticker':32} {'n':>5} {'p_eng':>6} {'p_mkt':>6} {'ev':>6} {'E[$]':>7}")
    for o in orders:
        pm = f"{o.p_market:.3f}" if o.p_market is not None else "    -"
        print(f"  {o.engine:8} {o.action:24} {o.ticker:32} {o.contracts:5d} {o.p_engine:6.3f} {pm:>6} "
              f"{o.ev:6.3f} {o.expected_profit:7.2f}")
        print(f"  {'':8} {o.rationale[:110]}")


def _print_report(report, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return
    for name, info in report.priced.items():
        extra = f", trust weight {info['weight']:.2f}" if "weight" in info else ""
        print(f"{name}: priced {info.get('markets', 0)} markets in {info.get('events', 0)} events{extra}")
    for n in report.notes:
        print(f"note: {n}")
    print()
    if report.selected:
        _print_orders("Tradeable at earned trust:", report.selected)
    else:
        print("No engine opportunity cleared the bar at earned trust. A no-trade scan is a valid result.")
    if report.shadow:
        _print_orders("\nShadow (each model fully trusted; paper only, never sent):", report.shadow)
    if report.baskets:
        print("\nArbitrage baskets (guaranteed payout > fee-inclusive cost, walked off live depth):")
        for b in report.baskets:
            tag = "RISK-FREE" if b.risk_free else "CHECK RULES"
            legs = " + ".join(f"{l.side.upper()} {l.ticker}@{l.price:.2f}" for l in b.legs)
            print(f"  [{tag}] {b.kind} {b.event_ticker}: {b.count} x ${b.profit:.3f} = ${b.total_profit:.2f}  ({legs})")
            print(f"    {b.note}")


def _save_report(report) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"scan_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(report.to_dict(detail=True), indent=1, default=str))
    return path


def cmd_scan(args: argparse.Namespace) -> None:
    from src.engines.http import Fetcher
    from src.engines.runner import scan

    with Fetcher() as f:
        report = scan(f, _engines(args), _eval_config(args, args.bankroll), args.max_orders,
                      series=args.series, log=_log(args))
    _print_report(report, args.json)
    if not args.json:
        print(f"\nfull report: {_save_report(report)}")


def cmd_run(args: argparse.Namespace) -> None:
    """Scan, then paper-record (default) or place guarded live orders (--live)."""
    from src.engines import ledger
    from src.engines.http import Fetcher
    from src.engines.runner import basket_opportunities, execute_basket_live, execute_live, live_plan, scan

    def _scan(bankroll: float):
        with Fetcher() as f:
            return scan(f, _engines(args), _eval_config(args, bankroll), args.max_orders,
                        series=args.series, log=_log(args))

    if not args.live:
        report = _scan(args.bankroll)
        _print_report(report, False)
        baskets, orders = live_plan(report)
        recs = [ledger.record_from_opportunity(o) for o in orders + report.shadow]
        for b in baskets:
            tag = f"{b.event_ticker}:{b.kind}:{datetime.now(timezone.utc):%Y%m%dT%H%M}"
            recs += [ledger.record_from_opportunity(o, basket=tag) for o in basket_opportunities(b)]
        ledger.append(recs)
        print(f"\nPAPER: recorded {len(recs)} orders ({len(report.shadow)} shadow) to "
              f"{ledger.DEFAULT_LEDGER_PATH}. Nothing was sent to Kalshi. "
              f"Score later with `cli.py engines paper --settle`.")
        return

    async def _live():
        from src.clients.kalshi_client import KalshiClient

        client = KalshiClient()
        try:
            bal = await client.get_balance()
            bankroll = (int(bal.get("balance", 0) or 0) + int(bal.get("portfolio_value", 0) or 0)) / 100.0
            print(f"LIVE: sizing from account equity ${bankroll:,.2f}")
            report = await asyncio.to_thread(_scan, bankroll)  # public data; keeps one event loop
            _print_report(report, False)
            baskets, orders = live_plan(report)  # shadow orders are never sent
            out = [await execute_basket_live(client, b) for b in baskets]
            out += await execute_live(client, orders)
            return out
        finally:
            await client.close()

    print(json.dumps(asyncio.run(_live()), indent=2, default=str))


def cmd_paper(args: argparse.Namespace) -> None:
    from src.engines import ledger
    from src.engines.http import Fetcher
    from src.engines.market import KalshiPublic

    records = ledger.load()
    if not records:
        print("Paper ledger is empty. Record some with `cli.py engines run` (dry by default).")
        return
    if args.settle:
        with Fetcher() as f:
            n = ledger.settle(records, KalshiPublic(f))
        ledger.save(records)
        print(f"settled {n} paper orders")
    summary = ledger.summarize(records)
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    print(f"{'engine':20} {'orders':>6} {'open':>5} {'filled':>6} {'wins':>5} {'staked':>9} {'P&L':>9} "
          f"{'ROI':>7} {'E[P&L]':>8} {'LL eng':>7} {'LL mkt':>7}")
    for name, s in summary.items():
        roi = f"{s['roi']:.1%}" if s["roi"] is not None else "-"
        lle = f"{s['log_loss_engine']:.3f}" if s["log_loss_engine"] is not None else "-"
        llm = f"{s['log_loss_market']:.3f}" if s["log_loss_market"] is not None else "-"
        print(f"{name:20} {s['orders']:6d} {s['open']:5d} {s['filled']:6d} {s['wins']:5d} {s['staked']:9.2f} "
              f"{s['pnl']:9.2f} {roi:>7} {s['expected_pnl']:8.2f} {lle:>7} {llm:>7}")
    print("\nLL = log loss of the engine's own probability vs the market's, for the side bought (lower is\n"
          "better; observation-certain orders excluded). `cli.py engines promote <engine>` turns a\n"
          "shadow record into trust only if LL eng beats LL mkt significantly AND shadow P&L is positive.")


def cmd_promote(args: argparse.Namespace) -> None:
    from src.engines import ledger
    from src.engines.trust import promotion_from_paper, save_weight

    verdict = promotion_from_paper(ledger.load(), args.engine)
    print(json.dumps(verdict, indent=2))
    if args.save:
        save_weight(args.engine, verdict["weight"],
                    f"shadow paper record: {verdict['settled']} settled, P&L ${verdict['pnl']}, "
                    f"z={verdict['z']} ({verdict['verdict']})")
        print(f"trust weight for {args.engine} set to {verdict['weight']}")


def cmd_daily(args: argparse.Namespace) -> None:
    import os

    from src.engines.daily import render, run_daily

    result = run_daily(live=args.live, live_cap=args.live_cap,
                       state_repo=args.state_repo or os.environ.get("BOT_STATE_REPO"),
                       log=lambda m: print(m, file=sys.stderr, flush=True))
    print(render(result))


def cmd_backtest(args: argparse.Namespace) -> None:
    from src.engines.http import Fetcher
    from src.engines.trust import save_weight
    from src.engines.weather.backtest import BacktestConfig, render_markdown, run_backtest, save_result
    from src.engines.weather.calibration import DEFAULT_CALIBRATION_PATH, WeatherCalibration

    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=58)
    train = date.fromisoformat(args.train_start) if args.train_start else start - timedelta(days=120)
    cfg = BacktestConfig(start=start, end=end, train_start=train, series=args.series,
                         bankroll=args.bankroll, min_edge=args.min_edge, kelly=args.kelly)
    with Fetcher() as f:
        result = run_backtest(f, cfg, log=lambda m: print(m, file=sys.stderr, flush=True))
    j, m = save_result(result)
    print(render_markdown(result))
    print(f"saved: {m}\n       {j}")
    if args.save_weight:
        cal = WeatherCalibration.from_dict(result.calibration)
        losing = result.trades.get("n", 0) and result.trades.get("pnl", 0) <= 0
        cal.weight = 0.0 if losing else result.final_weight
        cal.weight_source = (
            f"walk-forward backtest {start}..{end}, {result.n_rows} bucket decisions, earned w={result.final_weight} "
            f"(z={result.weight_z}), simulated P&L ${result.trades.get('pnl', 0)}"
            + ("; zeroed because simulated trading lost money" if losing else "") + f" ({m.name})")
        cal.save(DEFAULT_CALIBRATION_PATH)
        save_weight("weather", cal.weight, cal.weight_source)
        print(f"calibration + trust weight w={cal.weight} written to {DEFAULT_CALIBRATION_PATH}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    from src.engines.http import Fetcher
    from src.engines.weather.backtest import build_samples, collect_cli, collect_nbs
    from src.engines.weather.calibration import DEFAULT_CALIBRATION_PATH, WeatherCalibration
    from src.engines.weather.stations import STATIONS

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days)
    codes = set(args.stations.split(",")) if args.stations else set(STATIONS)
    with Fetcher() as f:
        nbs = collect_nbs(f, codes, start, end, log=lambda *_: None)
        cli = collect_cli(f, codes, start, end)
    samples = build_samples(nbs, cli, "day_ahead", start, end)
    old = WeatherCalibration.load(DEFAULT_CALIBRATION_PATH)
    cal = WeatherCalibration.fit(samples)
    # Calibration fits the model; trust is only earned by backtests and paper records.
    cal.weight, cal.weight_source = old.weight, old.weight_source
    cal.save(DEFAULT_CALIBRATION_PATH)
    print(json.dumps(cal.to_dict(), indent=1))
    print(f"\nfit on {len(samples)} forecast/outcome pairs ({start}..{end}); saved {DEFAULT_CALIBRATION_PATH}")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "engines",
        help="Model-driven edge engines: scan, run, paper, promote, backtest, calibrate",
        description=(
            "Edge engines price markets from their own models (weather: NOAA NBM calibrated to the "
            "settlement stations; games: DraftKings + Polymarket consensus; arbitrage: probability-axiom "
            "violations after fees), shrink toward the market by the trust each has earned, and trade only "
            "through the guarded, journaled order path."
        ),
    )
    sub = p.add_subparsers(dest="engines_command", required=True)

    def scan_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--engines", default=None, help=f"comma list (default: {','.join(ENGINE_NAMES)})")
        sp.add_argument("--series", nargs="*", default=None, help="limit to these series tickers")
        sp.add_argument("--bankroll", type=float, default=1000.0, help="sizing bankroll in $ (live: account equity)")
        sp.add_argument("--kelly", type=float, default=0.25, help="fraction of full Kelly (default 0.25)")
        sp.add_argument("--min-edge", dest="min_edge", type=float, default=0.03,
                        help="required EV per contract after fees, $ (default 0.03)")
        sp.add_argument("--max-bet", dest="max_bet", type=float, default=0.03,
                        help="max fraction of bankroll per market (default 0.03)")
        sp.add_argument("--max-orders", dest="max_orders", type=int, default=20)
        sp.add_argument("--taker-only", dest="taker_only", action="store_true", help="no resting maker entries")
        sp.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("scan", help="Price markets and list opportunities (read-only)")
    scan_flags(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_scan)

    r = sub.add_parser("run", help="Scan then paper-record (default) or place guarded orders (--live)")
    scan_flags(r)
    r.add_argument("--live", action="store_true",
                   help="Place REAL orders through the risk governor + Edge Policy (default: paper ledger)")
    r.set_defaults(func=cmd_run)

    pp = sub.add_parser("paper", help="Paper ledger: settle and score forward-tested engine orders")
    pp.add_argument("--settle", action="store_true", help="attach Kalshi settlements to open paper orders")
    pp.add_argument("--json", action="store_true")
    pp.set_defaults(func=cmd_paper)

    pr = sub.add_parser("promote", help="Earn (or lose) an engine's trust from its settled shadow paper record")
    pr.add_argument("engine", choices=[e for e in ENGINE_NAMES if e != "arbitrage"])
    pr.add_argument("--save", action="store_true", help="write the earned weight (default: report only)")
    pr.set_defaults(func=cmd_promote)

    d = sub.add_parser("daily", help="One unattended tick: restore record, learn, scan, trade what's earned, save")
    d.add_argument("--live", action="store_true",
                   help="Allow REAL orders, only for engines with earned trust and risk-free baskets")
    d.add_argument("--live-cap", dest="live_cap", type=float, default=500.0,
                   help="Max bankroll the bot sizes from, $ (default 500)")
    d.add_argument("--state-repo", dest="state_repo", default=None,
                   help="private owner/repo holding the bot's record (default: $BOT_STATE_REPO)")
    d.set_defaults(func=cmd_daily)

    b = sub.add_parser("backtest", help="Walk-forward weather backtest vs settled Kalshi markets")
    b.add_argument("--start", help="first traded target date (default: 59 days ago)")
    b.add_argument("--end", help="last traded target date (default: yesterday)")
    b.add_argument("--train-start", dest="train_start", help="calibration history start (default: start-120d)")
    b.add_argument("--series", nargs="*", default=None)
    b.add_argument("--bankroll", type=float, default=1000.0)
    b.add_argument("--min-edge", dest="min_edge", type=float, default=0.03)
    b.add_argument("--kelly", type=float, default=0.25)
    b.add_argument("--save-weight", dest="save_weight", action="store_true",
                   help="persist the measured trust weight + calibration for the live engine")
    b.set_defaults(func=cmd_backtest)

    c = sub.add_parser("calibrate", help="Refit weather calibration on recent NBM forecasts vs CLI reports")
    c.add_argument("--days", type=int, default=180)
    c.add_argument("--stations", default=None, help="comma list of CLI codes (default: all)")
    c.set_defaults(func=cmd_calibrate)
