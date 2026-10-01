"""``cli.py engines ...``: scan, backtest, calibrate, paper-trade and run the edge engines."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.engines import ENGINE_NAMES

REPORT_DIR = Path("data/engines/reports")


def _eval_config(args, bankroll: float):
    from src.engines.evaluate import EvalConfig

    return EvalConfig(
        bankroll=bankroll,
        kelly=args.kelly,
        min_edge=args.min_edge,
        max_bet_fraction=args.max_bet,
        include_maker=not args.taker_only,
    )


def _engines(args):
    names = [e.strip() for e in (args.engines or ",".join(ENGINE_NAMES)).split(",") if e.strip()]
    bad = [n for n in names if n not in ENGINE_NAMES]
    if bad:
        sys.exit(f"unknown engine(s): {', '.join(bad)} (choose from {', '.join(ENGINE_NAMES)})")
    return names


def _print_report(report, as_json: bool) -> None:
    if as_json:
        print(json.dumps({
            "selected": [o.to_dict() for o in report.selected],
            "opportunities": len(report.opportunities),
            "baskets": [b.to_dict() for b in report.baskets],
            "priced": {k: {kk: vv for kk, vv in v.items() if kk != "events_detail"} for k, v in report.priced.items()},
            "notes": report.notes,
        }, indent=2, default=str))
        return
    for name, info in report.priced.items():
        extra = ""
        if name == "weather":
            extra = f", trust weight {info.get('weight') or 0:.2f}, calibrated through {info.get('calibrated_through') or 'never'}"
        print(f"{name}: priced {info.get('markets', 0)} markets in {info.get('events', 0)} events{extra}")
    for n in report.notes:
        print(f"note: {n}")
    print()
    if report.selected:
        print(f"{'engine':9} {'action':24} {'ticker':30} {'n':>5} {'p_fair':>7} {'p_mkt':>7} {'ev':>7} {'E[$]':>8}")
        for o in report.selected:
            pm = f"{o.p_market:.3f}" if o.p_market is not None else "   -"
            print(f"{o.engine:9} {o.action:24} {o.ticker:30} {o.contracts:5d} {o.p_fair:7.3f} {pm:>7} "
                  f"{o.ev:7.3f} {o.expected_profit:8.2f}")
            print(f"{'':9} {o.rationale[:110]}")
    else:
        print("No engine opportunity cleared the bar. A no-trade scan is a valid result.")
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
    path.write_text(json.dumps({
        "selected": [o.to_dict() for o in report.selected],
        "opportunities": [o.to_dict() for o in report.opportunities],
        "baskets": [b.to_dict() for b in report.baskets],
        "priced": report.priced,
        "notes": report.notes,
    }, indent=1, default=str))
    return path


def cmd_scan(args) -> None:
    from src.engines.http import Fetcher
    from src.engines.runner import scan

    log = (lambda m: print(m, file=sys.stderr)) if args.verbose else (lambda *_: None)
    with Fetcher() as f:
        report = scan(f, _engines(args), _eval_config(args, args.bankroll), args.max_orders,
                      series=args.series, log=log)
    _print_report(report, args.json)
    if not args.json:
        print(f"\nfull report: {_save_report(report)}")


def cmd_run(args) -> None:
    """Scan, then paper-record (default) or place guarded live orders (--live)."""
    from src.engines import ledger
    from src.engines.http import Fetcher
    from src.engines.runner import basket_opportunities, execute_basket_live, execute_live, scan

    log = (lambda m: print(m, file=sys.stderr)) if args.verbose else (lambda *_: None)

    def _scan(bankroll: float):
        with Fetcher() as f:
            return scan(f, _engines(args), _eval_config(args, bankroll), args.max_orders,
                        series=args.series, log=log)

    if not args.live:
        report = _scan(args.bankroll)
        _print_report(report, False)
        recs = [ledger.record_from_opportunity(o) for o in report.selected]
        for b in (b for b in report.baskets if b.risk_free):
            tag = f"{b.event_ticker}:{b.kind}:{datetime.now(timezone.utc):%Y%m%dT%H%M}"
            recs += [ledger.record_from_opportunity(o, basket=tag) for o in basket_opportunities(b)]
        ledger.append(recs)
        print(f"\nPAPER: recorded {len(recs)} orders to {ledger.DEFAULT_LEDGER_PATH} "
              f"(settle + score later with `cli.py engines paper --settle`). Nothing was sent to Kalshi.")
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
            out = await execute_live(client, report.selected)
            for b in (b for b in report.baskets if b.risk_free):
                out.append(await execute_basket_live(client, b))
            return out
        finally:
            await client.close()

    print(json.dumps(asyncio.run(_live()), indent=2, default=str))


def cmd_paper(args) -> None:
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
    print(f"{'engine':10} {'orders':>6} {'open':>5} {'filled':>6} {'wins':>5} {'staked':>9} {'P&L':>9} {'ROI':>7} "
          f"{'E[P&L]':>8} {'LL eng':>7} {'LL mkt':>7}")
    for name, s in summary.items():
        roi = f"{s['roi']:.1%}" if s["roi"] is not None else "-"
        lle = f"{s['log_loss_engine']:.3f}" if s["log_loss_engine"] is not None else "-"
        llm = f"{s['log_loss_market']:.3f}" if s["log_loss_market"] is not None else "-"
        print(f"{name:10} {s['orders']:6d} {s['open']:5d} {s['filled']:6d} {s['wins']:5d} {s['staked']:9.2f} "
              f"{s['pnl']:9.2f} {roi:>7} {s['expected_pnl']:8.2f} {lle:>7} {llm:>7}")
    print("\nLL = log loss of the probability for the side bought (lower is better). An engine with real\n"
          "information shows LL eng < LL mkt over many settled orders; P&L alone is noisy.")


def cmd_backtest(args) -> None:
    from src.engines.http import Fetcher
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
        print(f"calibration + trust weight w={cal.weight} written to {DEFAULT_CALIBRATION_PATH}")


def cmd_calibrate(args) -> None:
    from src.engines.http import Fetcher
    from src.engines.market import KalshiPublic
    from src.engines.weather.backtest import build_samples, collect_cli, collect_nbs
    from src.engines.weather.calibration import DEFAULT_CALIBRATION_PATH, WeatherCalibration
    from src.engines.weather.stations import STATIONS

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days)
    codes = set(args.stations.split(",")) if args.stations else set(STATIONS)
    quiet = lambda *_: None  # noqa: E731
    with Fetcher() as f:
        nbs = collect_nbs(f, codes, start, end, log=quiet)
        cli = collect_cli(f, codes, start, end)
    samples = build_samples(nbs, cli, "day_ahead", start, end)
    old = WeatherCalibration.load(DEFAULT_CALIBRATION_PATH)
    cal = WeatherCalibration.fit(samples)
    cal.weight, cal.weight_source = old.weight, old.weight_source  # trust is earned in backtests, not here
    if args.weight is not None:
        cal.weight, cal.weight_source = args.weight, "set manually"
    cal.save(DEFAULT_CALIBRATION_PATH)
    print(json.dumps(cal.to_dict(), indent=1))
    print(f"\nfit on {len(samples)} forecast/outcome pairs ({start}..{end}); saved {DEFAULT_CALIBRATION_PATH}")


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "engines",
        help="Model-driven edge engines: scan, backtest, calibrate, paper-trade, run",
        description=(
            "Edge engines price markets from their own models (weather: NOAA NBM calibrated to the "
            "settlement stations; arbitrage: probability-axiom violations after fees), shrink toward "
            "the market by measured skill, and trade only through the guarded, journaled order path."
        ),
    )
    sub = p.add_subparsers(dest="engines_command", required=True)

    def common(sp, live=False):
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
    common(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_scan)

    r = sub.add_parser("run", help="Scan then paper-record (default) or place guarded orders (--live)")
    common(r)
    r.add_argument("--live", action="store_true",
                   help="Place REAL orders through the risk governor + Edge Policy (default: paper ledger)")
    r.set_defaults(func=cmd_run)

    pp = sub.add_parser("paper", help="Paper ledger: settle and score forward-tested engine orders")
    pp.add_argument("--settle", action="store_true", help="attach Kalshi settlements to open paper orders")
    pp.add_argument("--json", action="store_true")
    pp.set_defaults(func=cmd_paper)

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
    c.add_argument("--weight", type=float, default=None, help="override the trust weight (default: keep measured)")
    c.set_defaults(func=cmd_calibrate)
