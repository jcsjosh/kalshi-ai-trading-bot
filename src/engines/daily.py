"""One unattended daily tick: restore state, learn, scan, trade what's earned, save.

Built for a scheduled cloud session (or cron), where each run starts from a fresh
checkout. Order matters:

1. **Restore** the bot's record from a *private* state repo (``BOT_STATE_REPO``,
   e.g. ``you/kalshi-bot-state``). The trading repo may be public; the record
   never goes there.
2. **Learn**: settle the paper ledger, and let each model engine's settled shadow
   record earn (or lose) trust. With live credentials, also run the existing
   ``fills`` / ``learnings`` / ``improve`` loop on real trades.
3. **Scan** every engine and paper-record everything, shadow included.
4. **Trade live** only (a) engines whose trust is above 0 and (b) risk-free
   arbitrage baskets, sized from ``min(account equity, live cap)``, through the
   risk governor and Edge Policy. Nothing else is ever sent.
5. **Report** to ``data/engines/reports/daily_<date>.md`` and stdout.
6. **Save** the record back to the state repo.

Kill switch from anywhere (a phone included): create ``runtime/TRADING_HALTED``
in the state repo. The next run restores it and the governor halts all buying.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

STATE_DIRS = {"engines": Path("data/engines"), "runtime": Path("data/runtime")}
SKIP_IN_STATE = ("reports/scan_", "backtest/")  # large, regenerable
CHECKOUT = Path(".bot-state")

Log = Callable[[str], None]


def _git(*args: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=180)


@dataclass
class StateStore:
    """The bot's record in a private git repo, mirrored into ``data/``."""

    repo: Optional[str]
    log: Log = print
    ok: bool = False
    note: str = ""

    def restore(self) -> None:
        if not self.repo:
            self.note = "BOT_STATE_REPO not set: this run's record will not persist"
            return
        shutil.rmtree(CHECKOUT, ignore_errors=True)
        res = _git("clone", "--depth", "1", f"https://github.com/{self.repo}.git", str(CHECKOUT))
        if res.returncode != 0:
            self.note = f"could not clone {self.repo} ({res.stderr.strip()[:200]}): record will not persist"
            return
        for name, dest in STATE_DIRS.items():
            src = CHECKOUT / name
            if src.exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest, dirs_exist_ok=True)
        self.ok = True
        self.log(f"state: restored from {self.repo}")

    def save(self, message: str) -> None:
        if not self.ok:
            return
        for name, src in STATE_DIRS.items():
            if not src.exists():
                continue
            for f in src.rglob("*"):
                rel = f.relative_to(src)
                if f.is_dir() or any(str(rel).startswith(s) for s in SKIP_IN_STATE):
                    continue
                out = CHECKOUT / name / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)
        _git("add", "-A", cwd=CHECKOUT)
        if _git("diff", "--cached", "--quiet", cwd=CHECKOUT).returncode == 0:
            self.log("state: nothing changed")
            return
        _git("-c", "user.name=kalshi-bot", "-c", "user.email=kalshi-bot@users.noreply.github.com",
             "commit", "-m", message, cwd=CHECKOUT)
        for _ in range(4):
            if _git("push", "origin", "HEAD", cwd=CHECKOUT).returncode == 0:
                self.log(f"state: saved to {self.repo}")
                return
            _git("pull", "--rebase", "origin", "HEAD", cwd=CHECKOUT)
        self.note = f"push to {self.repo} failed: this run's record was not saved"
        self.log(f"state: {self.note}")


def live_credentials() -> bool:
    return bool(os.environ.get("KALSHI_API_KEY")) and bool(
        os.environ.get("KALSHI_PRIVATE_KEY") or Path(os.environ.get("KALSHI_PRIVATE_KEY_PATH",
                                                                     "kalshi_private_key.pem")).exists())


def _cli(*args: str, log: Log) -> str:
    res = subprocess.run([sys.executable, "cli.py", *args], capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        log(f"cli {' '.join(args)} failed: {res.stderr.strip()[-300:]}")
    return res.stdout


@dataclass
class DailyResult:
    date: str
    state: str
    promotions: Dict[str, dict] = field(default_factory=dict)
    paper_summary: Dict[str, dict] = field(default_factory=dict)
    recorded: int = 0
    shadow: int = 0
    live: List[dict] = field(default_factory=list)
    live_note: str = ""
    notes: List[str] = field(default_factory=list)


def render(r: DailyResult) -> str:
    lines = [f"# Kalshi bot daily report: {r.date}", "", f"State: {r.state}", ""]
    lines += ["## Trust (earned from settled shadow records)", "", "| Engine | Settled | P&L | z | Weight | Verdict |",
              "|---|---|---|---|---|---|"]
    for name, v in r.promotions.items():
        lines.append(f"| {name} | {v['settled']} | ${v['pnl']} | {v['z']} | {v['weight']} | {v['verdict']} |")
    lines += ["", "## Paper record (all time)", "", "| | Orders | Open | Settled | P&L | LL engine | LL market |",
              "|---|---|---|---|---|---|---|"]
    for name, s in r.paper_summary.items():
        lines.append(f"| {name} | {s['orders']} | {s['open']} | {s['settled']} | ${s['pnl']} | "
                     f"{s['log_loss_engine']} | {s['log_loss_market']} |")
    lines += ["", f"Recorded today: {r.recorded} paper orders ({r.shadow} shadow).", "", "## Live", "",
              r.live_note or "No live orders."]
    for x in r.live:
        lines.append(f"- {x.get('action') or x.get('basket')} {x.get('ticker') or x.get('event')}: "
                     f"{'ok' if x.get('ok') or x.get('complete') else x.get('reason') or x.get('note')}")
    if r.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in r.notes]
    return "\n".join(lines) + "\n"


def run_daily(live: bool, live_cap: float, state_repo: Optional[str], log: Log = print) -> DailyResult:
    import asyncio

    from src.engines import ENGINE_NAMES, ledger
    from src.engines.evaluate import EvalConfig
    from src.engines.http import Fetcher
    from src.engines.market import KalshiPublic
    from src.engines.runner import basket_opportunities, execute_basket_live, execute_live, live_plan, scan
    from src.engines.trust import promotion_from_paper, save_weight, weight_for

    today = datetime.now(timezone.utc).date().isoformat()
    store = StateStore(state_repo, log=log)
    store.restore()
    result = DailyResult(date=today, state=f"persisted in {state_repo}" if store.ok else store.note)

    can_trade = live and live_credentials()
    if live and not can_trade:
        result.live_note = ("Live trading requested but no Kalshi credentials in the environment "
                            "(KALSHI_API_KEY + KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH); paper only.")

    # 0. Keep the weather calibration fresh (refit weekly on NOAA vs CLI history).
    cal = Path("data/engines/weather_calibration.json")
    if not cal.exists() or (datetime.now().timestamp() - cal.stat().st_mtime) > 7 * 86400:
        _cli("engines", "calibrate", "--days", "120", log=log)
        result.notes.append("weather calibration refit on the last 120 days")

    # 1. Learn from what settled.
    records = ledger.load()
    with Fetcher() as f:
        if records:
            ledger.settle(records, KalshiPublic(f), log=log)
            ledger.save(records)
    for engine in (e for e in ENGINE_NAMES if e != "arbitrage"):
        verdict = promotion_from_paper(records, engine)
        result.promotions[engine] = verdict
        current, _ = weight_for(engine)
        if verdict["settled"] >= 50 or current > 0:
            # Enough evidence to have an opinion: trust follows it, both ways.
            save_weight(engine, verdict["weight"],
                        f"daily {today}: shadow record {verdict['settled']} settled, P&L ${verdict['pnl']}, "
                        f"z={verdict['z']} ({verdict['verdict']})")
    if can_trade:
        for cmd in (("fills",), ("learnings",), ("improve",)):
            _cli(*cmd, log=log)

    # 2. Scan + paper-record.
    async def _equity() -> float:
        from src.clients.kalshi_client import KalshiClient

        c = KalshiClient()
        try:
            bal = await c.get_balance()
            return (int(bal.get("balance", 0) or 0) + int(bal.get("portfolio_value", 0) or 0)) / 100.0
        finally:
            await c.close()

    bankroll = min(asyncio.run(_equity()), live_cap) if can_trade else live_cap
    cfg = EvalConfig(bankroll=bankroll)
    with Fetcher() as f:
        report = scan(f, list(ENGINE_NAMES), cfg, log=log)
    result.notes += report.notes
    baskets, orders = live_plan(report)
    recs = [ledger.record_from_opportunity(o) for o in orders + report.shadow]
    for b in baskets:
        tag = f"{b.event_ticker}:{b.kind}:{today}"
        recs += [ledger.record_from_opportunity(o, basket=tag) for o in basket_opportunities(b)]
    ledger.append(recs)
    result.recorded, result.shadow = len(recs), len(report.shadow)

    # 3. Live: earned engines and risk-free baskets only.
    if can_trade:
        earned = {e for e in ENGINE_NAMES if e != "arbitrage" and weight_for(e)[0] > 0}
        send = [o for o in orders if o.engine in earned]
        held_back = len(orders) - len(send)

        async def _live():
            from src.clients.kalshi_client import KalshiClient

            c = KalshiClient()
            try:
                out = [await execute_basket_live(c, b) for b in baskets]
                return out + await execute_live(c, send, log=log)
            finally:
                await c.close()

        result.live = asyncio.run(_live()) if (send or baskets) else []
        result.live_note = (f"Bankroll for sizing ${bankroll:,.2f} (cap ${live_cap:,.0f}). "
                            f"Engines with earned trust: {', '.join(sorted(earned)) or 'none'}. "
                            f"Sent {len(send)} orders + {len(baskets)} baskets; held back {held_back} from "
                            "engines that haven't earned trust.")
        if not result.live:
            result.live_note += " Nothing qualified today."

    # 4. Report + save.
    result.paper_summary = ledger.summarize(ledger.load())
    out = Path(f"data/engines/reports/daily_{today}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(result))
    store.save(f"daily {today}: {result.recorded} paper, {len(result.live)} live")
    if store.note:
        result.state = store.note
    return result
