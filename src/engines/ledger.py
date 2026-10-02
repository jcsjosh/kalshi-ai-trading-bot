"""Paper ledger: forward-test engines on live prices without risking money.

Every dry run of ``cli.py engines run`` records the orders it *would* have
placed, at live book prices, with the probabilities behind them. ``settle``
later attaches outcomes from Kalshi's settlements and scores each engine:
P&L, ROI, and whether its probabilities beat the market's at the moment of
decision. This is the forward, out-of-sample track record a backtest can't
give you.

Fill assumptions (stated in every summary): a taker order fills at the price
recorded, since it was priced off the live ask; a resting (maker) order counts
as filled only if Kalshi later printed a trade at or through its price before
the market closed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.engines.evaluate import Opportunity
from src.engines.trust import log_loss

DEFAULT_LEDGER_PATH = Path("data/engines/paper_ledger.jsonl")


def record_from_opportunity(o: Opportunity, now: Optional[datetime] = None, basket: Optional[str] = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    return {
        "ts": now.isoformat(timespec="seconds"),
        "engine": o.engine,
        "ticker": o.ticker,
        "event_ticker": o.event_ticker,
        "series": o.series_ticker,
        "side": o.side,
        "role": o.role,
        "price": o.price,
        "fee": o.fee,
        "contracts": o.contracts,
        "p_engine": o.p_engine,
        "p_market": o.p_market,
        "p_fair": o.p_fair,
        "ev": o.ev,
        "basket": basket,
        "shadow": bool(o.meta.get("shadow")),  # priced at full trust; tests the model, never sent
        "certain": bool(o.meta.get("bound_certain")),  # decided by observations, not forecast skill
        "rationale": o.rationale,
        "filled": True if o.role == "taker" else None,
        "outcome": None,
    }


def append(records: List[Dict[str, Any]], path: Path | str = DEFAULT_LEDGER_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load(path: Path | str = DEFAULT_LEDGER_PATH) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def save(records: List[Dict[str, Any]], path: Path | str = DEFAULT_LEDGER_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    tmp.replace(path)


def maker_filled(record: Dict[str, Any], trades: List[Dict[str, Any]]) -> bool:
    """Did anyone trade at or through our resting price after we posted it?"""
    posted = datetime.fromisoformat(record["ts"])
    for t in trades:
        ts = t.get("created_time")
        if not ts or datetime.fromisoformat(ts.replace("Z", "+00:00")) < posted:
            continue
        try:
            yes_px = float(t.get("yes_price_dollars"))
        except (TypeError, ValueError):
            continue
        # Our YES bid at p fills when YES trades at <= p; our NO bid at p when YES trades at >= 1 - p.
        if record["side"] == "yes" and yes_px <= record["price"] + 1e-9:
            return True
        if record["side"] == "no" and yes_px >= 1.0 - record["price"] - 1e-9:
            return True
    return False


def settle(records: List[Dict[str, Any]], kp, log=lambda *_: None) -> int:
    """Attach outcomes to settled paper orders in place. Returns how many settled."""
    n = 0
    markets: Dict[str, Dict[str, Any]] = {}
    for r in records:
        if r.get("outcome") is not None:
            continue
        t = r["ticker"]
        if t not in markets:
            try:
                markets[t] = kp.get(f"/markets/{t}").get("market", {})
            except Exception as exc:
                log(f"  {t}: {exc}")
                markets[t] = {}
        m = markets[t]
        result = m.get("result")
        if result not in ("yes", "no"):
            continue
        if r.get("filled") is None:
            posted = int(datetime.fromisoformat(r["ts"]).timestamp())
            try:
                trades = kp.get("/markets/trades", {"ticker": t, "limit": 1000, "min_ts": posted}).get("trades", [])
            except Exception:
                trades = []  # can't prove a fill: count it as unfilled
            r["filled"] = maker_filled(r, trades)
        won = (result == "yes") == (r["side"] == "yes")
        pnl = r["contracts"] * ((1.0 if won else 0.0) - r["price"] - r["fee"]) if r["filled"] else 0.0
        r["outcome"] = {"result": result, "won": won, "pnl": round(pnl, 2),
                        "settled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        n += 1
    return n


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per engine (live-trust and shadow records kept apart): fills, P&L, and
    log loss of the *engine's own* probability vs the market's for the side
    bought. Outcomes decided by observations are left out of the log-loss
    comparison: they need no forecasting skill and would flatter the engine."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        name = r["engine"] + (" (shadow)" if r.get("shadow") else "")
        groups[name].append(r)
        groups["ALL" + (" (shadow)" if r.get("shadow") else "")].append(r)
    out = {}
    for name, rs in sorted(groups.items()):
        settled = [r for r in rs if r.get("outcome")]
        filled = [r for r in settled if r.get("filled")]
        staked = sum(r["contracts"] * (r["price"] + r["fee"]) for r in filled)
        pnl = sum(r["outcome"]["pnl"] for r in filled)
        scored = [r for r in filled if r.get("p_market") is not None and not r.get("certain")]
        y = np.array([1.0 if r["outcome"]["won"] else 0.0 for r in scored])
        ll = (lambda key: round(float(np.mean(log_loss(np.array([r[key] for r in scored]), y))), 4)
              if scored else None)
        out[name] = {
            "orders": len(rs),
            "open": len(rs) - len(settled),
            "settled": len(settled),
            "filled": len(filled),
            "wins": sum(1 for r in filled if r["outcome"]["won"]),
            "staked": round(staked, 2),
            "pnl": round(pnl, 2),
            "roi": round(pnl / staked, 4) if staked else None,
            "expected_pnl": round(sum(r["contracts"] * r["ev"] for r in filled), 2),
            "log_loss_engine": ll("p_engine"),
            "log_loss_market": ll("p_market"),
        }
    return out
