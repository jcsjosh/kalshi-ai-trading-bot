"""Independent second opinion on arbitrage baskets from oracle3's constraint checker.

oracle3 (github.com/YichengYang-Ethan/oracle3-prediction-market-agent) prices
probability-axiom baskets with its own fee implementation. When it is
importable (``pip install oracle3``, or ``ORACLE3_PATH`` pointing at a clone),
every basket our engine calls risk-free is re-priced there, and a basket
oracle3 doesn't confirm is not traded. Two implementations agreeing is cheap
insurance against a bug in either.

Basket kinds map onto oracle3 relations:

* all-NO   -> ``exclusivity`` (NO on every outcome pays n - 1)
* all-YES  -> ``event_sum``   (YES on every outcome pays 1)
* ladder   -> ``implication`` with A = the unlikely leg, B = the likely leg
              ("above X2" implies "above X1"; the basket is NO on A + YES on B)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from src.engines.arbitrage import Basket
from src.engines.fees import FeeSchedule


def _oracle3():
    path = os.environ.get("ORACLE3_PATH")
    if path and path not in sys.path:
        sys.path.append(path)
    try:
        from oracle3 import arbitrage, fees  # noqa: PLC0415
    except ImportError:
        return None
    return arbitrage, fees


def available() -> bool:
    return _oracle3() is not None


def oracle3_verdict(b: Basket, schedule: FeeSchedule) -> Optional[Dict[str, Any]]:
    """oracle3's net edge for ``b`` at its walked prices, or None if oracle3 isn't installed."""
    mods = _oracle3()
    if mods is None:
        return None
    arbitrage, o3fees = mods
    from decimal import Decimal

    sched = o3fees.KalshiSchedule(multiplier=Decimal(str(schedule.multiplier)), maker_fees=schedule.maker_fees)

    def quote(leg):
        kw = {"yes_ask": leg.price} if leg.side == "yes" else {"no_ask": leg.price}
        return arbitrage.Quote(market_id=leg.ticker, venue="kalshi", schedule=sched, **kw)

    if b.kind == "all-NO":
        relation, quotes, want = "exclusivity", [quote(l) for l in b.legs], "NO on every outcome"
    elif b.kind == "all-YES":
        relation, quotes, want = "event_sum", [quote(l) for l in b.legs], "YES on every outcome"
    elif b.kind.startswith("ladder"):
        likely, unlikely = b.legs  # (YES on the likelier strike, NO on the less likely)
        relation, quotes, want = "implication", [quote(unlikely), quote(likely)], "NO on A + YES on B"
    else:
        return {"relation": None, "agrees": False, "reason": f"no oracle3 relation for {b.kind}"}

    check = arbitrage.check_constraint(relation, quotes, contracts=float(b.count))
    match = next((x for x in check.baskets if x.description == want), None)
    if match is None:
        return {"relation": relation, "agrees": False, "reason": f"missing quotes {check.missing_quotes}"}
    # oracle3 prices every contract at the worst level we walked to (we average
    # across levels), so its edge is the conservative one: it must still be positive.
    return {
        "relation": relation,
        "net_edge": round(match.net_edge, 4),
        "ours": round(b.profit * b.count, 4),
        "agrees": match.net_edge > 0,
    }
