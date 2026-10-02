"""How much an engine is trusted over the market, and how that trust is earned.

Trust is the weight ``w`` in ``logit(p_fair) = w * logit(p_engine) + (1 - w) *
logit(p_market)``. It starts at 0 for every engine and is only ever *earned*,
from out-of-sample evidence, by one of two routes:

* a walk-forward backtest (``cli.py engines backtest --save-weight``), or
* the engine's own shadow paper record (``cli.py engines promote <engine>``).

Either way the bar is the same: blending the engine in must beat the market's
log loss by at least ``z`` standard errors, *and* the trades it would have made
must have made money. The second condition exists because a model can be
slightly informative on average and still be worst exactly where it bets.

Earned weights live in ``data/engines/trust.json`` with a note of the evidence.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize

from src.engines.evaluate import logit

TRUST_PATH = Path("data/engines/trust.json")
EPS = 1e-6


def log_loss(p, y):
    """Binary log loss; works on floats or numpy arrays."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def scores(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    """Brier score and mean log loss of ``row[key]`` against ``row["outcome"]``."""
    if not rows:
        return {"brier": float("nan"), "log_loss": float("nan")}
    p = np.array([r[key] for r in rows], float)
    y = np.array([r["outcome"] for r in rows], float)
    return {
        "brier": round(float(np.mean((np.clip(p, EPS, 1 - EPS) - y) ** 2)), 5),
        "log_loss": round(float(np.mean(log_loss(p, y))), 5),
    }


def _arrays(rows):
    pm = np.array([logit(r["p_model"]) for r in rows])
    pk = np.array([logit(r["p_market"]) for r in rows])
    y = np.array([r["outcome"] for r in rows], float)
    return pm, pk, y


def fit_weight(rows: Sequence[Dict[str, Any]]) -> float:
    """Log-loss-optimal trust in the model vs the market on resolved rows
    (each row: ``p_model``, ``p_market``, ``outcome``)."""
    if not rows:
        return 0.0
    pm, pk, y = _arrays(rows)

    def nll(w):
        return float(np.mean(log_loss(1.0 / (1.0 + np.exp(-(w * pm + (1 - w) * pk))), y)))

    return float(optimize.minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded").x)


def earned_weight(rows: Sequence[Dict[str, Any]], z: float = 2.0) -> Tuple[float, float]:
    """The weight the record has *earned*, and the z-score behind it.

    Fit the optimal blend weight, then ask whether blending beat the market by
    more than noise: the log-loss improvement, summed per event (outcomes in one
    event are not independent), must be ``z`` standard errors above zero.
    Otherwise 0: a weight fitted on noise is how a backtest talks itself into
    losing trades.
    """
    if len(rows) < 50:
        return 0.0, 0.0
    w = fit_weight(rows)
    if w <= 0:
        return 0.0, 0.0
    pm, pk, y = _arrays(rows)
    gain = log_loss(1.0 / (1.0 + np.exp(-pk)), y) - log_loss(1.0 / (1.0 + np.exp(-(w * pm + (1 - w) * pk))), y)
    groups: Dict[tuple, float] = defaultdict(float)
    for r, g in zip(rows, gain):
        groups[(r.get("event") or r["ticker"].rsplit("-", 1)[0], r.get("schedule"))] += float(g)
    vals = np.array(list(groups.values()))
    if len(vals) < 20 or vals.std(ddof=1) <= 0:
        return 0.0, 0.0
    zscore = float(vals.mean() / (vals.std(ddof=1) / math.sqrt(len(vals))))
    return (w if zscore >= z else 0.0), round(zscore, 2)


# -- the store -------------------------------------------------------------------


def load_trust(path: Path = TRUST_PATH) -> Dict[str, Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def weight_for(engine: str, default: float = 0.0, path: Path = TRUST_PATH) -> Tuple[float, Optional[str]]:
    entry = load_trust(path).get(engine)
    if not entry:
        return default, None
    return float(entry.get("weight") or 0.0), entry.get("source")


def save_weight(engine: str, weight: float, source: str, path: Path = TRUST_PATH) -> None:
    data = load_trust(path)
    data[engine] = {"weight": round(float(weight), 4), "source": source,
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


# -- promotion from the paper record -------------------------------------------


def promotion_from_paper(records: Sequence[Dict[str, Any]], engine: str) -> Dict[str, Any]:
    """Score an engine's settled *shadow* paper orders and return the weight they earn.

    Shadow orders are what the engine would have bought trusting its own
    model fully, so they test the model rather than the blend. Orders whose
    outcome was already certain from observations are excluded: they need no
    forecasting skill and would flatter the record.
    """
    rows, pnl, staked = [], 0.0, 0.0
    for r in records:
        if r.get("engine") != engine or not r.get("shadow") or r.get("certain"):
            continue
        out = r.get("outcome")
        if not out or not r.get("filled") or r.get("p_market") is None:
            continue
        rows.append({"p_model": r["p_engine"], "p_market": r["p_market"], "outcome": 1 if out["won"] else 0,
                     "ticker": r["ticker"], "event": r.get("event_ticker")})
        pnl += out["pnl"]
        staked += r["contracts"] * (r["price"] + r["fee"])
    w, zscore = earned_weight(rows)
    if pnl <= 0:
        w = 0.0
    return {"engine": engine, "settled": len(rows), "pnl": round(pnl, 2), "staked": round(staked, 2),
            "z": zscore, "weight": round(w, 4),
            "verdict": ("earned" if w > 0 else
                        "not enough settled shadow orders" if len(rows) < 50 else
                        "no edge over the market" if zscore < 2 else "lost money")}
