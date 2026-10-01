"""Historical NWS Daily Climate Reports (the settlement source), via IEM.

Kalshi's temperature markets settle on these exact numbers, so they are the
ground truth for calibrating the model, independent of whether Kalshi listed
a market that day.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Optional, Tuple

IEM_CLI = "https://mesonet.agron.iastate.edu/json/cli.py"


def _int(raw) -> Optional[int]:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None  # IEM uses "M" for missing


def parse_cli(results) -> Dict[date, Tuple[Optional[int], Optional[int]]]:
    out = {}
    for r in results or []:
        try:
            d = date.fromisoformat(r["valid"])
        except (KeyError, ValueError):
            continue
        out[d] = (_int(r.get("high")), _int(r.get("low")))
    return out


def fetch_cli_year(fetcher, icao: str, year: int, cache_ttl: Optional[float] = None) -> Dict[date, Tuple[Optional[int], Optional[int]]]:
    """``{date: (high, low)}`` for one station-year. Past years never change, so
    callers pass ``cache_ttl=None`` for them and a few hours for the current year."""
    data = fetcher.get(IEM_CLI, {"station": icao, "year": year}, cache=True, cache_ttl=cache_ttl)
    return parse_cli(data.get("results"))
