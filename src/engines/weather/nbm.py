"""NOAA National Blend of Models (NBM) station forecasts, via the IEM archive.

The NBM short-range text bulletin (``NBS``) gives, for each station, the
forecast daily maximum / minimum temperature (``txn``) and NOAA's own standard
deviation for it (``xnd``). It is a calibrated blend of ~30 models, issued for
the exact stations Kalshi settles on, and IEM archives every run with its
issue time, so a backtest can use only what was knowable at decision time.

``txn`` at an ``ftime`` of 00Z is the daytime maximum ending that evening; at
12Z it is the overnight minimum ending that morning.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

IEM_MOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"

# NBM text products reach the archive roughly an hour after the cycle; assume
# two so a backtest never peeks at a run before it was published.
PUBLISH_LATENCY = timedelta(hours=2)


@dataclass(frozen=True)
class NbmForecast:
    runtime: datetime
    ftime: datetime
    txn: float
    xnd: float

    def available_at(self) -> datetime:
        return self.runtime + PUBLISH_LATENCY


def _ts(raw: str) -> datetime:
    return datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def parse_nbs_csv(text: str) -> List[NbmForecast]:
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        txn, xnd = (row.get("txn") or "").strip(), (row.get("xnd") or "").strip()
        if not txn:
            continue
        try:
            out.append(NbmForecast(_ts(row["runtime"]), _ts(row["ftime"]), float(txn),
                                   float(xnd) if xnd else 0.0))
        except (KeyError, ValueError):
            continue
    return out


def fetch_nbs(fetcher, icao: str, start: datetime, end: datetime, cache_ttl: Optional[float] = 1800) -> List[NbmForecast]:
    """NBS ``txn``/``xnd`` rows for runs issued in [start, end]."""
    params = {
        "station": icao,
        "model": "NBS",
        "sts": start.strftime("%Y-%m-%dT%H:%MZ"),
        "ets": end.strftime("%Y-%m-%dT%H:%MZ"),
        "format": "csv",
    }
    text = fetcher.get(IEM_MOS, params, as_text=True, cache=True, cache_ttl=cache_ttl)
    return parse_nbs_csv(text)


def target_ftime(target: date, kind: str) -> datetime:
    """The NBS valid time whose ``txn`` forecasts this CLI day's high or low."""
    if kind == "high":
        nxt = target + timedelta(days=1)
        return datetime(nxt.year, nxt.month, nxt.day, 0, tzinfo=timezone.utc)
    return datetime(target.year, target.month, target.day, 12, tzinfo=timezone.utc)


class NbmIndex:
    """Forecast rows indexed by valid time, for fast as-of lookups."""

    def __init__(self, rows: Iterable[NbmForecast]):
        self.by_ftime: Dict[datetime, List[NbmForecast]] = {}
        for r in rows:
            self.by_ftime.setdefault(r.ftime, []).append(r)
        for lst in self.by_ftime.values():
            lst.sort(key=lambda r: r.runtime)

    def latest(self, target: date, kind: str, as_of: datetime) -> Optional[NbmForecast]:
        best = None
        for r in self.by_ftime.get(target_ftime(target, kind), []):
            if r.available_at() > as_of:
                break
            best = r
        return best


def latest_forecast(
    rows: Iterable[NbmForecast], target: date, kind: str, as_of: datetime
) -> Optional[NbmForecast]:
    """Most recent run, published by ``as_of``, that forecasts ``target``'s high/low."""
    index = rows if isinstance(rows, NbmIndex) else NbmIndex(rows)
    return index.latest(target, kind, as_of)
