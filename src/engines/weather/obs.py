"""Observed temperatures so far in the current CLI day (NWS station observations).

Once the day has started, what already happened is a hard bound: the CLI
maximum can't be below the warmest reading so far, and the minimum can't be
above the coldest. Late in the afternoon that bound is most of the answer.

Readings are rounded to whole degrees the way the CLI reports them. The CLI
draws on higher-frequency sensor data than hourly observations, so the true
extreme can only be *more* extreme than what we see, never less: the bound is
safe in the direction we use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, Optional, Tuple

from src.engines.weather.stations import Station

NWS_API = "https://api.weather.gov"


@dataclass(frozen=True)
class ObservedSoFar:
    max_f: Optional[int]
    min_f: Optional[int]
    n_obs: int
    last_obs: Optional[datetime]
    max_exact: bool = True  # binding reading had tenths of a degree C (vs whole-degree C)
    min_exact: bool = True


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


# NWS quality-control flags for rejected, questioned or subjectively bad readings.
BAD_QC = {"X", "Q", "B"}
SPIKE_F = 3.0  # degrees of jump between near-simultaneous readings that looks like a glitch
SPIKE_WINDOW = timedelta(minutes=90)  # only readings this close together can vouch for each other


def _deglitch(series):
    """Drop isolated spikes from time-ordered (t, temp_f) readings.

    Pass 1: an interior reading that jumps more than SPIKE_F away from both
    neighbours (each within SPIKE_WINDOW, same direction) is a sensor glitch.
    Pass 2: the newest reading has no later neighbour to confirm it, so a jump
    from the previous clean reading is held back until the next observation.
    Changes spread over hours are weather and are always kept.
    """
    def close(a, b):
        return abs(a[0] - b[0]) <= SPIKE_WINDOW

    kept = []
    for i, cur in enumerate(series):
        if 0 < i < len(series) - 1:
            prev, nxt = series[i - 1], series[i + 1]
            if close(prev, cur) and close(cur, nxt):
                dp, dn = cur[1] - prev[1], cur[1] - nxt[1]
                if (dp > SPIKE_F and dn > SPIKE_F) or (dp < -SPIKE_F and dn < -SPIKE_F):
                    continue
        kept.append(cur)
    if len(kept) >= 2 and close(kept[-2], kept[-1]) and abs(kept[-1][1] - kept[-2][1]) > SPIKE_F:
        kept.pop()
    return kept


_TGROUP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")


def reading_celsius(props: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """(temperature C, half-width of its precision C) for one observation.

    Hourly METARs carry a T-group with tenths ("T02440233" = 24.4 C); prefer it.
    Without one, a whole-degree value (the 5-minute ASOS feed: 24.0, 25.0) only
    says the truth was within +-0.5 C, nearly a degree F either way.
    """
    m = _TGROUP.search(props.get("rawMessage") or "")
    if m:
        t = int(m.group(2)) / 10.0
        return (-t if m.group(1) == "1" else t), 0.05
    val = (props.get("temperature") or {}).get("value")
    if val is None:
        return None
    v = float(val)
    return v, (0.5 if abs(v - round(v)) < 1e-6 else 0.05)


def summarize_observations(features, start: datetime, end: datetime) -> ObservedSoFar:
    """Bounds on today's extremes that hold whatever each reading's precision:
    the high is at least the *lowest* temperature a reading could have been,
    the low at most the *highest*."""
    series = []
    for f in features or []:
        props = f.get("properties", {})
        ts = props.get("timestamp")
        reading = props.get("temperature") or {}
        if ts is None or reading.get("qualityControl") in BAD_QC:
            continue
        got = reading_celsius(props)
        if got is None:
            continue
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if start <= t < end:
            series.append((t, _c_to_f(got[0]), _c_to_f(got[0] - got[1]), _c_to_f(got[0] + got[1]), got[1] < 0.1))
    kept = _deglitch([(r[0], r[1]) for r in sorted(series)])
    keep_ts = {t for t, _ in kept}
    series = [r for r in sorted(series) if r[0] in keep_ts]
    if not series:
        return ObservedSoFar(None, None, 0, None)
    hi_r = max(series, key=lambda r: r[2])  # reading that sets the floor on the high
    lo_r = min(series, key=lambda r: r[3])  # reading that sets the ceiling on the low
    return ObservedSoFar(round(hi_r[2]), round(lo_r[3]), len(series), series[-1][0],
                         max_exact=hi_r[4], min_exact=lo_r[4])


def observed_so_far(fetcher, station: Station, now: Optional[datetime] = None) -> ObservedSoFar:
    """Extremes recorded since the start of the current CLI day."""
    now = now or datetime.now(timezone.utc)
    start, end = station.climate_day_utc(station.climate_date(now))
    data = fetcher.get(
        f"{NWS_API}/stations/{station.icao}/observations",
        {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ")},
        cache=True,
        cache_ttl=300,
        headers={"Accept": "application/geo+json"},
    )
    return summarize_observations(data.get("features"), start, min(end, now))
