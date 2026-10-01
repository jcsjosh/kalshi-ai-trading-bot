"""Live weather engine: price every open Kalshi temperature market.

For each open event it finds the settlement station and day from the rules,
takes the latest published NBM forecast for that station, applies the fitted
calibration, truncates by what has already been observed today, and returns a
``Fair`` per market. The ``weight`` (trust vs the market) comes from the last
backtest; with no measured edge it is 0 and the engine can only report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.engines.evaluate import Fair
from src.engines.market import Event, KalshiPublic, Quote
from src.engines.weather.backtest import devig, discover_temperature_series
from src.engines.weather.calibration import DEFAULT_CALIBRATION_PATH, WeatherCalibration
from src.engines.weather.model import TempDistribution
from src.engines.weather.nbm import NbmIndex, fetch_nbs
from src.engines.weather.obs import ObservedSoFar, observed_so_far
from src.engines.weather.stations import (
    STATIONS,
    date_from_event,
    kind_from_rules,
    station_from_rules,
)

ACTIVE_SERIES_PATH = Path("data/engines/weather_active_series.json")
PEAK_HOUR_LST = 16  # after this, today's high is mostly set and the forecast part is stale
BASE_STDERR = 0.01
LATE_DAY_STDERR = 0.08


@dataclass
class PricedEvent:
    event: Event
    code: str
    kind: str
    target: str
    mu: float
    sigma: float
    forecast_run: str
    observed: Optional[ObservedSoFar]
    fairs: List[Fair]


def active_series(kp: KalshiPublic, max_age_hours: float = 12.0, path: Path = ACTIVE_SERIES_PATH) -> List[str]:
    """Temperature series with open events, re-discovered at most every ``max_age_hours``."""
    now = datetime.now(timezone.utc)
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if now - datetime.fromisoformat(d["at"]) < timedelta(hours=max_age_hours):
                return d["series"]
        except (ValueError, KeyError):
            pass
    found = []
    for s in discover_temperature_series(kp):
        data = kp.get("/events", {"series_ticker": s, "status": "open", "limit": 1})
        if data.get("events"):
            found.append(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"at": now.isoformat(), "series": found}))
    return found


class WeatherEngine:
    name = "weather"

    def __init__(self, fetcher, kp: Optional[KalshiPublic] = None,
                 calibration: Optional[WeatherCalibration] = None, now: Optional[datetime] = None,
                 log=lambda *_: None):
        self.fetcher = fetcher
        self.kp = kp or KalshiPublic(fetcher)
        self.cal = calibration or WeatherCalibration.load(DEFAULT_CALIBRATION_PATH)
        self.now = now or datetime.now(timezone.utc)
        self.log = log
        self._nbm: Dict[str, NbmIndex] = {}
        self._obs: Dict[str, Optional[ObservedSoFar]] = {}

    @property
    def weight(self) -> float:
        return float(self.cal.weight or 0.0)

    def _forecasts(self, code: str) -> NbmIndex:
        if code not in self._nbm:
            rows = fetch_nbs(self.fetcher, STATIONS[code].icao, self.now - timedelta(hours=30), self.now,
                             cache_ttl=900)
            self._nbm[code] = NbmIndex(rows)
        return self._nbm[code]

    def _observed(self, code: str) -> Optional[ObservedSoFar]:
        if code not in self._obs:
            try:
                self._obs[code] = observed_so_far(self.fetcher, STATIONS[code], self.now)
            except Exception as exc:
                self.log(f"  observations {code}: {exc}")
                self._obs[code] = None
        return self._obs[code]

    def price_event(self, ev: Event) -> Optional[PricedEvent]:
        ms = ev.active_markets()
        rules = next((m.rules for m in ev.markets if m.rules), "")
        st, kind, target = station_from_rules(rules), kind_from_rules(rules), date_from_event(ev.event_ticker)
        if not ms or not st or not kind or not target:
            return None
        code, station = st
        f = self._forecasts(code).latest(target, kind, self.now)
        if f is None:
            return None

        lower = upper = None
        stderr = BASE_STDERR
        observed = None
        today = station.climate_date(self.now)
        if target < today:
            return None  # the day is over; settlement is a lookup, not a forecast
        if target == today:
            observed = self._observed(code)
            if observed is None:
                return None  # same-day without observations would ignore what everyone else can see
            if kind == "high" and observed.max_f is not None:
                lower = observed.max_f
            if kind == "low" and observed.min_f is not None:
                upper = observed.min_f
            local = self.now.astimezone(ZoneInfo(station.tz))
            lst_hour = (local - (local.dst() or timedelta(0))).hour
            if kind == "high" and lst_hour >= PEAK_HOUR_LST:
                stderr = LATE_DAY_STDERR
            if kind == "low" and lst_hour >= 9:
                stderr = LATE_DAY_STDERR  # overnight low is in; a cold evening is what's left

        params = self.cal.params(code, kind)
        dist = TempDistribution.from_forecast(f.txn, f.xnd, params, lower=lower, upper=upper)
        implied = devig(ms)
        fairs = []
        for q in ms:
            p = dist.prob(q)
            if p is None:
                continue
            # An outcome the observations have already ruled out needs no forecast skill.
            certain = p <= 1e-6 or p >= 1 - 1e-6
            fairs.append(Fair(
                ticker=q.ticker,
                p_yes=p,
                weight=1.0 if certain else self.weight,
                stderr=0.0 if certain else stderr,
                p_market_yes=implied.get(q.ticker),
                rationale=(f"{station.name} {kind} {target}: NBM {f.txn:.0f}F +/-{f.xnd:.0f} "
                           f"(run {f.runtime:%m-%d %HZ}) -> mu {dist.mu:.1f} sigma {dist.sigma:.1f}"
                           + (f"; observed {'max' if kind == 'high' else 'min'} so far "
                              f"{lower if kind == 'high' else upper}F" if observed else "")),
                meta={"station": code, "kind": kind, "target": target.isoformat(), "mu": round(dist.mu, 2),
                      "sigma": round(dist.sigma, 2), "same_day": target == today, "bound_certain": certain},
            ))
        return PricedEvent(ev, code, kind, target.isoformat(), round(dist.mu, 2), round(dist.sigma, 2),
                           f.runtime.isoformat(), observed, fairs)

    def scan(self, series: Optional[List[str]] = None) -> List[PricedEvent]:
        out = []
        for s in series or active_series(self.kp):
            for ev in self.kp.iter_events(status="open", series_ticker=s):
                try:
                    priced = self.price_event(ev)
                except Exception as exc:  # one station's outage shouldn't stop the scan
                    self.log(f"  {ev.event_ticker}: {exc}")
                    continue
                if priced:
                    out.append(priced)
        return out


def quotes_by_ticker(priced: List[PricedEvent]) -> Dict[str, Tuple[Quote, Fair]]:
    out = {}
    for pe in priced:
        qs = {q.ticker: q for q in pe.event.markets}
        for f in pe.fairs:
            if f.ticker in qs:
                out[f.ticker] = (qs[f.ticker], f)
    return out
