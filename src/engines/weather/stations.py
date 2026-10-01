"""Settlement stations for Kalshi's daily temperature markets.

Every US temperature series settles on the NWS Daily Climate Report (``CLI``)
for one station, named in the market rules as e.g. "recorded at New York City
(CLINYC)". The CLI day runs midnight to midnight *local standard time*, so in
summer it spans 1:00 AM to 12:59 AM daylight time.

Coordinates and time zones come from ``api.weather.gov/stations/{ICAO}``.
Note the stations that are not the obvious airport: NYC is Central Park (a
leafy site that runs cooler than models expect), Chicago is Midway, Houston is
Hobby, Washington is Reagan National.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Station:
    icao: str
    name: str
    lat: float
    lon: float
    tz: str

    def standard_offset(self, on: date) -> timedelta:
        """UTC offset of local *standard* time (DST removed) on ``on``."""
        zone = ZoneInfo(self.tz)
        noon = datetime(on.year, on.month, on.day, 12, tzinfo=zone)
        return noon.utcoffset() - (noon.dst() or timedelta(0))

    def climate_day_utc(self, on: date) -> Tuple[datetime, datetime]:
        """[start, end) of the CLI day for ``on``, in UTC."""
        start_local = datetime(on.year, on.month, on.day)
        off = self.standard_offset(on)
        start = (start_local - off).replace(tzinfo=timezone.utc)
        return start, start + timedelta(days=1)

    def climate_date(self, at: datetime) -> date:
        """The CLI day an instant belongs to."""
        approx = at.astimezone(ZoneInfo(self.tz)).date()
        off = self.standard_offset(approx)
        return (at.astimezone(timezone.utc) + off).date()


STATIONS = {
    "ATL": Station("KATL", "Atlanta Hartsfield-Jackson", 33.6403, -84.4269, "America/New_York"),
    "AUS": Station("KAUS", "Austin-Bergstrom", 30.183, -97.6799, "America/Chicago"),
    "BOS": Station("KBOS", "Boston Logan", 42.3606, -71.0106, "America/New_York"),
    "DCA": Station("KDCA", "Washington Reagan National", 38.8483, -77.0342, "America/New_York"),
    "DEN": Station("KDEN", "Denver International", 39.8466, -104.6562, "America/Denver"),
    "DFW": Station("KDFW", "Dallas/Fort Worth", 32.8974, -97.022, "America/Chicago"),
    "EWR": Station("KEWR", "Newark", 40.6825, -74.1694, "America/New_York"),
    "HOU": Station("KHOU", "Houston Hobby", 29.6375, -95.2825, "America/Chicago"),
    "LAS": Station("KLAS", "Las Vegas Harry Reid", 36.0719, -115.1634, "America/Los_Angeles"),
    "LAX": Station("KLAX", "Los Angeles International", 33.9381, -118.3889, "America/Los_Angeles"),
    "MDW": Station("KMDW", "Chicago Midway", 41.7842, -87.7553, "America/Chicago"),
    "MIA": Station("KMIA", "Miami International", 25.7906, -80.3164, "America/New_York"),
    "MSP": Station("KMSP", "Minneapolis-St. Paul", 44.8831, -93.2289, "America/Chicago"),
    "MSY": Station("KMSY", "New Orleans", 29.9928, -90.2508, "America/Chicago"),
    "NYC": Station("KNYC", "New York Central Park", 40.7833, -73.9667, "America/New_York"),
    "OKC": Station("KOKC", "Oklahoma City Will Rogers", 35.3886, -97.6003, "America/Chicago"),
    "PHL": Station("KPHL", "Philadelphia International", 39.8733, -75.2268, "America/New_York"),
    "PHX": Station("KPHX", "Phoenix Sky Harbor", 33.4278, -112.0035, "America/Phoenix"),
    "SAN": Station("KSAN", "San Diego International", 32.7336, -117.1831, "America/Los_Angeles"),
    "SAT": Station("KSAT", "San Antonio International", 29.5328, -98.4636, "America/Chicago"),
    "SDF": Station("KSDF", "Louisville", 38.1741, -85.7365, "America/Kentucky/Louisville"),
    "SEA": Station("KSEA", "Seattle-Tacoma", 47.4447, -122.3136, "America/Los_Angeles"),
    "SFO": Station("KSFO", "San Francisco International", 37.6196, -122.3656, "America/Los_Angeles"),
    "TTN": Station("KTTN", "Trenton-Mercer", 40.2764, -74.8164, "America/New_York"),
}

_CLI_RE = re.compile(r"\(CLI([A-Z]{3})\)")
_MONTHS = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}
_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(?:-|$)")


def station_from_rules(rules: str) -> Optional[Tuple[str, Station]]:
    """``("NYC", Station)`` from a rules string naming ``(CLINYC)``."""
    m = _CLI_RE.search(rules or "")
    if not m or m.group(1) not in STATIONS:
        return None
    return m.group(1), STATIONS[m.group(1)]


def kind_from_rules(rules: str) -> Optional[str]:
    text = (rules or "").lower()
    if "maximum temperature" in text:
        return "high"
    if "minimum temperature" in text:
        return "low"
    return None


def date_from_event(event_ticker: str) -> Optional[date]:
    """``KXHIGHNY-26OCT02`` -> 2026-10-02."""
    m = _DATE_RE.search(event_ticker or "")
    if not m or m.group(2) not in _MONTHS:
        return None
    try:
        return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))
    except ValueError:
        return None
