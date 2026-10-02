"""Games engine: Kalshi game-winner markets priced from the sportsbook and Polymarket.

For each open Kalshi game (NFL, college football, MLB, NHL, NBA, WNBA) it forms
a consensus win probability from two independent books:

* **DraftKings** moneylines via ESPN (``sports-skills``), de-vigged two-way;
* **Polymarket**'s executable mid, read off the public CLOB book for the
  matching game (paired with Kalshi by ``sports-skills``' date + team-code
  matcher; the book endpoint is the one ``dr-manhattan``'s Polymarket client uses).

The fork's own research found liquid sports books efficient, and live-game
prices make pre-game lines stale instantly, so this engine starts with trust
0, skips games that have started, and must earn trust from its shadow paper
record (``cli.py engines promote games``) like any other engine.

``sports-skills`` is an optional dependency (``pip install sports-skills``);
without it the engine reports that and prices nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.engines.evaluate import Fair, logit, sigmoid
from src.engines.market import Event, KalshiPublic, Quote, devig

POLY_CLOB = "https://clob.polymarket.com"
SPORT_SERIES = {
    "nfl": "KXNFLGAME",
    "cfb": "KXNCAAFGAME",
    "mlb": "KXMLBGAME",
    "nhl": "KXNHLGAME",
    "nba": "KXNBAGAME",
    "wnba": "KXWNBAGAME",
}
# Kalshi ticker code -> ESPN abbreviation, where the two disagree.
CODE_ALIASES = {"WAS": {"WSH"}, "WSH": {"WAS"}, "CWS": {"CHW"}, "GSW": {"GS"}, "GS": {"GSW"},
                "JAC": {"JAX"}, "LA": {"LAR"}, "NOP": {"NO"}, "SA": {"SAS"}, "UTA": {"UTAH"}}
START_BUFFER = timedelta(minutes=10)  # pre-game lines go stale the moment play starts
MAX_POLY_SPREAD = 0.05
EASTERN = ZoneInfo("America/New_York")
_DATE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_MONTHS = {m: i for i, m in enumerate(
    ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}


def american_to_prob(odds: Any) -> Optional[float]:
    """'+170' -> 0.370, '-205' -> 0.672 (vig included)."""
    try:
        o = float(str(odds).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if o >= 100:
        return 100.0 / (o + 100.0)
    if o <= -100:
        return -o / (-o + 100.0)
    return None


def devig_moneyline(home: Any, away: Any) -> Optional[Tuple[float, float]]:
    ph, pa = american_to_prob(home), american_to_prob(away)
    if ph is None or pa is None or ph + pa <= 0:
        return None
    return ph / (ph + pa), pa / (ph + pa)


def name_matches(short: str, full: str) -> bool:
    """Kalshi's abbreviated team label vs an ESPN full name.

    "Washington" ~ "Washington Commanders"; "New York G" ~ "New York Giants"
    (a trailing initial); "Chicago WS" ~ "Chicago White Sox" (initials of the
    remaining words). Never matches "New York G" to "New York Jets".
    """
    s, f = short.lower().split(), full.lower().split()
    if not s:
        return False
    for i, tok in enumerate(s):
        if i >= len(f):
            return False
        if f[i].startswith(tok):
            continue
        rest = "".join(w[0] for w in f[i:])
        return i == len(s) - 1 and len(tok) >= 1 and rest.startswith(tok)
    return True


def team_matches(code: str, label: str, team: Dict[str, Any]) -> bool:
    abbr = (team.get("abbreviation") or "").upper()
    return (code == abbr or abbr in CODE_ALIASES.get(code, set())
            or name_matches(label, team.get("name") or ""))


def event_date(event_ticker: str) -> Optional[date]:
    m = _DATE.search(event_ticker)
    if not m or m.group(2) not in _MONTHS:
        return None
    try:
        return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))
    except ValueError:
        return None


def _start(game: Dict[str, Any]) -> Optional[datetime]:
    raw = game.get("start_time")
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def match_game(ev: Event, games: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], Dict[str, str]]]:
    """The ESPN game for a Kalshi game event, and ticker -> 'home'/'away'."""
    day = event_date(ev.event_ticker)
    ms = ev.markets
    if day is None or len(ms) != 2:
        return None
    hits = []
    for g in games:
        start = _start(g)
        if start is None or start.astimezone(EASTERN).date() != day:
            continue
        sides: Dict[str, str] = {}
        for m in ms:
            code = m.ticker.rsplit("-", 1)[-1]
            for side in ("home", "away"):
                if team_matches(code, m.subtitle, g.get(side) or {}):
                    sides.setdefault(m.ticker, side)
        if len(sides) == 2 and set(sides.values()) == {"home", "away"}:
            hits.append((g, sides))
    return hits[0] if len(hits) == 1 else None  # ambiguity is refusal, not a guess


def polymarket_mid(fetcher, token_id: str) -> Optional[float]:
    book = fetcher.get(f"{POLY_CLOB}/book", {"token_id": token_id}, cache=True, cache_ttl=60)
    bids = [float(b["price"]) for b in book.get("bids") or []]
    asks = [float(a["price"]) for a in book.get("asks") or []]
    if not bids or not asks:
        return None
    bid, ask = max(bids), min(asks)
    if ask - bid > MAX_POLY_SPREAD or ask <= bid:
        return None
    return (bid + ask) / 2.0


@dataclass
class PricedGame:
    event: Event
    sport: str
    title: str
    start: str
    sources: Dict[str, Dict[str, float]]  # source -> ticker -> p_yes
    fairs: List[Fair] = field(default_factory=list)


class GamesEngine:
    name = "games"

    def __init__(self, fetcher, kp: Optional[KalshiPublic] = None, weight: float = 0.0,
                 now: Optional[datetime] = None, log: Callable[[str], None] = lambda *_: None,
                 sports: Optional[Any] = None):
        self.fetcher = fetcher
        self.kp = kp or KalshiPublic(fetcher)
        self.weight = weight
        self.now = now or datetime.now(timezone.utc)
        self.log = log
        self.sports = sports  # the sports_skills.markets module (injectable for tests)

    def _markets_api(self):
        if self.sports is None:
            from sports_skills import markets  # optional dependency

            self.sports = markets
        return self.sports

    def _poly_by_event(self, sport: str) -> Dict[str, Dict[str, Any]]:
        try:
            res = self._markets_api().match_markets(sport=sport)
        except Exception as exc:
            self.log(f"  polymarket match {sport}: {exc}")
            return {}
        out = {}
        for mt in (res.get("data") or {}).get("matches") or []:
            k = (mt.get("kalshi") or {}).get("event_ticker")
            pm = ((mt.get("polymarket") or {}).get("markets") or [None])[0]
            if k and pm:
                out[k] = pm
        return out

    def price_sport(self, sport: str) -> List[PricedGame]:
        api = self._markets_api()
        sched = api.get_sport_schedule(sport=sport)
        games = [g for g in (sched.get("data") or {}).get("games") or []
                 if (_start(g) or self.now) > self.now + START_BUFFER]
        if not games:
            return []
        poly = self._poly_by_event(sport)
        out = []
        for ev in self.kp.iter_events(status="open", series_ticker=SPORT_SERIES[sport]):
            hit = match_game(ev, games)
            if not hit:
                continue
            g, sides = hit
            sources: Dict[str, Dict[str, float]] = {}
            ml = ((g.get("espn_odds") or {}).get("moneyline")) or {}
            dv = devig_moneyline(ml.get("home"), ml.get("away"))
            if dv:
                p = {"home": dv[0], "away": dv[1]}
                sources["draftkings"] = {t: p[side] for t, side in sides.items()}
            pm = poly.get(ev.event_ticker)
            if pm:
                mids = {}
                for oc in pm.get("outcomes") or []:
                    for t, side in sides.items():
                        if name_matches(oc.get("name", ""), (g.get(side) or {}).get("name", "")) or \
                                (g.get(side) or {}).get("name", "").lower().endswith(oc.get("name", "").lower()):
                            try:
                                mids[t] = polymarket_mid(self.fetcher, oc["clob_token_id"])
                            except Exception as exc:
                                self.log(f"  polymarket book {ev.event_ticker}: {exc}")
                if len(mids) == 2 and all(v is not None for v in mids.values()):
                    total = sum(mids.values())
                    sources["polymarket"] = {t: v / total for t, v in mids.items()}
            if not sources:
                continue
            implied = devig(ev.markets)
            fairs = []
            for q in ev.markets:
                vals = [src[q.ticker] for src in sources.values() if q.ticker in src]
                p = sigmoid(sum(logit(v) for v in vals) / len(vals))
                fairs.append(Fair(
                    ticker=q.ticker, p_yes=p, weight=self.weight, stderr=0.01,
                    p_market_yes=implied.get(q.ticker),
                    rationale=f"{g.get('short_name') or ev.title}: " + ", ".join(
                        f"{k} {v[q.ticker]:.3f}" for k, v in sources.items() if q.ticker in v),
                    meta={"sport": sport, "start": g.get("start_time"), "sources": sorted(sources)},
                ))
            out.append(PricedGame(ev, sport, ev.title, g.get("start_time", ""), sources, fairs))
        return out

    def scan(self, sports: Optional[List[str]] = None) -> List[PricedGame]:
        out = []
        for sport in sports or list(SPORT_SERIES):
            try:
                out.extend(self.price_sport(sport))
            except Exception as exc:  # one league's outage shouldn't stop the others
                self.log(f"  games {sport}: {exc}")
        return out
