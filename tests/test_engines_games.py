"""Games engine: odds maths, ESPN <-> Kalshi matching, consensus pricing (sports-skills mocked)."""

from datetime import datetime, timezone

import pytest

from src.engines.games import (
    GamesEngine,
    american_to_prob,
    devig_moneyline,
    event_date,
    match_game,
    name_matches,
)
from src.engines.market import Event, Quote

NOW = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)


def test_american_odds_and_devig():
    assert american_to_prob("+170") == pytest.approx(100 / 270)
    assert american_to_prob("-205") == pytest.approx(205 / 305)
    assert american_to_prob("EVEN") is None and american_to_prob(50) is None
    home, away = devig_moneyline("+170", "-205")
    assert home + away == pytest.approx(1.0) and away == pytest.approx(0.6448, abs=1e-3)


def test_name_matching_handles_kalshi_abbreviations():
    assert name_matches("Washington", "Washington Commanders")
    assert name_matches("New York G", "New York Giants")
    assert not name_matches("New York G", "New York Jets")
    assert name_matches("Chicago WS", "Chicago White Sox")
    assert not name_matches("Chicago WS", "Chicago Cubs")
    assert name_matches("Los Angeles D", "Los Angeles Dodgers")
    assert event_date("KXMLBGAME-26OCT031300CWSCLE").isoformat() == "2026-10-03"


def _game(away, away_name, home, home_name, start="2026-10-04T17:00Z", ml=("+170", "-205")):
    return {"event_id": "1", "short_name": f"{away} @ {home}", "start_time": start,
            "home": {"abbreviation": home, "name": home_name}, "away": {"abbreviation": away, "name": away_name},
            "espn_odds": {"moneyline": {"home": ml[0], "away": ml[1]}}}


def _kalshi_event():
    ms = [Quote("KXNFLGAME-26OCT04INDWAS-WAS", "KXNFLGAME-26OCT04INDWAS", subtitle="Washington",
                status="active", yes_bid=0.34, yes_ask=0.35),
          Quote("KXNFLGAME-26OCT04INDWAS-IND", "KXNFLGAME-26OCT04INDWAS", subtitle="Indianapolis",
                status="active", yes_bid=0.64, yes_ask=0.65)]
    return Event("KXNFLGAME-26OCT04INDWAS", "KXNFLGAME", title="IND vs WAS", markets=ms)


def test_match_game_uses_aliases_and_refuses_ambiguity():
    g = _game("IND", "Indianapolis Colts", "WSH", "Washington Commanders")
    hit = match_game(_kalshi_event(), [g, _game("ARI", "Arizona Cardinals", "NYG", "New York Giants")])
    assert hit and hit[1] == {"KXNFLGAME-26OCT04INDWAS-WAS": "home", "KXNFLGAME-26OCT04INDWAS-IND": "away"}
    assert match_game(_kalshi_event(), [g, dict(g)]) is None  # two candidates: refuse, don't guess
    assert match_game(_kalshi_event(), [_game("IND", "Indianapolis Colts", "WSH", "Washington Commanders",
                                              start="2026-10-05T17:00Z")]) is None  # wrong day


class FakeSports:
    def get_sport_schedule(self, sport):
        return {"data": {"games": [_game("IND", "Indianapolis Colts", "WSH", "Washington Commanders")]}}

    def match_markets(self, sport):
        return {"data": {"matches": [{"kalshi": {"event_ticker": "KXNFLGAME-26OCT04INDWAS"},
                                      "polymarket": {"markets": [{"outcomes": [
                                          {"name": "Colts", "clob_token_id": "tc"},
                                          {"name": "Commanders", "clob_token_id": "tw"}]}]}}]}}


class FakeFetcher:
    def get(self, url, params=None, **kw):
        mid = {"tc": (0.65, 0.66), "tw": (0.34, 0.35)}[params["token_id"]]
        return {"bids": [{"price": str(mid[0]), "size": "100"}], "asks": [{"price": str(mid[1]), "size": "100"}]}


class FakeKalshi:
    def iter_events(self, status, series_ticker):
        return [_kalshi_event()] if series_ticker == "KXNFLGAME" else []


def test_engine_prices_consensus_and_skips_started_games():
    eng = GamesEngine(FakeFetcher(), FakeKalshi(), weight=0.0, now=NOW, sports=FakeSports())
    priced = eng.price_sport("nfl")
    assert len(priced) == 1 and set(priced[0].sources) == {"draftkings", "polymarket"}
    ind = next(f for f in priced[0].fairs if f.ticker.endswith("-IND"))
    assert 0.64 < ind.p_yes < 0.66 and ind.weight == 0.0 and ind.p_market_yes == pytest.approx(0.645, abs=0.01)
    late = GamesEngine(FakeFetcher(), FakeKalshi(), now=datetime(2026, 10, 4, 17, 5, tzinfo=timezone.utc),
                       sports=FakeSports())
    assert late.price_sport("nfl") == []  # kickoff passed: pre-game lines are stale
