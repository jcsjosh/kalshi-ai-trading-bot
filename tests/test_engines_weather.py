"""Weather engine: stations, NBM parsing, the integer-temperature model, calibration."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from src.engines.market import Quote
from src.engines.weather.calibration import Sample, WeatherCalibration
from src.engines.market import devig
from src.engines.trust import fit_weight
from src.engines.weather.backtest import _candle_quote
from src.engines.weather.climate import parse_cli
from src.engines.weather.model import CalibParams, TempDistribution
from src.engines.weather.nbm import NbmIndex, latest_forecast, parse_nbs_csv, target_ftime
from src.engines.weather.obs import summarize_observations
from src.engines.weather.stations import (
    STATIONS,
    date_from_event,
    kind_from_rules,
    station_from_rules,
)

RULES_HIGH = ("If the maximum temperature recorded at New York City (CLINYC) for Oct 2, 2026, is less than "
              "81° fahrenheit according to The Weather Company, then the market resolves to Yes.")


def test_station_and_kind_from_rules():
    code, st = station_from_rules(RULES_HIGH)
    assert code == "NYC" and st.icao == "KNYC"
    assert kind_from_rules(RULES_HIGH) == "high"
    assert kind_from_rules(RULES_HIGH.replace("maximum", "minimum")) == "low"
    assert station_from_rules("no station here") is None
    assert date_from_event("KXHIGHNY-26OCT02") == date(2026, 10, 2)
    assert date_from_event("KXLOWTMIA-26SEP30") == date(2026, 9, 30)


def test_climate_day_is_local_standard_time():
    nyc = STATIONS["NYC"]
    start, end = nyc.climate_day_utc(date(2026, 7, 4))  # EDT in effect; CLI still uses EST
    assert start == datetime(2026, 7, 4, 5, tzinfo=timezone.utc)
    assert end - start == timedelta(days=1)
    # 12:30 AM EDT on Jul 5 is still Jul 4 in standard time.
    assert nyc.climate_date(datetime(2026, 7, 5, 4, 30, tzinfo=timezone.utc)) == date(2026, 7, 4)
    phx = STATIONS["PHX"]  # Arizona: no DST
    assert phx.climate_day_utc(date(2026, 7, 4))[0] == datetime(2026, 7, 4, 7, tzinfo=timezone.utc)


NBS_CSV = """runtime,ftime,model,tmp,txn,xnd,station
2026-09-28 12:00:00,2026-09-29 00:00:00,NBS,66,65.0,2.0,KNYC
2026-09-28 12:00:00,2026-09-29 03:00:00,NBS,62,,,KNYC
2026-09-28 18:00:00,2026-09-29 00:00:00,NBS,64,64.0,1.0,KNYC
2026-09-28 12:00:00,2026-09-29 12:00:00,NBS,58,57.0,2.0,KNYC
"""


def test_nbs_parse_and_as_of_lookup():
    rows = parse_nbs_csv(NBS_CSV)
    assert len(rows) == 3  # rows without txn are dropped
    assert target_ftime(date(2026, 9, 28), "high") == datetime(2026, 9, 29, 0, tzinfo=timezone.utc)
    assert target_ftime(date(2026, 9, 29), "low") == datetime(2026, 9, 29, 12, tzinfo=timezone.utc)
    idx = NbmIndex(rows)
    # The 18Z run is published at 20Z: invisible at 19Z, visible at 20Z.
    early = idx.latest(date(2026, 9, 28), "high", datetime(2026, 9, 28, 19, tzinfo=timezone.utc))
    late = latest_forecast(rows, date(2026, 9, 28), "high", datetime(2026, 9, 28, 20, tzinfo=timezone.utc))
    assert early.txn == 65.0 and late.txn == 64.0
    assert idx.latest(date(2026, 9, 28), "high", datetime(2026, 9, 28, 13, tzinfo=timezone.utc)) is None


def test_cli_parse_handles_missing():
    out = parse_cli([{"valid": "2026-09-30", "high": 74, "low": 60}, {"valid": "2026-10-01", "high": "M", "low": 65}])
    assert out[date(2026, 9, 30)] == (74, 60) and out[date(2026, 10, 1)] == (None, 65)


def _bucket(kind, lo=None, hi=None):
    return Quote(ticker=f"T-{kind}-{lo}-{hi}", event_ticker="E", strike_type=kind, floor_strike=lo, cap_strike=hi)


BUCKETS = [_bucket("less", hi=75), _bucket("between", 75, 76), _bucket("between", 77, 78),
           _bucket("between", 79, 80), _bucket("greater", lo=80)]


def test_distribution_sums_to_one_across_exhaustive_buckets():
    d = TempDistribution.from_forecast(77.3, 2.0, CalibParams(bias=-0.5, scale=1.0, floor=1.0))
    assert sum(d.pmf().values()) == pytest.approx(1.0)
    assert sum(d.prob(b) for b in BUCKETS) == pytest.approx(1.0)
    # mu = 76.8 sits in the 75-76 / 77-78 boundary region
    assert d.prob(BUCKETS[1]) > 0.25 and d.prob(BUCKETS[2]) > 0.25


def test_observed_max_truncates_the_high():
    d = TempDistribution.from_forecast(76.0, 2.0, CalibParams(floor=1.0), lower=79)
    assert d.prob(BUCKETS[0]) == 0.0 and d.prob(BUCKETS[1]) == 0.0
    # 78 survives only as a sliver of rounding slack, even with the forecast centred below it.
    assert d.pmf()[78] < 0.06
    assert sum(d.prob(b) for b in BUCKETS[3:]) > 0.94


def test_observed_min_truncates_the_low():
    d = TempDistribution.from_forecast(60.0, 2.0, CalibParams(floor=1.0), upper=55)
    assert max(d.pmf()) <= 56


def test_non_numeric_market_has_no_opinion():
    d = TempDistribution.from_forecast(70, 2, CalibParams())
    assert d.prob(Quote(ticker="X", event_ticker="E", strike_type="custom")) is None


def test_calibration_recovers_bias_and_station_quirk():
    rng = np.random.default_rng(7)
    samples = []
    start = date(2026, 5, 1)
    for i in range(240):
        for code, quirk in (("NYC", -1.5), ("MDW", 0.0), ("LAX", 1.0)):
            truth = 75 + 10 * np.sin(i / 20)
            xnd = 2.0
            txn = truth - quirk + rng.normal(0, 1.6)
            samples.append(Sample(code, "high", start + timedelta(days=i), round(txn), xnd, int(round(truth))))
    cal = WeatherCalibration.fit(samples)
    assert cal.n["high"] == 720
    nyc, lax = cal.params("NYC", "high"), cal.params("LAX", "high")
    resid = {c: np.mean([s.actual - s.txn for s in samples if s.station == c]) for c in ("NYC", "LAX")}
    # Station bias = its own mean residual, shrunk slightly toward the pool (n=240 vs prior 15).
    assert nyc.bias == pytest.approx(resid["NYC"], abs=0.15) and nyc.bias < -0.8
    assert lax.bias == pytest.approx(resid["LAX"], abs=0.15) and lax.bias > 0.7
    assert 1.0 < nyc.sigma(2.0) < 2.8
    # Low has no data: defaults survive.
    assert cal.params("NYC", "low").scale == pytest.approx(1.2)


def test_calibration_roundtrip(tmp_path):
    cal = WeatherCalibration()
    cal.station_bias = {"high": {"NYC": -1.1}}
    path = cal.save(tmp_path / "c.json")
    assert WeatherCalibration.load(path).params("NYC", "high").bias == -1.1
    assert WeatherCalibration.load(tmp_path / "missing.json").params("NYC", "high").bias == 0.0


def test_observations_summary_respects_window():
    start = datetime(2026, 10, 1, 5, tzinfo=timezone.utc)
    feats = [
        {"properties": {"timestamp": "2026-10-01T04:51:00+00:00", "temperature": {"value": 30.0}}},  # before window
        {"properties": {"timestamp": "2026-10-01T17:51:00+00:00", "temperature": {"value": 23.9}}},
        {"properties": {"timestamp": "2026-10-01T09:51:00+00:00", "temperature": {"value": 18.3}}},
        {"properties": {"timestamp": "2026-10-01T10:51:00+00:00", "temperature": {"value": None}}},
    ]
    o = summarize_observations(feats, start, start + timedelta(days=1))
    assert (o.max_f, o.min_f, o.n_obs) == (75, 65, 2)


def test_observations_ignore_flagged_readings_and_spikes():
    start = datetime(2026, 10, 1, 5, tzinfo=timezone.utc)

    def obs(hour, c, qc="V"):
        return {"properties": {"timestamp": f"2026-10-01T{hour:02d}:51:00+00:00",
                               "temperature": {"value": c, "qualityControl": qc}}}

    day = start + timedelta(days=1)
    # Tenths (METAR precision) so these test glitch handling, not precision widening.
    feats = [obs(12, 20.1), obs(13, 21.1), obs(14, 30.1, qc="X"), obs(15, 21.6)]
    assert summarize_observations(feats, start, day).max_f == 71  # 21.6C; the rejected 30C is dropped
    feats = [obs(12, 20.1), obs(13, 26.1), obs(14, 21.1)]  # isolated 79F between 68F and 70F
    assert summarize_observations(feats, start, day).max_f == 70
    feats = [obs(12, 20.1), obs(13, 21.1), obs(14, 26.1)]  # newest reading jumps: unconfirmed yet
    assert summarize_observations(feats, start, day).max_f == 70
    feats = [obs(9, 12.1), obs(12, 15.1), obs(15, 18.1), obs(18, 21.1)]  # steady 5-6F/3h warming is weather
    assert summarize_observations(feats, start, day).max_f == 70


def test_backtest_helpers():
    at = datetime(2026, 9, 29, 20, tzinfo=timezone.utc)
    ts = int(at.timestamp())
    candles = [
        {"end_period_ts": ts - 3600, "yes_bid": {"close_dollars": "0.30"}, "yes_ask": {"close_dollars": "0.32"}},
        {"end_period_ts": ts, "yes_bid": {"close_dollars": "0.40"}, "yes_ask": {"close_dollars": "0.42"}},
        {"end_period_ts": ts + 3600, "yes_bid": {"close_dollars": "0.90"}, "yes_ask": {"close_dollars": "0.95"}},
    ]
    assert _candle_quote(candles, at) == (0.40, 0.42)  # never the future candle
    qs = [Quote("a", "E", yes_bid=0.40, yes_ask=0.42), Quote("b", "E", yes_bid=0.60, yes_ask=0.62)]
    implied = devig(qs)
    assert sum(implied.values()) == pytest.approx(1.0) and implied["a"] == pytest.approx(0.41 / 1.02)
    # The weight goes to whichever source is informative.
    rows = [{"p_model": 0.9 if y else 0.1, "p_market": 0.5, "outcome": y} for y in [1, 0] * 50]
    assert fit_weight(rows) > 0.9
    rows = [{"p_model": 0.5, "p_market": 0.9 if y else 0.1, "outcome": y} for y in [1, 0] * 50]
    assert fit_weight(rows) < 0.1


def test_whole_degree_celsius_readings_widen_the_bound():
    """Real KHOU data: the 5-minute feed reports whole C. 24 C could be 76.1 F, so the
    CLI low can still be 76 and the 76-77 bucket is live, not ruled out."""
    start = datetime(2026, 10, 2, 6, tzinfo=timezone.utc)
    coarse = [{"properties": {"timestamp": f"2026-10-02T{h:02d}:05:00+00:00",
                              "temperature": {"value": c, "qualityControl": "V"}}}
              for h, c in ((7, 25.0), (8, 24.0), (9, 25.0))]
    o = summarize_observations(coarse, start, start + timedelta(days=1))
    assert o.min_f == 76  # 24.5 C = 76.1 F, not 75
    assert o.max_f == 76  # high is at least 24.5 C = 76.1 F
    precise = [{"properties": {"timestamp": "2026-10-02T08:53:00+00:00", "rawMessage":
                               "KHOU 020853Z 36011KT 10SM BKN023 A2988 RMK AO2 T02390233",
                               "temperature": {"value": 24.0, "qualityControl": "V"}}}]
    assert summarize_observations(precise, start, start + timedelta(days=1)).min_f == 75  # 23.9 C exact


def test_edge_policy_replay_blocks_a_losing_engine():
    from src.engines.weather.backtest import apply_edge_policy

    def t(day, target, pnl, series="KXHIGHNY"):
        return {"decided_at": f"{day}T20:00:00+00:00", "target": target, "pnl": pnl, "series": series,
                "ticker": f"{series}-{target}"}

    losers = [t("2026-08-01", "2026-08-02", -10.0) for _ in range(5)]
    later = [t("2026-08-05", "2026-08-06", 50.0)]
    allowed, blocks = apply_edge_policy(losers + later)
    assert blocks["engine"] == "2026-08-05" and len(allowed) == 5
    # Unsettled losers (target not yet passed) can't block anything.
    allowed, blocks = apply_edge_policy(losers + [t("2026-08-02", "2026-08-03", 5.0)])
    assert "engine" not in blocks and len(allowed) == 6


def test_earned_weight_requires_significance():
    from src.engines.weather.backtest import earned_weight

    rng = np.random.default_rng(3)
    informative, noise = [], []
    for i in range(400):
        y = int(rng.random() < 0.5)
        ev = f"KXHIGHNY-26AUG{i:03d}-B1"
        informative.append({"ticker": ev, "schedule": "day_ahead", "outcome": y,
                            "p_model": 0.8 if y else 0.2, "p_market": 0.5})
        noise.append({"ticker": ev, "schedule": "day_ahead", "outcome": y,
                      "p_model": float(rng.uniform(0.3, 0.7)), "p_market": 0.5})
    w, z = earned_weight(informative)
    assert w > 0.5 and z > 2
    assert earned_weight(noise)[0] == 0.0


def test_observed_extreme_combines_observed_and_rest_of_day():
    from src.engines.weather.model import ObservedExtreme
    from src.engines.weather.nbm import NbmTemp, remaining_extreme

    # Houston low, afternoon: observed ceiling 76 F (exact), rest of day forecast ~85 F.
    rest = TempDistribution(mu=85.0, sigma=2.0)
    d = ObservedExtreme("low", 76, exact=True, remaining=rest)
    assert sum(d.pmf().values()) == pytest.approx(1.0)
    assert d.pmf()[76] > 0.99  # the evening won't come close to 76: the low is set
    # Coarse (whole-C) reading: the true low may sit 1-2 degrees under the ceiling.
    coarse = ObservedExtreme("low", 76, exact=False, remaining=rest)
    mass = {v for v, p in coarse.pmf().items() if p > 1e-3}
    assert mass == {74, 75, 76} and coarse.pmf()[75] == pytest.approx(0.5, abs=1e-3)
    # Morning high: observed floor 70, the afternoon forecast ~80 dominates.
    hi = ObservedExtreme("high", 70, remaining=TempDistribution(mu=80.0, sigma=2.0))
    assert max(hi.pmf(), key=hi.pmf().get) == 80 and hi.pmf().get(69, 0.0) == 0.0
    # No hours left: the observation is the answer.
    assert ObservedExtreme("high", 88).pmf() == {88: 1.0}

    now = datetime(2026, 10, 2, 19, tzinfo=timezone.utc)
    temps = [NbmTemp(datetime(2026, 10, 2, 12, tzinfo=timezone.utc), datetime(2026, 10, 2, h, tzinfo=timezone.utc), t)
             for h, t in ((18, 90.0), (21, 88.0), (23, 84.0))]
    assert remaining_extreme(temps, "low", now, datetime(2026, 10, 3, 6, tzinfo=timezone.utc)) == 84.0
    assert remaining_extreme(temps, "high", now, datetime(2026, 10, 3, 6, tzinfo=timezone.utc)) == 88.0
    assert remaining_extreme(temps, "low", datetime(2026, 10, 3, 1, tzinfo=timezone.utc),
                             datetime(2026, 10, 3, 6, tzinfo=timezone.utc)) is None
